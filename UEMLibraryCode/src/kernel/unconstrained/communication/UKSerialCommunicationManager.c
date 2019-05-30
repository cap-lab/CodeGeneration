/*
 * UKSerialCommunicationManager.c
 *
 *  Created on: 2018. 10. 6.
 *      Author: chjej202
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <UCBasic.h>
#include <UCAlloc.h>
#include <UCFixedSizeQueue.h>
#include <UCThread.h>
#include <UCTime.h>
#include <UCThreadMutex.h>

#include <uem_protocol_data.h>
#include <uem_data.h>

#include <UKUEMLiteProtocol.h>

#include <UKVirtualCommunication.h>

#include <UKSerialCommunicationManager.h>


#define RECEIVER_THREAD_DESTROY_TIMEOUT (3000)

typedef struct _SChannelIdToQueueMap {
	int nChannelId;
	HFixedSizeQueue hChannelQueue;
	void *pTempBuffer;
	int nBufLen;
} SChannelIdToQueueMap;

typedef struct _SSerialCommunicationManager {
	EUemModuleId enId;
	HUEMLiteProtocol hSendData;
	HUEMLiteProtocol hReceiveData;
	HVirtualSocket hSocket;
	SVirtualCommunicationAPI *pstAPI;
	HFixedSizeQueue hSendQueue; // queue (with mutex) (queue size : number of channels between this connection)
	HThread hReceiveHandlingThread;
	SChannelIdToQueueMap *pastChannelQueueTable;
	int nMaxChannelNum;
	int nSetChannelNum;
	HThreadMutex hMutex;
} SSerialCommunicationManager;


static uem_result findChannelIdToQueueMap(SSerialCommunicationManager *pstManager, int nChannelId, OUT SChannelIdToQueueMap **ppstChannelIdToQueueMap)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;
	SChannelIdToQueueMap *pastChannelQueueMap = NULL;

	pastChannelQueueMap = pstManager->pastChannelQueueTable;
	for(nLoop = 0 ; nLoop < pstManager->nMaxChannelNum ; nLoop++)
	{
		if(pastChannelQueueMap[nLoop].nChannelId == nChannelId)
		{
			*ppstChannelIdToQueueMap = &(pastChannelQueueMap[nLoop]);
			break;
		}
	}

	if(nLoop == pstManager->nMaxChannelNum)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static uem_result findChannelQueue(SSerialCommunicationManager *pstManager, int nChannelId, OUT HFixedSizeQueue *phQueue)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;
	SChannelIdToQueueMap *pastChannelQueueMap;

	pastChannelQueueMap = pstManager->pastChannelQueueTable;
	for(nLoop = 0 ; nLoop < pstManager->nMaxChannelNum ; nLoop++)
	{
		if(pastChannelQueueMap[nLoop].nChannelId == nChannelId)
		{
			*phQueue = pastChannelQueueMap[nLoop].hChannelQueue;
			break;
		}
	}

	if(nLoop == pstManager->nMaxChannelNum)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static uem_result passRequestItemToChannel(SSerialCommunicationManager *pstManager, int nChannelId, EMessageType enMessageType, int nParamNum, short *pastParam)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCommunicationQueueItem stItem;
	HFixedSizeQueue hChannelQueue = NULL;

	stItem.nChannelId = nChannelId;
	stItem.enMessageType = enMessageType;
	stItem.uDetailItem.stRequest.nRequestDataSize = 0;

	switch(enMessageType)
	{
	case MESSAGE_TYPE_READ_QUEUE:
		stItem.uDetailItem.stRequest.nRequestDataSize = pastParam[READ_QUEUE_SIZE_TO_READ_INDEX];
		break;
	case MESSAGE_TYPE_READ_BUFFER:
		stItem.uDetailItem.stRequest.nRequestDataSize = pastParam[READ_BUFFER_SIZE_TO_READ_INDEX];
		break;
	case MESSAGE_TYPE_AVAILABLE_DATA:
		// do nothing
		break;
	case MESSAGE_TYPE_RESULT:
	case MESSAGE_TYPE_HANDSHAKE:
	case MESSAGE_TYPE_AVAILABLE_INDEX:
	default:
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
		break;
	}

	result = findChannelQueue(pstManager, nChannelId, &hChannelQueue);
	ERRIFGOTO(result, _EXIT);

	result = UCFixedSizeQueue_PutItem(hChannelQueue, (void *) &stItem, sizeof(SCommunicationQueueItem));
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static uem_result reallocTempBuffer(SChannelIdToQueueMap *pstChannelIdToQueueMap, int nTargetSize)
{
	uem_result result = ERR_UEM_UNKNOWN;

	if(pstChannelIdToQueueMap->nBufLen < nTargetSize)
	{
		SAFEMEMFREE(pstChannelIdToQueueMap->pTempBuffer);

		pstChannelIdToQueueMap->pTempBuffer = UCAlloc_malloc(nTargetSize);
		ERRMEMGOTO(pstChannelIdToQueueMap->pTempBuffer, result, _EXIT);

		pstChannelIdToQueueMap->nBufLen = nTargetSize;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static uem_result passResultItemToChannel(SSerialCommunicationManager *pstManager, int nChannelId, EMessageType enMessageType,
											int nParamNum, short *pastParam)
{
	uem_result result = ERR_UEM_UNKNOWN;
	void *pBodyData = NULL;
	SChannelIdToQueueMap *pstChannelIdToQueueMap = NULL;
	SCommunicationQueueItem stItem;

	stItem.enMessageType = enMessageType;
	stItem.nChannelId = nChannelId;
	stItem.uDetailItem.stResponse.enRequestMessageType = (EMessageType) pastParam[RESULT_REQUEST_PACKET_INDEX];

	result = findChannelIdToQueueMap(pstManager, nChannelId, &pstChannelIdToQueueMap);
	ERRIFGOTO(result, _EXIT);

	switch(stItem.uDetailItem.stResponse.enRequestMessageType)
	{
	case MESSAGE_TYPE_READ_QUEUE:
	case MESSAGE_TYPE_READ_BUFFER:
		result = UKUEMLiteProtocol_GetBodyDataFromReceivedData(pstManager->hReceiveData,
																&(stItem.uDetailItem.stResponse.nDataSize), &pBodyData);
		ERRIFGOTO(result, _EXIT);

		result = reallocTempBuffer(pstChannelIdToQueueMap, stItem.uDetailItem.stResponse.nDataSize);
		ERRIFGOTO(result, _EXIT);

		UC_memcpy(pstChannelIdToQueueMap->pTempBuffer, pBodyData, stItem.uDetailItem.stResponse.nDataSize);
		stItem.uDetailItem.stResponse.pData = pstChannelIdToQueueMap->pTempBuffer;
		break;
	case MESSAGE_TYPE_AVAILABLE_DATA:
		stItem.uDetailItem.stResponse.nReturnValue = (int) pastParam[RESULT_RETURN_VALUE_INDEX];
		break;
	case MESSAGE_TYPE_RESULT:
	case MESSAGE_TYPE_HANDSHAKE:
	case MESSAGE_TYPE_AVAILABLE_INDEX:
	default:
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
		break;
	}

	result = UCFixedSizeQueue_PutItem(pstChannelIdToQueueMap->hChannelQueue, (void *) &stItem, sizeof(SCommunicationQueueItem));
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static uem_result handleReceivedMessage(SSerialCommunicationManager *pstManager)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nChannelId;
	EMessageType enMessageType;
	int nParamNum;
	short *pasParam = NULL;

	result = UKUEMLiteProtocol_GetHeaderFromReceivedData(pstManager->hReceiveData, &nChannelId, &enMessageType, &nParamNum, &pasParam);
	ERRIFGOTO(result, _EXIT);

	switch(enMessageType)
	{
	case MESSAGE_TYPE_READ_QUEUE:
	case MESSAGE_TYPE_READ_BUFFER:
	case MESSAGE_TYPE_AVAILABLE_DATA:
		result = passRequestItemToChannel(pstManager, nChannelId, enMessageType, nParamNum, pasParam);
		ERRIFGOTO(result, _EXIT);
		break;
	case MESSAGE_TYPE_RESULT:
		result = passResultItemToChannel(pstManager, nChannelId, enMessageType, nParamNum, pasParam);
		ERRIFGOTO(result, _EXIT);
		break;
	case MESSAGE_TYPE_HANDSHAKE:
	case MESSAGE_TYPE_AVAILABLE_INDEX:
	default:
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
		break;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static void *receiveHandlingThread(void *pData)
{
	SSerialCommunicationManager *pstManager = NULL;
	uem_result result = ERR_UEM_UNKNOWN;


	pstManager = (SSerialCommunicationManager *) pData;

	while(g_bSystemExit == FALSE || pstManager->nSetChannelNum > 0)
	{
		result = UKUEMLiteProtocol_Receive(pstManager->hReceiveData);
		if(result == ERR_UEM_NET_TIMEOUT)
		{
			continue;
		}
		ERRIFGOTO(result, _EXIT);

		result = handleReceivedMessage(pstManager);
		ERRIFGOTO(result, _EXIT);
	}

_EXIT:
	if(result != ERR_UEM_NOERROR && g_bSystemExit == FALSE)
	{
		UEM_DEBUG_PRINT("serial communication manager receiveHandlingThread is exited with error: %08x\n", result);
	}
	return NULL;
}



uem_result UKSerialCommunicationManager_Create(HVirtualSocket hSocket, SVirtualCommunicationAPI *pstAPI, int nMaxChannelNum, OUT HSerialCommunicationManager *phManager)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSerialCommunicationManager *pstManager = NULL;
	int nLoop = 0;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(phManager, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	pstManager = UCAlloc_malloc(sizeof(SSerialCommunicationManager));
	ERRMEMGOTO(pstManager, result, _EXIT);

	pstManager->enId = ID_UEM_SERIAL_COMMUNICATION_MANAGER;
	pstManager->hReceiveData = NULL;
	pstManager->hSendData = NULL;
	pstManager->hSendQueue = NULL;
	pstManager->hReceiveHandlingThread = NULL;
	pstManager->pastChannelQueueTable = NULL;
	pstManager->nMaxChannelNum = nMaxChannelNum;
	pstManager->nSetChannelNum = 0;
	pstManager->hMutex = NULL;

	pstManager->pastChannelQueueTable = UCAlloc_malloc(nMaxChannelNum * sizeof(SChannelIdToQueueMap));
	ERRMEMGOTO(pstManager->pastChannelQueueTable, result, _EXIT);

	for(nLoop = 0 ; nLoop < nMaxChannelNum ; nLoop++)
	{
		pstManager->pastChannelQueueTable[nLoop].nChannelId = INVALID_CHANNEL_ID;
		pstManager->pastChannelQueueTable[nLoop].hChannelQueue = NULL;
		pstManager->pastChannelQueueTable[nLoop].pTempBuffer = NULL;
		pstManager->pastChannelQueueTable[nLoop].nBufLen = 0;
	}

	for(nLoop = 0 ; nLoop < nMaxChannelNum ; nLoop++) // each channel's queue
	{
		result = UCFixedSizeQueue_Create(sizeof(SCommunicationQueueItem), 1, &(pstManager->pastChannelQueueTable[nLoop].hChannelQueue));
		ERRIFGOTO(result, _EXIT);
	}

	result = UCFixedSizeQueue_Create(sizeof(SCommunicationQueueItem), nMaxChannelNum, &(pstManager->hSendQueue));
	ERRIFGOTO(result, _EXIT);

	result = UKUEMLiteProtocol_Create(&(pstManager->hReceiveData));
	ERRIFGOTO(result, _EXIT);

	result = UKUEMLiteProtocol_SetVirtualSocket(pstManager->hReceiveData, hSocket, pstAPI);
	ERRIFGOTO(result, _EXIT);

	result = UKUEMLiteProtocol_Create(&(pstManager->hSendData));
	ERRIFGOTO(result, _EXIT);

	result = UKUEMLiteProtocol_SetVirtualSocket(pstManager->hSendData, hSocket, pstAPI);
	ERRIFGOTO(result, _EXIT);

	result = UCThreadMutex_Create(&(pstManager->hMutex));
	ERRIFGOTO(result, _EXIT);

	*phManager = pstManager;

	result = ERR_UEM_NOERROR;
_EXIT:
	if(result != ERR_UEM_NOERROR && pstManager != NULL)
	{
		UKSerialCommunicationManager_Destroy((HSerialCommunicationManager *)&pstManager);
	}
	return result;
}


uem_result UKSerialCommunicationManager_Destroy(IN OUT HSerialCommunicationManager *phManager)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSerialCommunicationManager *pstManager = NULL;
	int nLoop = 0;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(phManager, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	if(IS_VALID_HANDLE(*phManager, ID_UEM_SERIAL_COMMUNICATION_MANAGER) == FALSE)
	{
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}
#endif
	pstManager = *phManager;

	if(pstManager->hMutex != NULL)
	{
		UCThreadMutex_Destroy(&(pstManager->hMutex));
	}

	if(pstManager->hSendQueue != NULL)
	{
		UCFixedSizeQueue_Destroy(&(pstManager->hSendQueue));
	}

	if(pstManager->hReceiveData != NULL)
	{
		UKUEMLiteProtocol_Destroy(&(pstManager->hReceiveData));
	}

	if(pstManager->hSendData != NULL )
	{
		UKUEMLiteProtocol_Destroy(&(pstManager->hSendData));
	}

	if(pstManager->pastChannelQueueTable != NULL)
	{
		for(nLoop = 0 ; nLoop < pstManager->nMaxChannelNum ; nLoop++)
		{
			if(pstManager->pastChannelQueueTable[nLoop].hChannelQueue != NULL)
			{
				UCFixedSizeQueue_Destroy(&(pstManager->pastChannelQueueTable[nLoop].hChannelQueue));
			}
			SAFEMEMFREE(pstManager->pastChannelQueueTable[nLoop].pTempBuffer);
		}
	}

	SAFEMEMFREE(pstManager->pastChannelQueueTable);

	SAFEMEMFREE(pstManager);

	*phManager = NULL;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static uem_result setResultMessageWithError(SSerialCommunicationManager *pstManager, int nChannelId, SResponseItem *pstResponseItem)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = UKUEMLiteProtocol_SetResultMessage(pstManager->hSendData, pstResponseItem->enRequestMessageType,
													nChannelId, ERR_UEMPROTOCOL_INTERNAL, pstResponseItem->nReturnValue);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static uem_result setResultMessage(SSerialCommunicationManager *pstManager, int nChannelId, SResponseItem *pstResponseItem)
{
	uem_result result = ERR_UEM_UNKNOWN;

	switch(pstResponseItem->enRequestMessageType)
	{
	case MESSAGE_TYPE_READ_QUEUE:
	case MESSAGE_TYPE_READ_BUFFER:
		result = UKUEMLiteProtocol_SetResultMessageHeaderUsingBuffer(pstManager->hSendData,
														pstResponseItem->enRequestMessageType, nChannelId,
														ERR_UEMPROTOCOL_NOERROR, pstResponseItem->nDataSize,
														pstResponseItem->pData);
		ERRIFGOTO(result, _EXIT);
		break;
	case MESSAGE_TYPE_AVAILABLE_DATA:
		result = UKUEMLiteProtocol_SetResultMessage(pstManager->hSendData, pstResponseItem->enRequestMessageType,
														nChannelId, ERR_UEMPROTOCOL_NOERROR, pstResponseItem->nReturnValue);
		ERRIFGOTO(result, _EXIT);
		break;
	case MESSAGE_TYPE_HANDSHAKE:
	case MESSAGE_TYPE_RESULT:
	case MESSAGE_TYPE_AVAILABLE_INDEX:
	default:
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
		break;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static uem_result handleDataFromQueue(SSerialCommunicationManager *pstManager, SCommunicationQueueItem *pstItem)
{
	uem_result result = ERR_UEM_UNKNOWN;

	switch(pstItem->enMessageType)
	{
	case MESSAGE_TYPE_READ_QUEUE:
		result = UKUEMLiteProtocol_SetReadQueueRequest(pstManager->hSendData, pstItem->nChannelId,
															pstItem->uDetailItem.stRequest.nRequestDataSize);
		ERRIFGOTO(result, _EXIT);
		break;
	case MESSAGE_TYPE_READ_BUFFER:
		result = UKUEMLiteProtocol_SetReadBufferRequest(pstManager->hSendData, pstItem->nChannelId,
															pstItem->uDetailItem.stRequest.nRequestDataSize);
		ERRIFGOTO(result, _EXIT);
		break;
	case MESSAGE_TYPE_AVAILABLE_DATA:
		result = UKUEMLiteProtocol_SetAvailableDataRequest(pstManager->hSendData, pstItem->nChannelId);
		ERRIFGOTO(result, _EXIT);
		break;
	case MESSAGE_TYPE_RESULT:
		result = setResultMessage(pstManager, pstItem->nChannelId, &(pstItem->uDetailItem.stResponse));
		ERRIFGOTO(result, _EXIT);
		break;
	case MESSAGE_TYPE_NONE:
		result = setResultMessageWithError(pstManager, pstItem->nChannelId, &(pstItem->uDetailItem.stResponse));
		ERRIFGOTO(result, _EXIT);
		break;
	case MESSAGE_TYPE_HANDSHAKE:
	case MESSAGE_TYPE_AVAILABLE_INDEX:
	default:
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
		break;
	}

	result = UKUEMLiteProtocol_Send(pstManager->hSendData);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKSerialCommunicationManager_Run(HSerialCommunicationManager hManager)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSerialCommunicationManager *pstManager = NULL;
	SCommunicationQueueItem stItem;
	int nElementSize = 0;
#ifdef ARGUMENT_CHECK
	if(IS_VALID_HANDLE(hManager, ID_UEM_SERIAL_COMMUNICATION_MANAGER) == FALSE)
	{
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}
#endif
	pstManager = (SSerialCommunicationManager *) hManager;

    while(pstManager->nSetChannelNum < pstManager->nMaxChannelNum)
    {   
        UCTime_Sleep(10);
    }   

	result = UCThread_Create(receiveHandlingThread, pstManager, &(pstManager->hReceiveHandlingThread));
	ERRIFGOTO(result, _EXIT);

	while(g_bSystemExit == FALSE || pstManager->nSetChannelNum > 0)
	{
		result = UCFixedSizeQueue_GetItem(pstManager->hSendQueue, (void *)&stItem, &nElementSize);
		if(result == ERR_UEM_TIME_EXPIRED)
		{
			continue;
		}
		ERRIFGOTO(result, _EXIT);

		result = handleDataFromQueue(pstManager, &stItem);
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	if(result != ERR_UEM_NOERROR)
	{
		UEM_DEBUG_PRINT("serial communication manager main thread is exited with error: %08x\n", result);
	}
	if(pstManager->hReceiveHandlingThread != NULL)
	{
		UCThread_Destroy(&(pstManager->hReceiveHandlingThread), FALSE, RECEIVER_THREAD_DESTROY_TIMEOUT);
	}

	return result;
}


uem_result UKSerialCommunicationManager_PutItemToSend(HSerialCommunicationManager hManager, SCommunicationQueueItem *pstItem)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSerialCommunicationManager *pstSerialManager = NULL;
#ifdef ARGUMENT_CHECK
	if(IS_VALID_HANDLE(hManager, ID_UEM_SERIAL_COMMUNICATION_MANAGER) == FALSE)
	{
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}
#endif
	pstSerialManager = (SSerialCommunicationManager *) hManager;

	result = UCFixedSizeQueue_PutItem(pstSerialManager->hSendQueue, pstItem, sizeof(SCommunicationQueueItem));
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKSerialCommunicationManager_SetChannel(HSerialCommunicationManager hManager, int nChannelId)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSerialCommunicationManager *pstManager = NULL;
#ifdef ARGUMENT_CHECK
	if(IS_VALID_HANDLE(hManager, ID_UEM_SERIAL_COMMUNICATION_MANAGER) == FALSE)
	{
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}
#endif
	pstManager = (SSerialCommunicationManager *) hManager;

	result = UCThreadMutex_Lock(pstManager->hMutex);
	ERRIFGOTO(result, _EXIT);

	if(pstManager->nSetChannelNum == pstManager->nMaxChannelNum)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT_LOCK);
	}

	pstManager->pastChannelQueueTable[pstManager->nSetChannelNum].nChannelId = nChannelId;
	pstManager->nSetChannelNum++;

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
	UCThreadMutex_Unlock(pstManager->hMutex);
_EXIT:
	return result;
}


uem_result UKSerialCommunicationManager_ReleaseChannel(HSerialCommunicationManager hManager, int nChannelId)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSerialCommunicationManager *pstManager = NULL;
	int nLoop = 0;
#ifdef ARGUMENT_CHECK
	if(IS_VALID_HANDLE(hManager, ID_UEM_SERIAL_COMMUNICATION_MANAGER) == FALSE)
	{
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}
#endif
	pstManager = (SSerialCommunicationManager *) hManager;

	result = UCThreadMutex_Lock(pstManager->hMutex);
	ERRIFGOTO(result, _EXIT);

	for(nLoop = 0; nLoop < pstManager->nMaxChannelNum ; nLoop++)
	{
		if(pstManager->pastChannelQueueTable[nLoop].nChannelId == nChannelId)
		{
			pstManager->pastChannelQueueTable[nLoop].nChannelId = INVALID_CHANNEL_ID;
			pstManager->nSetChannelNum--;
			break;
		}
	}

	result = UCThreadMutex_Unlock(pstManager->hMutex);
	ERRIFGOTO(result, _EXIT);

	if(nLoop == pstManager->nMaxChannelNum)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKSerialCommunicationManager_GetChannelQueue(HSerialCommunicationManager hManager, int nChannelId, HFixedSizeQueue *phQueue)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSerialCommunicationManager *pstManager = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(phQueue, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	if(IS_VALID_HANDLE(hManager, ID_UEM_SERIAL_COMMUNICATION_MANAGER) == FALSE)
	{
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}
#endif
	pstManager = (SSerialCommunicationManager *) hManager;

	result = findChannelQueue(pstManager, nChannelId, phQueue);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}



uem_result UKSerialCommunicationManager_Handshake(HSerialCommunicationManager hManager)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSerialCommunicationManager *pstManager = NULL;
#ifdef ARGUMENT_CHECK
	if(IS_VALID_HANDLE(hManager, ID_UEM_SERIAL_COMMUNICATION_MANAGER) == FALSE)
	{
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}
#endif
	pstManager = (SSerialCommunicationManager *) hManager;

	result = UKUEMLiteProtocol_HandShake(pstManager->hSendData, pstManager->hReceiveData, 0);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKSerialCommunicationManager_AcceptHandshake(HSerialCommunicationManager hManager)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSerialCommunicationManager *pstManager = NULL;
	EMessageType enMessageType = MESSAGE_TYPE_NONE;
#ifdef ARGUMENT_CHECK
	if(IS_VALID_HANDLE(hManager, ID_UEM_SERIAL_COMMUNICATION_MANAGER) == FALSE)
	{
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}
#endif
	pstManager = (SSerialCommunicationManager *) hManager;

	// if it cannot receive anything until timeout, handle as an error
	result = UKUEMLiteProtocol_Receive(pstManager->hReceiveData);
	ERRIFGOTO(result, _EXIT);

	// TODO: deal something with device key
	result = UKUEMLiteProtocol_GetHeaderFromReceivedData(pstManager->hReceiveData, NULL, &enMessageType, NULL, NULL);
	ERRIFGOTO(result, _EXIT);
	if(enMessageType != MESSAGE_TYPE_HANDSHAKE)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
	}

	result = UKUEMLiteProtocol_SetResultMessage(pstManager->hSendData, MESSAGE_TYPE_HANDSHAKE, 0, ERR_UEMPROTOCOL_NOERROR, 0);
	ERRIFGOTO(result, _EXIT);

	result = UKUEMLiteProtocol_Send(pstManager->hSendData);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


