/*
 * UKRemoteChannel.c
 *
 *  Created on: 2019. 5. 28.
 *      Author: jej
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif


#include <uem_common.h>

#include <UCTime.h>
#include <UCAlloc.h>
#include <UCThreadMutex.h>

#include <uem_data.h>

#include <UKChannelMemory.h>

#include <UKUEMProtocol.h>
#include <UKRemoteChannel.h>

#include <uem_remote_data.h>

#define CONNECT_TIMEOUT (3)
#define CONNECT_RETRY_COUNT (100)
#define SECOND_IN_MILLISECOND (1000)

#define REMOTE_REQUEST_WAIT_SLEEP_TIME (100)
#define REMOTE_REQUEST_WAIT_RETRY_COUNT (30)

uem_result UKRemoteChannel_Clear(SChannel *pstChannel)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SRemoteWriterChannel *pstWriterChannel = NULL;

	switch(pstChannel->enType)
	{
	case COMMUNICATION_TYPE_REMOTE_READER:
		// do nothing
		break;
	case COMMUNICATION_TYPE_REMOTE_WRITER:
		pstWriterChannel = (SRemoteWriterChannel *) pstChannel->pChannelStruct;
		result = UKChannelMemory_Clear(pstChannel, pstWriterChannel->pstInternalChannel);
		ERRIFGOTO(result, _EXIT);
		break;
	default:
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
		break;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result reallocTempBuffer(SRemoteWriterChannel *pstWriterChannel, int nTargetSize)
{
	uem_result result = ERR_UEM_UNKNOWN;

	if(pstWriterChannel->nBufLen < nTargetSize)
	{
		SAFEMEMFREE(pstWriterChannel->pBuffer);

		pstWriterChannel->pBuffer = UCAlloc_malloc(nTargetSize);
		ERRMEMGOTO(pstWriterChannel->pBuffer, result, _EXIT);

		pstWriterChannel->nBufLen = nTargetSize;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result handleReadQueueFromPacket(SChannel *pstChannel, int *panParam, int nParamNum, HUEMProtocol hProtocol)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SRemoteWriterChannel *pstWriterChannel = NULL;
	int nDataToRead = 0;
	int nChunkIndex = 0;
	int nDataRead = 0;
	int nChannelId = INVALID_CHANNEL_ID;

	pstWriterChannel = (SRemoteWriterChannel *) pstChannel->pChannelStruct;

	nChannelId = panParam[READ_QUEUE_CHANNEL_ID_INDEX];

	if(pstChannel->nChannelIndex != nChannelId)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
	}

	nDataToRead = panParam[READ_QUEUE_SIZE_TO_READ_INDEX];
	nChunkIndex = panParam[READ_QUEUE_CHUNK_INDEX_INDEX];

	result = reallocTempBuffer(pstWriterChannel, nDataToRead);
	ERRIFGOTO(result, _EXIT);

	result = UKChannelMemory_ReadFromQueue(pstChannel, pstWriterChannel->pstInternalChannel, (unsigned char *)pstWriterChannel->pBuffer, nDataToRead, nChunkIndex, &nDataRead);
	ERRIFGOTO(result, _EXIT);

	result = UKUEMProtocol_SetResultMessageWithBuffer(hProtocol, ERR_UEMPROTOCOL_NOERROR, nDataRead, pstWriterChannel->pBuffer);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result handleReadBufferFromPacket(SChannel *pstChannel, int *panParam, int nParamNum, HUEMProtocol hProtocol)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SRemoteWriterChannel *pstWriterChannel = NULL;
	int nDataToRead = 0;
	int nChunkIndex = 0;
	int nDataRead = 0;
	int nChannelId = INVALID_CHANNEL_ID;

	pstWriterChannel = (SRemoteWriterChannel *) pstChannel->pChannelStruct;

	nChannelId = panParam[READ_BUFFER_CHANNEL_ID_INDEX];

	if(pstChannel->nChannelIndex != nChannelId)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
	}

	nDataToRead = panParam[READ_BUFFER_SIZE_TO_READ_INDEX];
	nChunkIndex = panParam[READ_BUFFER_CHUNK_INDEX_INDEX];

	result = reallocTempBuffer(pstWriterChannel, nDataToRead);
	ERRIFGOTO(result, _EXIT);

	result = UKChannelMemory_ReadFromBuffer(pstChannel, pstWriterChannel->pstInternalChannel, (unsigned char *)pstWriterChannel->pBuffer, nDataToRead, nChunkIndex, &nDataRead);
	ERRIFGOTO(result, _EXIT);

	result = UKUEMProtocol_SetResultMessageWithBuffer(hProtocol, ERR_UEMPROTOCOL_NOERROR, nDataRead, pstWriterChannel->pBuffer);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result handleAvailableIndexFromPacket(SChannel *pstChannel, int *panParam, int nParamNum, HUEMProtocol hProtocol)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SRemoteWriterChannel *pstWriterChannel = NULL;
	int nChunkIndex = 0;
	int nChannelId = INVALID_CHANNEL_ID;

	pstWriterChannel = (SRemoteWriterChannel *) pstChannel->pChannelStruct;

	nChannelId = panParam[AVAILABLE_INDEX_CHANNEL_ID_INDEX];

	if(pstChannel->nChannelIndex != nChannelId)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
	}

	result = UKChannelMemory_GetAvailableChunk(pstChannel, pstWriterChannel->pstInternalChannel, &nChunkIndex);
	ERRIFGOTO(result, _EXIT);

	result = UKUEMProtocol_SetResultMessage(hProtocol, ERR_UEMPROTOCOL_NOERROR, nChunkIndex);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result handleAvailableDataFromPacket(SChannel *pstChannel, int *panParam, int nParamNum, HUEMProtocol hProtocol)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SRemoteWriterChannel *pstWriterChannel = NULL;
	int nChunkIndex = 0;
	int nChannelId = INVALID_CHANNEL_ID;
	int nDataNum = 0;

	pstWriterChannel = (SRemoteWriterChannel *) pstChannel->pChannelStruct;

	nChannelId = panParam[AVAILABLE_DATA_CHANNEL_ID_INDEX];

	if(pstChannel->nChannelIndex != nChannelId)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
	}

	nChunkIndex = panParam[AVAILABLE_DATA_CHUNK_INDEX_INDEX];

	result = UKChannelMemory_GetNumOfAvailableData(pstChannel, pstWriterChannel->pstInternalChannel, nChunkIndex, &nDataNum);
	ERRIFGOTO(result, _EXIT);

	result = UKUEMProtocol_SetResultMessage(hProtocol, ERR_UEMPROTOCOL_NOERROR, nDataNum);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result handleReadQueueFromAggregator(SChannel *pstChannel, SCommunicationQueueItem *pstReceivedItem)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SRemoteWriterChannel *pstWriterChannel = NULL;
	SAggregateConnectionInfo *pstConnectionInfo = NULL;
	SCommunicationQueueItem stItemToSend;
	int nDataToRead = 0;
	int nDataRead = 0;

	pstWriterChannel = (SRemoteWriterChannel *) pstChannel->pChannelStruct;
	pstConnectionInfo = (SAggregateConnectionInfo *) pstWriterChannel->stCommonInfo.pConnectionInfo;

	stItemToSend.enMessageType = MESSAGE_TYPE_RESULT;
	stItemToSend.nChannelId = pstChannel->nChannelIndex;
	stItemToSend.uDetailItem.stResponse.enRequestMessageType = pstReceivedItem->enMessageType;
	stItemToSend.uDetailItem.stResponse.nReturnValue = 0;
	stItemToSend.uDetailItem.stResponse.pData = NULL;
	stItemToSend.uDetailItem.stResponse.nDataSize = 0;

	nDataToRead = pstReceivedItem->uDetailItem.stRequest.nRequestDataSize;

	result = reallocTempBuffer(pstWriterChannel, nDataToRead);
	ERRIFGOTO(result, _EXIT);

	result = UKChannelMemory_ReadFromQueue(pstChannel, pstWriterChannel->pstInternalChannel, (unsigned char *)pstWriterChannel->pBuffer, nDataToRead, 0, &nDataRead);
	ERRIFGOTO(result, _EXIT);

	stItemToSend.uDetailItem.stResponse.pData = pstWriterChannel->pBuffer;
	stItemToSend.uDetailItem.stResponse.nDataSize = nDataRead;
	stItemToSend.uDetailItem.stResponse.nReturnValue = nDataRead;

	result = UKSerialCommunicationManager_PutItemToSend(pstConnectionInfo->pstServiceInfo->hManager, &stItemToSend);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static uem_result handleReadBufferFromAggregator(SChannel *pstChannel, SCommunicationQueueItem *pstReceivedItem)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SRemoteWriterChannel *pstWriterChannel = NULL;
	SAggregateConnectionInfo *pstConnectionInfo = NULL;
	SCommunicationQueueItem stItemToSend;
	int nDataToRead = 0;
	int nDataRead = 0;

	pstWriterChannel = (SRemoteWriterChannel *) pstChannel->pChannelStruct;
	pstConnectionInfo = (SAggregateConnectionInfo *) pstWriterChannel->stCommonInfo.pConnectionInfo;

	stItemToSend.enMessageType = MESSAGE_TYPE_RESULT;
	stItemToSend.nChannelId = pstChannel->nChannelIndex;
	stItemToSend.uDetailItem.stResponse.enRequestMessageType = pstReceivedItem->enMessageType;
	stItemToSend.uDetailItem.stResponse.nReturnValue = 0;
	stItemToSend.uDetailItem.stResponse.pData = NULL;
	stItemToSend.uDetailItem.stResponse.nDataSize = 0;

	nDataToRead = pstReceivedItem->uDetailItem.stRequest.nRequestDataSize;

	result = reallocTempBuffer(pstWriterChannel, nDataToRead);
	ERRIFGOTO(result, _EXIT);

	result = UKChannelMemory_ReadFromBuffer(pstChannel, pstWriterChannel->pstInternalChannel, (unsigned char *)pstWriterChannel->pBuffer, nDataToRead, 0, &nDataRead);
	ERRIFGOTO(result, _EXIT);

	stItemToSend.uDetailItem.stResponse.pData = pstWriterChannel->pBuffer;
	stItemToSend.uDetailItem.stResponse.nDataSize = nDataRead;
	stItemToSend.uDetailItem.stResponse.nReturnValue = nDataRead;

	result = UKSerialCommunicationManager_PutItemToSend(pstConnectionInfo->pstServiceInfo->hManager, &stItemToSend);
	ERRIFGOTO(result, _EXIT);


	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result handleAvailableDataFromAggregator(SChannel *pstChannel, SCommunicationQueueItem *pstReceivedItem)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SRemoteWriterChannel *pstWriterChannel = NULL;
	SAggregateConnectionInfo *pstConnectionInfo = NULL;
	int nDataNum = 0;
	SCommunicationQueueItem stItemToSend;

	pstWriterChannel = (SRemoteWriterChannel *) pstChannel->pChannelStruct;
	pstConnectionInfo = (SAggregateConnectionInfo*) pstWriterChannel->stCommonInfo.pConnectionInfo;

	stItemToSend.enMessageType = MESSAGE_TYPE_RESULT;
	stItemToSend.nChannelId = pstChannel->nChannelIndex;
	stItemToSend.uDetailItem.stResponse.enRequestMessageType = pstReceivedItem->enMessageType;
	stItemToSend.uDetailItem.stResponse.nReturnValue = 0;
	stItemToSend.uDetailItem.stResponse.pData = NULL;
	stItemToSend.uDetailItem.stResponse.nDataSize = 0;

	result = UKChannelMemory_GetNumOfAvailableData(pstChannel, pstWriterChannel->pstInternalChannel, 0, &nDataNum);
	ERRIFGOTO(result, _EXIT);

	stItemToSend.uDetailItem.stResponse.nReturnValue = nDataNum;

	result = UKSerialCommunicationManager_PutItemToSend(pstConnectionInfo->pstServiceInfo->hManager, &stItemToSend);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result handleQueueRequestFromReader(SCommunicationQueueItem *pstReceivedItem, SChannel *pstChannel)
{
	uem_result result = ERR_UEM_UNKNOWN;

	if(pstChannel->nChannelIndex != pstReceivedItem->nChannelId)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
	}

	switch(pstReceivedItem->enMessageType)
	{
	case MESSAGE_TYPE_READ_QUEUE:
		result = handleReadQueueFromAggregator(pstChannel, pstReceivedItem);
		ERRIFGOTO(result, _EXIT);
		break;
	case MESSAGE_TYPE_READ_BUFFER:
		result = handleReadBufferFromAggregator(pstChannel, pstReceivedItem);
		ERRIFGOTO(result, _EXIT);
		break;
	case MESSAGE_TYPE_AVAILABLE_DATA:
		result = handleAvailableDataFromAggregator(pstChannel, pstReceivedItem);
		ERRIFGOTO(result, _EXIT);
		break;
	case MESSAGE_TYPE_AVAILABLE_INDEX:
	default:
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
		break;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result handlePacketRequestFromReader(HUEMProtocol hProtocol, SChannel *pstChannel)
{
	uem_result result = ERR_UEM_UNKNOWN;
	EMessageType enMessageType = MESSAGE_TYPE_NONE;
	int nParamNum = 0;
	int *panParam = NULL;

	result = UKUEMProtocol_GetRequestFromReceivedData(hProtocol, &enMessageType, &nParamNum, &panParam);
	ERRIFGOTO(result, _EXIT);

	switch(enMessageType)
	{
	case MESSAGE_TYPE_READ_QUEUE:
		result = handleReadQueueFromPacket(pstChannel, panParam, nParamNum, hProtocol);
		ERRIFGOTO(result, _EXIT);
		break;
	case MESSAGE_TYPE_READ_BUFFER:
		result = handleReadBufferFromPacket(pstChannel, panParam, nParamNum, hProtocol);
		ERRIFGOTO(result, _EXIT);
		break;
	case MESSAGE_TYPE_AVAILABLE_INDEX:
		result = handleAvailableIndexFromPacket(pstChannel, panParam, nParamNum, hProtocol);
		ERRIFGOTO(result, _EXIT);
		break;
	case MESSAGE_TYPE_AVAILABLE_DATA:
		result = handleAvailableDataFromPacket(pstChannel, panParam, nParamNum, hProtocol);
		ERRIFGOTO(result, _EXIT);
		break;
	default:
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
		break;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

// receiver thread for individual connection
static void *packetReceiverThread(void *pData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SChannel *pstChannel = NULL;
	SRemoteWriterChannel *pstWriterChannel = NULL;
	SIndividualConnectionInfo *pstConnectionInfo = NULL;
	HUEMProtocol hProtocol = NULL;

	pstChannel = (SChannel *) pData;
	pstWriterChannel = (SRemoteWriterChannel *) pstChannel->pChannelStruct;

	pstConnectionInfo = pstWriterChannel->stCommonInfo.pConnectionInfo;

	hProtocol = pstConnectionInfo->hProtocol;

	while(pstWriterChannel->stCommonInfo.bChannelExit == FALSE)
	{
		result = UKUEMProtocol_Receive(hProtocol);
		if(result == ERR_UEM_NET_TIMEOUT)
		{
			continue;
		}
		ERRIFGOTO(result, _EXIT);

		result = handlePacketRequestFromReader(hProtocol, pstChannel);
		if(result != ERR_UEM_NOERROR)
		{
			result = UKUEMProtocol_SetResultMessage(hProtocol, ERR_UEMPROTOCOL_INTERNAL, 0);
		}
		ERRIFGOTO(result, _EXIT);

		result = UKUEMProtocol_Send(hProtocol);
		ERRIFGOTO(result, _EXIT);
	}

_EXIT:
	if(result != ERR_UEM_NOERROR && pstWriterChannel->stCommonInfo.bChannelExit == FALSE)
	{
		UEM_DEBUG_PRINT("channelReceiverHandlingThread is exited with error: %08x\n", result);
	}
	return NULL;
}


static uem_result notifyAndPushError(SChannel *pstChannel, SCommunicationQueueItem *pstReceivedItem)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SRemoteWriterChannel *pstWriterChannel = NULL;
	SAggregateConnectionInfo *pstConnectionInfo = NULL;
	SCommunicationQueueItem stItemToSend;
	SCommunicationQueueItem *pstItem = NULL;

	pstItem = &stItemToSend;

	pstWriterChannel = (SRemoteWriterChannel *) pstChannel->pChannelStruct;
	pstConnectionInfo = (SAggregateConnectionInfo *) pstWriterChannel->stCommonInfo.pConnectionInfo;

	pstItem->enMessageType = MESSAGE_TYPE_NONE;
	pstItem->nChannelId = pstChannel->nChannelIndex;
	pstItem->uDetailItem.stResponse.enRequestMessageType = pstReceivedItem->enMessageType;
	pstItem->uDetailItem.stResponse.nReturnValue = 0;
	pstItem->uDetailItem.stResponse.pData = NULL;
	pstItem->uDetailItem.stResponse.nDataSize = 0;

	result = UKSerialCommunicationManager_PutItemToSend(pstConnectionInfo->pstServiceInfo->hManager, pstItem);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


// receiver thread for aggregate connection
static void *queueReceiverThread(void *pData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SChannel *pstChannel = NULL;
	SRemoteWriterChannel *pstWriterChannel = NULL;
	SCommunicationQueueItem stReceivedItem;
	SAggregateConnectionInfo *pstConnectionInfo = NULL;
	int nElementSize;

	pstChannel = (SChannel *) pData;
	pstWriterChannel = (SRemoteWriterChannel *) pstChannel->pChannelStruct;
	pstConnectionInfo = (SAggregateConnectionInfo *) pstWriterChannel->stCommonInfo.pConnectionInfo;

	while(pstWriterChannel->stCommonInfo.bChannelExit == FALSE)
	{
		result = UCFixedSizeQueue_GetItem(pstConnectionInfo->uQueue.hRequestQueue, (void *)&stReceivedItem, &nElementSize);
		if(result == ERR_UEM_TIME_EXPIRED)
		{
			continue;
		}
		ERRIFGOTO(result, _EXIT);

		result = handleQueueRequestFromReader(&stReceivedItem, pstChannel);
		if(result != ERR_UEM_NOERROR)
		{
			result = notifyAndPushError(pstChannel, &stReceivedItem);
			ERRIFGOTO(result, _EXIT);
		}
		ERRIFGOTO(result, _EXIT);
	}

_EXIT:
	if(result != ERR_UEM_NOERROR && pstWriterChannel->stCommonInfo.bChannelExit == FALSE)
	{
		UEM_DEBUG_PRINT("queue receiver thread of channel %d is exited with error: %08x\n", pstChannel->nChannelIndex, result);
	}
	return NULL;
}


static uem_result createReceiverThread(SChannel *pstChannel, EConnectionMethod enMethod)
{
	uem_result result = ERR_UEM_UNKNOWN;

	SRemoteWriterChannel *pstWriterChannel = NULL;

	pstWriterChannel = (SRemoteWriterChannel *) pstChannel->pChannelStruct;

	switch(enMethod)
	{
	case CONNECTION_METHOD_AGGREGATE:
		result = UCThread_Create(queueReceiverThread, (void *) pstChannel, &(pstWriterChannel->hReceivingThread));
		ERRIFGOTO(result, _EXIT);
		break;
	case CONNECTION_METHOD_INDIVIDUAL:
		result = UCThread_Create(packetReceiverThread, (void *) pstChannel, &(pstWriterChannel->hReceivingThread));
		ERRIFGOTO(result, _EXIT);
		break;
	default:
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
		break;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result destroyReceiverThread(SChannel *pstChannel)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SRemoteWriterChannel *pstWriterChannel = NULL;

	pstWriterChannel = (SRemoteWriterChannel *) pstChannel->pChannelStruct;

	result = UCThread_Destroy(&(pstWriterChannel->hReceivingThread), FALSE, 0);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result disconnectToServer(SIndividualConnectionInfo *pstConnectionInfo)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SVirtualCommunicationAPI *pstAPI = NULL;

	pstAPI = pstConnectionInfo->pstCommunicationAPI;

	if(pstConnectionInfo->hProtocol != NULL)
	{
		UKUEMProtocol_Destroy(&(pstConnectionInfo->hProtocol));
	}

	if(pstConnectionInfo->hSocket != NULL)
	{
		pstAPI->fnDestroy(&(pstConnectionInfo->hSocket));
	}

	result = ERR_UEM_NOERROR;

	return result;
}



static uem_result initializeAggregateChannel(IN uem_bool *pbExitFlag, SAggregateServiceInfo *pstServiceInfo, int nChannelId, OUT HFixedSizeQueue *phRecevingQueue)
{
	uem_result result = ERR_UEM_UNKNOWN;


	result = UKSerialCommunicationManager_SetChannel(pstServiceInfo->hManager, nChannelId);
	ERRIFGOTO(result, _EXIT);

	result = UKSerialCommunicationManager_GetChannelQueue(pstServiceInfo->hManager, nChannelId, phRecevingQueue);
	ERRIFGOTO(result, _EXIT);

	// wait until serial manager is initialized
	while(pstServiceInfo->bInitialized == FALSE)
	{
		if(*pbExitFlag == TRUE)
		{
			ERRASSIGNGOTO(result, ERR_UEM_SUSPEND, _EXIT);
		}
		UCTime_Sleep(10);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result connectToServer(uem_bool *pbExitFlag, SIndividualConnectionInfo *pstConnectionInfo)
{
	uem_result result = ERR_UEM_UNKNOWN;
	HUEMProtocol hProtocol = NULL;
	HVirtualSocket hSocket = NULL;
	int nRetryCount = 0;
	SVirtualCommunicationAPI *pstAPI = NULL;

	pstAPI = pstConnectionInfo->pstCommunicationAPI;

	result = pstAPI->fnCreate(&hSocket, pstConnectionInfo->pCommunicationInfo);
	ERRIFGOTO(result, _EXIT);

	while(nRetryCount < CONNECT_RETRY_COUNT)
	{
		if(*pbExitFlag == TRUE)
		{
			ERRASSIGNGOTO(result, ERR_UEM_SUSPEND, _EXIT);
		}

		result = pstAPI->fnConnect(hSocket, CONNECT_TIMEOUT);
		if(result == ERR_UEM_NET_TIMEOUT || result == ERR_UEM_CONNECT_ERROR)
		{
			nRetryCount++;
			UCTime_Sleep(CONNECT_TIMEOUT*SECOND_IN_MILLISECOND);
			continue;
		}
		ERRIFGOTO(result, _EXIT);
		break;
	}

	result = UKUEMProtocol_Create(&hProtocol);
	ERRIFGOTO(result, _EXIT);

	result = UKUEMProtocol_SetSocket(hProtocol, hSocket, pstAPI);
	ERRIFGOTO(result, _EXIT);

	result = UKUEMProtocol_HandShake(hProtocol, 0, pstConnectionInfo->nChannelId);
	ERRIFGOTO(result, _EXIT);

	pstConnectionInfo->hSocket = hSocket;
	pstConnectionInfo->hProtocol = hProtocol;

	hSocket = NULL;
	hProtocol = NULL;

	result = ERR_UEM_NOERROR;
_EXIT:
	if(hProtocol != NULL)
	{
		UKUEMProtocol_Destroy(&hProtocol);
	}

	if(hSocket != NULL && pstAPI != NULL)
	{
		pstAPI->fnDestroy(&hSocket);
	}
	return result;
}

static uem_result waitFromClient(uem_bool *pbExitFlag, SIndividualConnectionInfo *pstConnectionInfo)
{
	uem_result result = ERR_UEM_UNKNOWN;

	while(pstConnectionInfo->hProtocol == NULL)
	{
		if(*pbExitFlag == TRUE)
		{
			ERRASSIGNGOTO(result, ERR_UEM_SUSPEND, _EXIT);
		}
		UCTime_Sleep(10);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result connectToRemoteDevice(uem_bool *pbExitFlag, SIndividualConnectionInfo *pstConnectionInfo)
{
	uem_result result = ERR_UEM_UNKNOWN;

	switch(pstConnectionInfo->enType)
	{
	case PAIR_TYPE_CLIENT:
		result = connectToServer(pbExitFlag, pstConnectionInfo);
		ERRIFGOTO(result, _EXIT);
		break;
	case PAIR_TYPE_SERVER:
		result = waitFromClient(pbExitFlag, pstConnectionInfo);
		ERRIFGOTO(result, _EXIT);
		break;
	default:
		ERRASSIGNGOTO(result, ERR_UEM_NOT_SUPPORTED, _EXIT);
		break;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


//pstCommonChannel->bChannelExit = FALSE;
//pstCommonChannel->hMutex = NULL;
//pstCommonChannel->
uem_result UKRemoteChannel_Initialize(SChannel *pstChannel)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SRemoteWriterChannel *pstWriterChannel = NULL;
	SRemoteReaderChannel *pstReaderChannel = NULL;
	SRemoteChannel *pstCommonChannel = NULL;
	SAggregateConnectionInfo *pstAggregateConnectionInfo = NULL;
	SIndividualConnectionInfo *pstIndividualConnectionInfo = NULL;
	HFixedSizeQueue hQueue;
	int nLoop = 0;

	switch(pstChannel->enType)
	{
	case COMMUNICATION_TYPE_REMOTE_READER:
		pstReaderChannel = (SRemoteReaderChannel *) pstChannel->pChannelStruct;
		pstCommonChannel = &(pstReaderChannel->stCommonInfo);
		break;
	case COMMUNICATION_TYPE_REMOTE_WRITER:
		pstWriterChannel = (SRemoteWriterChannel *) pstChannel->pChannelStruct;
		pstWriterChannel->nBufLen = 0;
		pstWriterChannel->pBuffer = NULL;
		pstWriterChannel->hReceivingThread = NULL;
		pstCommonChannel = &(pstWriterChannel->stCommonInfo);

		result = UKChannelMemory_Initialize(pstChannel, pstWriterChannel->pstInternalChannel);
		ERRIFGOTO(result, _EXIT);
		break;
	default:
		ERRASSIGNGOTO(result, ERR_UEM_NOT_SUPPORTED, _EXIT);
		break;
	}

	switch(pstCommonChannel->enConnectionMethod)
	{
	case CONNECTION_METHOD_INDIVIDUAL:
		for(nLoop = 0 ; nLoop < g_nIndividualConnectionInfoNum ; nLoop++)
		{
			if(g_astIndividualConnectionInfo[nLoop].nChannelId == pstChannel->nChannelIndex)
			{
				pstCommonChannel->pConnectionInfo = (void *) &(g_astIndividualConnectionInfo[nLoop]);
				break;
			}
		}

		pstIndividualConnectionInfo = pstCommonChannel->pConnectionInfo;

		result = connectToRemoteDevice(&(pstCommonChannel->bChannelExit), pstIndividualConnectionInfo);
		ERRIFGOTO(result, _EXIT);
		break;
	case CONNECTION_METHOD_AGGREGATE:
		for(nLoop = 0 ; nLoop < g_nAggregateConnectionInfoNum ; nLoop++)
		{
			if(g_astAggregateConnectionInfo[nLoop].nChannelId == pstChannel->nChannelIndex)
			{
				pstCommonChannel->pConnectionInfo = (void *) &(g_astAggregateConnectionInfo[nLoop]);
				break;
			}
		}

		pstAggregateConnectionInfo = (SAggregateConnectionInfo *) pstCommonChannel->pConnectionInfo;
		result = initializeAggregateChannel(&(pstCommonChannel->bChannelExit), pstAggregateConnectionInfo->pstServiceInfo,
											pstChannel->nChannelIndex, &hQueue);
		ERRIFGOTO(result, _EXIT);

		if(pstChannel->enType == COMMUNICATION_TYPE_REMOTE_READER)
		{
			pstAggregateConnectionInfo->uQueue.hResponseQueue = hQueue;
		}
		else if(pstChannel->enType == COMMUNICATION_TYPE_REMOTE_WRITER)
		{
			pstAggregateConnectionInfo->uQueue.hRequestQueue = hQueue;
		}
		break;
	default:
		ERRASSIGNGOTO(result, ERR_UEM_NOT_SUPPORTED, _EXIT);
		break;
	}

	if(pstChannel->enType == COMMUNICATION_TYPE_REMOTE_WRITER)
	{
		result = createReceiverThread(pstChannel, pstCommonChannel->enConnectionMethod);
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static uem_result sendAndCheckResult(SRemoteReaderChannel *pstReaderChannel, HUEMProtocol hProtocol, int *pnReturnValue)
{
	uem_result result = ERR_UEM_UNKNOWN;
	EProtocolError enErrorCode = ERR_UEMPROTOCOL_INTERNAL;

	result = UKUEMProtocol_Send(hProtocol);
	ERRIFGOTO(result, _EXIT);

	do
	{
		if(pstReaderChannel->stCommonInfo.bChannelExit == TRUE)
		{
			ERRASSIGNGOTO(result, ERR_UEM_SUSPEND, _EXIT);
		}
		result = UKUEMProtocol_Receive(hProtocol);
		if(result == ERR_UEM_NET_TIMEOUT)
		{
			continue;
		}
		ERRIFGOTO(result, _EXIT);
	}while(result == ERR_UEM_NET_TIMEOUT);


	result = UKUEMProtocol_GetResultFromReceivedData(hProtocol, &enErrorCode, pnReturnValue);
	ERRIFGOTO(result, _EXIT);

	if(enErrorCode != ERR_UEMPROTOCOL_NOERROR)
	{
		UEM_DEBUG_PRINT("Error is retrieved from remote writer\n");
		ERRASSIGNGOTO(result, ERR_UEM_INTERNAL_FAIL, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static uem_result readFromQueueThroughCommunication(SRemoteReaderChannel *pstRemoteChannel, SIndividualConnectionInfo *pstConnectionInfo,
													IN int nDataToRead, IN int nChunkIndex, OUT unsigned char **ppDataPointer, OUT int *pnDataRead)
{
	uem_result result = ERR_UEM_UNKNOWN;
	HUEMProtocol hProtocol = NULL;
	int nDataRead = 0;
	void *pBody = NULL;

	hProtocol = pstConnectionInfo->hProtocol;

	result = UKUEMProtocol_SetReadQueueRequest(hProtocol, nChunkIndex, nDataToRead);
	ERRIFGOTO(result, _EXIT);

	result = sendAndCheckResult(pstRemoteChannel, hProtocol, NULL);
	ERRIFGOTO(result, _EXIT);

	result = UKUEMProtocol_GetBodyDataFromReceivedData(hProtocol, &nDataRead, &pBody);
	ERRIFGOTO(result, _EXIT);

	*ppDataPointer = pBody;
	*pnDataRead = nDataRead;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static uem_result readFromQueueThroughAggregator(int nChannelId, SRemoteReaderChannel *pstRemoteChannel, SAggregateConnectionInfo *pstConnectionInfo,
												IN int nDataToRead, IN int nChunkIndex, OUT unsigned char **ppDataPointer, OUT int *pnDataRead)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCommunicationQueueItem stItem;
	SCommunicationQueueItem stResponseItem;
	SAggregateServiceInfo *pstServiceInfo = NULL;
	int nElementSize = 0;

	pstServiceInfo = pstConnectionInfo->pstServiceInfo;

	stItem.enMessageType = MESSAGE_TYPE_READ_QUEUE;
	stItem.nChannelId = nChannelId;
	stItem.uDetailItem.stRequest.nRequestDataSize = nDataToRead;
	stItem.uDetailItem.stRequest.nChunkIndex = nChunkIndex; // not used

	result = UKSerialCommunicationManager_PutItemToSend(pstServiceInfo->hManager, &stItem);
	ERRIFGOTO(result, _EXIT);

	// blocking happened here
	do {
		result = UCFixedSizeQueue_GetItem(pstConnectionInfo->uQueue.hResponseQueue, &stResponseItem, &nElementSize);
	}while(result == ERR_UEM_TIME_EXPIRED && pstRemoteChannel->stCommonInfo.bChannelExit == FALSE);
	ERRIFGOTO(result, _EXIT);

	if(stResponseItem.enMessageType != MESSAGE_TYPE_RESULT ||
		stResponseItem.nChannelId != nChannelId ||
		stResponseItem.uDetailItem.stResponse.enRequestMessageType != MESSAGE_TYPE_READ_QUEUE)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
	}

	*ppDataPointer = stResponseItem.uDetailItem.stResponse.pData;
	*pnDataRead = stResponseItem.uDetailItem.stResponse.nDataSize;


	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKRemoteChannel_ReadFromQueue(SChannel *pstChannel, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SRemoteReaderChannel *pstReaderChannel = NULL;
	SIndividualConnectionInfo *pstIndividualConnectionInfo = NULL;
	SAggregateConnectionInfo *pstAggregateConnectionInfo = NULL;
	int nDataRead = 0;
	unsigned char *pData = NULL;

	pstReaderChannel = (SRemoteReaderChannel *) pstChannel->pChannelStruct;

	switch(pstReaderChannel->stCommonInfo.enConnectionMethod)
	{
	case CONNECTION_METHOD_INDIVIDUAL:
		pstIndividualConnectionInfo = (SIndividualConnectionInfo *) pstReaderChannel->stCommonInfo.pConnectionInfo;
		result = readFromQueueThroughCommunication(pstReaderChannel, pstIndividualConnectionInfo, nDataToRead, nChunkIndex, &pData, &nDataRead);
		ERRIFGOTO(result, _EXIT);
		break;
	case CONNECTION_METHOD_AGGREGATE:
		pstAggregateConnectionInfo = (SAggregateConnectionInfo *) pstReaderChannel->stCommonInfo.pConnectionInfo;
		result = readFromQueueThroughAggregator(pstChannel->nChannelIndex, pstReaderChannel, pstAggregateConnectionInfo,
												nDataToRead, nChunkIndex, &pData, &nDataRead);
		ERRIFGOTO(result, _EXIT);
		break;
	default:
		ERRASSIGNGOTO(result, ERR_UEM_NOT_SUPPORTED, _EXIT);
		break;
	}

	if(pBuffer != NULL)
	{
		result = pstReaderChannel->pstReaderAccess->fnCopyFromMemory(pBuffer, pData, nDataRead);
		ERRIFGOTO(result, _EXIT);
	}

	*pnDataRead = nDataRead;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result readFromBufferThroughCommunication(SRemoteReaderChannel *pstRemoteChannel, SIndividualConnectionInfo *pstConnectionInfo,
													IN int nDataToRead, IN int nChunkIndex, OUT unsigned char **ppDataPointer, OUT int *pnDataRead)
{
	uem_result result = ERR_UEM_UNKNOWN;
	HUEMProtocol hProtocol = NULL;
	int nDataRead = 0;
	void *pBody = NULL;

	hProtocol = pstConnectionInfo->hProtocol;

	result = UKUEMProtocol_SetReadBufferRequest(hProtocol, nChunkIndex, nDataToRead);
	ERRIFGOTO(result, _EXIT);

	result = sendAndCheckResult(pstRemoteChannel, hProtocol, NULL);
	ERRIFGOTO(result, _EXIT);

	result = UKUEMProtocol_GetBodyDataFromReceivedData(hProtocol, &nDataRead, &pBody);
	ERRIFGOTO(result, _EXIT);

	*ppDataPointer = pBody;
	*pnDataRead = nDataRead;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static uem_result readFromBufferThroughAggregator(int nChannelId, SRemoteReaderChannel *pstRemoteChannel, SAggregateConnectionInfo *pstConnectionInfo,
												IN int nDataToRead, IN int nChunkIndex, OUT unsigned char **ppDataPointer, OUT int *pnDataRead)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCommunicationQueueItem stItem;
	SCommunicationQueueItem stResponseItem;
	SAggregateServiceInfo *pstServiceInfo = NULL;
	int nElementSize = 0;

	pstServiceInfo = pstConnectionInfo->pstServiceInfo;

	stItem.enMessageType = MESSAGE_TYPE_READ_BUFFER;
	stItem.nChannelId = nChannelId;
	stItem.uDetailItem.stRequest.nRequestDataSize = nDataToRead;
	stItem.uDetailItem.stRequest.nChunkIndex = nChunkIndex; // not used

	result = UKSerialCommunicationManager_PutItemToSend(pstServiceInfo->hManager, &stItem);
	ERRIFGOTO(result, _EXIT);

	// blocking happened here
	do {
		result = UCFixedSizeQueue_GetItem(pstConnectionInfo->uQueue.hResponseQueue, &stResponseItem, &nElementSize);
	}while(result == ERR_UEM_TIME_EXPIRED && pstRemoteChannel->stCommonInfo.bChannelExit == FALSE);
	ERRIFGOTO(result, _EXIT);

	if(stResponseItem.enMessageType != MESSAGE_TYPE_RESULT ||
		stResponseItem.nChannelId != nChannelId ||
		stResponseItem.uDetailItem.stResponse.enRequestMessageType != MESSAGE_TYPE_READ_BUFFER)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
	}

	*ppDataPointer = stResponseItem.uDetailItem.stResponse.pData;
	*pnDataRead = stResponseItem.uDetailItem.stResponse.nDataSize;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}



uem_result UKRemoteChannel_ReadFromBuffer(SChannel *pstChannel, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SRemoteReaderChannel *pstReaderChannel = NULL;
	SIndividualConnectionInfo *pstIndividualConnectionInfo = NULL;
	SAggregateConnectionInfo *pstAggregateConnectionInfo = NULL;
	int nDataRead = 0;
	unsigned char *pData = NULL;

	pstReaderChannel = (SRemoteReaderChannel *) pstChannel->pChannelStruct;

	switch(pstReaderChannel->stCommonInfo.enConnectionMethod)
	{
	case CONNECTION_METHOD_INDIVIDUAL:
		pstIndividualConnectionInfo = (SIndividualConnectionInfo *) pstReaderChannel->stCommonInfo.pConnectionInfo;
		result = readFromBufferThroughCommunication(pstReaderChannel, pstIndividualConnectionInfo, nDataToRead, nChunkIndex, &pData, &nDataRead);
		ERRIFGOTO(result, _EXIT);
		break;
	case CONNECTION_METHOD_AGGREGATE:
		pstAggregateConnectionInfo = (SAggregateConnectionInfo *) pstReaderChannel->stCommonInfo.pConnectionInfo;
		result = readFromBufferThroughAggregator(pstChannel->nChannelIndex, pstReaderChannel, pstAggregateConnectionInfo,
												nDataToRead, nChunkIndex, &pData, &nDataRead);
		ERRIFGOTO(result, _EXIT);
		break;
	default:
		ERRASSIGNGOTO(result, ERR_UEM_NOT_SUPPORTED, _EXIT);
		break;
	}

	result = pstReaderChannel->pstReaderAccess->fnCopyFromMemory(pBuffer, pData, nDataRead);
	ERRIFGOTO(result, _EXIT);

	*pnDataRead = nDataRead;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKRemoteChannel_GetAvailableChunk (SChannel *pstChannel, OUT int *pnChunkIndex)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SRemoteReaderChannel *pstReaderChannel = NULL;
	int nAvailableIndex = 0;
	SIndividualConnectionInfo *pstIndividualConnectionInfo = NULL;

	pstReaderChannel = (SRemoteReaderChannel *) pstChannel->pChannelStruct;

	switch(pstReaderChannel->stCommonInfo.enConnectionMethod)
	{
	case CONNECTION_METHOD_INDIVIDUAL:
		pstIndividualConnectionInfo = (SIndividualConnectionInfo *) pstReaderChannel->stCommonInfo.pConnectionInfo;

		result = UKUEMProtocol_SetAvailableIndexRequest(pstIndividualConnectionInfo->hProtocol);
		ERRIFGOTO(result, _EXIT);

		result = sendAndCheckResult(pstReaderChannel, pstIndividualConnectionInfo->hProtocol, &nAvailableIndex);
		ERRIFGOTO(result, _EXIT);
		break;
	case CONNECTION_METHOD_AGGREGATE:
		// TODO: Aggregate connection does not support this function because underlying UEMLiteProtocol does not support this (maybe support later?)
		// always return 0
		nAvailableIndex = 0;
		break;
	default:
		ERRASSIGNGOTO(result, ERR_UEM_NOT_SUPPORTED, _EXIT);
		break;
	}

	*pnChunkIndex = nAvailableIndex;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result getNumOfAvailableDataThroughCommunication(SRemoteReaderChannel *pstRemoteChannel, SIndividualConnectionInfo *pstConnectionInfo,
												IN int nChunkIndex, OUT int *pnDataNum)
{
	uem_result result = ERR_UEM_UNKNOWN;
	HUEMProtocol hProtocol = NULL;
	int nDataNum = 0;

	hProtocol = pstConnectionInfo->hProtocol;

	result = UKUEMProtocol_SetAvailableDataRequest(hProtocol, nChunkIndex);
	ERRIFGOTO(result, _EXIT);

	result = sendAndCheckResult(pstRemoteChannel, hProtocol, &nDataNum);
	ERRIFGOTO(result, _EXIT);

	*pnDataNum = nDataNum;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result getNumOfAvailableDataThroughAggregator(int nChannelId, SRemoteReaderChannel *pstRemoteChannel, SAggregateConnectionInfo *pstConnectionInfo,
												IN int nChunkIndex, OUT int *pnDataNum)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCommunicationQueueItem stItem;
	SCommunicationQueueItem stResponseItem;
	int nElementSize = 0;
	SAggregateServiceInfo *pstServiceInfo = NULL;

	pstServiceInfo = pstConnectionInfo->pstServiceInfo;

	stItem.enMessageType = MESSAGE_TYPE_AVAILABLE_DATA;
	stItem.nChannelId = nChannelId;
	stItem.uDetailItem.stRequest.nRequestDataSize = 0; // not used
	stItem.uDetailItem.stRequest.nChunkIndex = nChunkIndex; // not used

	result = UKSerialCommunicationManager_PutItemToSend(pstServiceInfo->hManager, &stItem);
	ERRIFGOTO(result, _EXIT);

	// blocking happened here
	do {
		result = UCFixedSizeQueue_GetItem(pstConnectionInfo->uQueue.hResponseQueue, &stResponseItem, &nElementSize);
	}while(result == ERR_UEM_TIME_EXPIRED && pstRemoteChannel->stCommonInfo.bChannelExit == FALSE);
	ERRIFGOTO(result, _EXIT);

	if(stResponseItem.enMessageType != MESSAGE_TYPE_RESULT ||
		stResponseItem.nChannelId != nChannelId ||
		stResponseItem.uDetailItem.stResponse.enRequestMessageType != MESSAGE_TYPE_AVAILABLE_DATA)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
	}

	*pnDataNum = stResponseItem.uDetailItem.stResponse.nReturnValue;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKRemoteChannel_GetNumOfAvailableData (SChannel *pstChannel, IN int nChunkIndex, OUT int *pnDataNum)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SRemoteReaderChannel *pstReaderChannel = NULL;
	SRemoteWriterChannel *pstWriterChannel = NULL;
	int nDataNum = 0;
	SIndividualConnectionInfo *pstIndividualConnectionInfo = NULL;
	SAggregateConnectionInfo *pstAggregateConnectionInfo = NULL;

	switch(pstChannel->enType)
	{
	case COMMUNICATION_TYPE_REMOTE_READER:
		pstReaderChannel = (SRemoteReaderChannel *)  pstChannel->pChannelStruct;
		switch(pstReaderChannel->stCommonInfo.enConnectionMethod)
		{
		case CONNECTION_METHOD_INDIVIDUAL:
			pstIndividualConnectionInfo = (SIndividualConnectionInfo *) pstReaderChannel->stCommonInfo.pConnectionInfo;
			result = getNumOfAvailableDataThroughCommunication(pstReaderChannel, pstIndividualConnectionInfo, nChunkIndex, &nDataNum);
			ERRIFGOTO(result, _EXIT);
			break;
		case CONNECTION_METHOD_AGGREGATE:
			pstAggregateConnectionInfo = (SAggregateConnectionInfo *) pstReaderChannel->stCommonInfo.pConnectionInfo;
			result = getNumOfAvailableDataThroughAggregator(pstChannel->nChannelIndex, pstReaderChannel, pstAggregateConnectionInfo,
					nChunkIndex, &nDataNum);
			ERRIFGOTO(result, _EXIT);
			break;
		default:
			ERRASSIGNGOTO(result, ERR_UEM_NOT_SUPPORTED, _EXIT);
			break;
		}

		*pnDataNum = nDataNum;
		break;
	case COMMUNICATION_TYPE_REMOTE_WRITER:
		pstWriterChannel = (SRemoteWriterChannel *)  pstChannel->pChannelStruct;
		result = UKChannelMemory_GetNumOfAvailableData(pstChannel, pstWriterChannel->pstInternalChannel, nChunkIndex, pnDataNum);
		ERRIFGOTO(result, _EXIT);
		break;
	default:
		ERRASSIGNGOTO(result, ERR_UEM_NOT_SUPPORTED, _EXIT);
		break;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


// same to shared memory channel write
uem_result UKRemoteChannel_WriteToQueue (SChannel *pstChannel, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SRemoteWriterChannel *pstWriterChannel = NULL;

	pstWriterChannel = (SRemoteWriterChannel *) pstChannel->pChannelStruct;

	result = UKChannelMemory_WriteToQueue(pstChannel, pstWriterChannel->pstInternalChannel, pBuffer, nDataToWrite, nChunkIndex, pnDataWritten);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKRemoteChannel_FillInitialData(SChannel *pstChannel)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SRemoteWriterChannel *pstWriterChannel = NULL;

	pstWriterChannel = (SRemoteWriterChannel *) pstChannel->pChannelStruct;

	result = UKChannelMemory_FillInitialData(pstChannel, pstWriterChannel->pstInternalChannel);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


// same to shared memory channel write
uem_result UKRemoteChannel_WriteToBuffer (SChannel *pstChannel, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SRemoteWriterChannel *pstWriterChannel = NULL;

	pstWriterChannel = (SRemoteWriterChannel *) pstChannel->pChannelStruct;

	result = UKChannelMemory_WriteToBuffer(pstChannel, pstWriterChannel->pstInternalChannel, pBuffer, nDataToWrite, nChunkIndex, pnDataWritten);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKRemoteChannel_SetExit(SChannel *pstChannel, int nExitFlag)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SRemoteWriterChannel *pstWriterChannel = NULL;
	SRemoteReaderChannel *pstReaderChannel = NULL;

	switch(pstChannel->enType)
	{
	case COMMUNICATION_TYPE_REMOTE_READER:
		pstReaderChannel = (SRemoteReaderChannel *)  pstChannel->pChannelStruct;
		pstReaderChannel->stCommonInfo.bChannelExit = TRUE;
		break;
	case COMMUNICATION_TYPE_REMOTE_WRITER:
		pstWriterChannel = (SRemoteWriterChannel *)  pstChannel->pChannelStruct;
		pstWriterChannel->stCommonInfo.bChannelExit = TRUE;
		result = UKChannelMemory_SetExit(pstChannel, pstWriterChannel->pstInternalChannel, nExitFlag);
		ERRIFGOTO(result, _EXIT);
		break;
	default:
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
		break;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKRemoteChannel_ClearExit(SChannel *pstChannel, int nExitFlag)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SRemoteWriterChannel *pstWriterChannel = NULL;
	SRemoteReaderChannel *pstReaderChannel = NULL;

	switch(pstChannel->enType)
	{
	case COMMUNICATION_TYPE_REMOTE_READER:
		pstReaderChannel = (SRemoteReaderChannel *)  pstChannel->pChannelStruct;
		pstReaderChannel->stCommonInfo.bChannelExit = FALSE;
		break;
	case COMMUNICATION_TYPE_REMOTE_WRITER:
		pstWriterChannel = (SRemoteWriterChannel *)  pstChannel->pChannelStruct;
		pstWriterChannel->stCommonInfo.bChannelExit = FALSE;
		result = UKChannelMemory_ClearExit(pstChannel, pstWriterChannel->pstInternalChannel, nExitFlag);
		ERRIFGOTO(result, _EXIT);
		break;
	default:
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
		break;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKRemoteChannel_Finalize(SChannel *pstChannel)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SRemoteWriterChannel *pstWriterChannel = NULL;
	SRemoteReaderChannel *pstReaderChannel = NULL;
	SRemoteChannel *pstCommonChannel = NULL;
	SAggregateConnectionInfo *pstAggregateConnectionInfo = NULL;
	SIndividualConnectionInfo *pstIndividualConnectionInfo = NULL;
	int nRetryCount = 0;

	UKRemoteChannel_SetExit(pstChannel, EXIT_FLAG_READ | EXIT_FLAG_WRITE);

	switch(pstChannel->enType)
	{
	case COMMUNICATION_TYPE_REMOTE_READER:
		pstReaderChannel = (SRemoteReaderChannel *) pstChannel->pChannelStruct;
		pstCommonChannel = &(pstReaderChannel->stCommonInfo);
		break;
	case COMMUNICATION_TYPE_REMOTE_WRITER:
		pstWriterChannel = (SRemoteWriterChannel *) pstChannel->pChannelStruct;
		pstCommonChannel = &(pstWriterChannel->stCommonInfo);
		// wait until the reader consumes the data for several seconds
		while(pstWriterChannel->pstInternalChannel->nDataLen > 0 &&
				nRetryCount < REMOTE_REQUEST_WAIT_RETRY_COUNT)
		{
			nRetryCount++;
			UCTime_Sleep(REMOTE_REQUEST_WAIT_SLEEP_TIME);
		}
		break;
	default:
		ERRASSIGNGOTO(result, ERR_UEM_NOT_SUPPORTED, _EXIT);
		break;
	}

	switch(pstCommonChannel->enConnectionMethod)
	{
	case CONNECTION_METHOD_INDIVIDUAL:
		pstIndividualConnectionInfo = pstCommonChannel->pConnectionInfo;
		if(pstIndividualConnectionInfo->enType == PAIR_TYPE_CLIENT)
		{
			result = disconnectToServer(pstIndividualConnectionInfo);
			ERRIFGOTO(result, _EXIT);
		}
		break;
	case CONNECTION_METHOD_AGGREGATE:
		pstAggregateConnectionInfo = (SAggregateConnectionInfo *) pstCommonChannel->pConnectionInfo;
		result = UKSerialCommunicationManager_ReleaseChannel(pstAggregateConnectionInfo->pstServiceInfo->hManager, pstChannel->nChannelIndex);
		ERRIFGOTO(result, _EXIT);
		break;
	default:
		ERRASSIGNGOTO(result, ERR_UEM_NOT_SUPPORTED, _EXIT);
		break;
	}

	if(pstChannel->enType == COMMUNICATION_TYPE_REMOTE_WRITER)
	{
		result = destroyReceiverThread(pstChannel);
		ERRIFGOTO(result, _EXIT);

		result = UKChannelMemory_Finalize(pstChannel, pstWriterChannel->pstInternalChannel);
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKRemoteChannel_APIInitialize()
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;

	for(nLoop = 0 ; nLoop < g_nRemoteCommunicationModuleNum ; nLoop++)
	{
		result = g_aFnRemoteCommunicationModuleIntializeList[nLoop]();
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKRemoteChannel_APIFinalize()
{
	uem_result result = ERR_UEM_UNKNOWN;

	int nLoop = 0;

	for(nLoop = 0 ; nLoop < g_nRemoteCommunicationModuleNum ; nLoop++)
	{
		result = g_aFnRemoteCommunicationModuleFinalizeList[nLoop]();
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}




