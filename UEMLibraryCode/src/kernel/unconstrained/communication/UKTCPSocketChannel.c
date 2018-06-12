/*
 * UKTCPSocketChannel.c
 *
 *  Created on: 2018. 6. 2.
 *      Author: jej
 */


#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <UCBasic.h>
#include <UCThreadMutex.h>

#include <uem_data.h>

#include <UKChannelMemory.h>
#include <UKTask.h>

#include <UKUEMProtocol.h>


#define CONNECT_TIMEOUT (3)
#define CONNECT_RETRY_COUNT (100)
#define SECOND_IN_MILLISECOND (1000)

uem_result UKTCPSocketChannel_Clear(SChannel *pstChannel)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STCPSocketChannel *pstTCPChannel = NULL;

	pstTCPChannel = (STCPSocketChannel *) pstChannel->pChannelStruct;

	switch(pstChannel->enType)
	{
	case COMMUNICATION_TYPE_TCP_CLIENT_READER:
	case COMMUNICATION_TYPE_TCP_SERVER_READER:
		// do nothing
		break;
	case COMMUNICATION_TYPE_TCP_CLIENT_WRITER:
	case COMMUNICATION_TYPE_TCP_SERVER_WRITER:
		result = UKChannelMemory_Clear(pstChannel, pstTCPChannel->pstInternalChannel);
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

static uem_result reallocTempBuffer(STCPSocketChannel *pstTCPChannel, int nTargetSize)
{
	uem_result result = ERR_UEM_UNKNOWN;

	if(pstTCPChannel->nBufLen < nTargetSize)
	{
		SAFEMEMFREE(pstTCPChannel->pBuffer);

		pstTCPChannel->pBuffer = UC_malloc(nTargetSize);
		ERRMEMGOTO(pstTCPChannel->pBuffer, result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}



static uem_result handleReadQueue(SChannel *pstChannel, int *panParam, int nParamNum)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STCPSocketChannel *pstTCPChannel = NULL;
	int nDataToRead = 0;
	int nChunkIndex = 0;
	int nDataRead = 0;
	int nChannelId = INVALID_CHANNEL_ID;

	pstTCPChannel = (STCPSocketChannel *) pstChannel->pChannelStruct;

	nChannelId = panParam[READ_QUEUE_CHANNEL_ID_INDEX];

	if(pstChannel->nChannelIndex != nChannelId)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
	}

	nDataToRead = panParam[READ_QUEUE_SIZE_TO_READ_INDEX];
	nChunkIndex = panParam[READ_QUEUE_CHUNK_INDEX_INDEX];

	result = reallocTempBuffer(pstTCPChannel, nDataToRead);
	ERRIFGOTO(result, _EXIT);

	result = UKChannelMemory_ReadFromQueue(pstChannel, pstTCPChannel->pstInternalChannel, pstTCPChannel->pBuffer, nDataToRead, nChunkIndex, &nDataRead);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result handleReadBuffer(SChannel *pstChannel, int *panParam, int nParamNum, HUEMProtocol hProtocol)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STCPSocketChannel *pstTCPChannel = NULL;
	int nDataToRead = 0;
	int nChunkIndex = 0;
	int nDataRead = 0;
	int nChannelId = INVALID_CHANNEL_ID;

	pstTCPChannel = (STCPSocketChannel *) pstChannel->pChannelStruct;

	nChannelId = panParam[READ_BUFFER_CHANNEL_ID_INDEX];

	if(pstChannel->nChannelIndex != nChannelId)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
	}

	nDataToRead = panParam[READ_BUFFER_SIZE_TO_READ_INDEX];
	nChunkIndex = panParam[READ_BUFFER_CHUNK_INDEX_INDEX];

	result = reallocTempBuffer(pstTCPChannel, nDataToRead);
	ERRIFGOTO(result, _EXIT);

	result = UKChannelMemory_ReadFromBuffer(pstChannel, pstTCPChannel->pstInternalChannel, pstTCPChannel->pBuffer, nDataToRead, nChunkIndex, &nDataRead);
	ERRIFGOTO(result, _EXIT);

	result = UKUEMProtocol_SetResultMessageWithBuffer(hProtocol, ERR_UEMPROTOCOL_NOERROR, nDataRead, pstTCPChannel->pBuffer);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result handleAvailableIndex(SChannel *pstChannel, int *panParam, int nParamNum, HUEMProtocol hProtocol)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STCPSocketChannel *pstTCPChannel = NULL;
	int nChunkIndex = 0;
	int nChannelId = INVALID_CHANNEL_ID;

	pstTCPChannel = (STCPSocketChannel *) pstChannel->pChannelStruct;

	nChannelId = panParam[AVAILABLE_INDEX_CHANNEL_ID_INDEX];

	if(pstChannel->nChannelIndex != nChannelId)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
	}

	result = UKChannelMemory_GetAvailableChunk(pstChannel, pstTCPChannel->pstInternalChannel, pstTCPChannel->pBuffer, &nChunkIndex);
	ERRIFGOTO(result, _EXIT);

	result = UKUEMProtocol_SetResultMessage(hProtocol, ERR_UEMPROTOCOL_NOERROR, nChunkIndex);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result handleAvailableData(SChannel *pstChannel, int *panParam, int nParamNum, HUEMProtocol hProtocol)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STCPSocketChannel *pstTCPChannel = NULL;
	int nChunkIndex = 0;
	int nChannelId = INVALID_CHANNEL_ID;
	int nDataNum = 0;

	pstTCPChannel = (STCPSocketChannel *) pstChannel->pChannelStruct;

	nChannelId = panParam[AVAILABLE_DATA_CHANNEL_ID_INDEX];

	if(pstChannel->nChannelIndex != nChannelId)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
	}

	nChunkIndex = panParam[AVAILABLE_DATA_CHUNK_INDEX_INDEX];

	result = UKChannelMemory_GetNumOfAvailableData(pstChannel, pstTCPChannel->pstInternalChannel, nChunkIndex, &nDataNum);
	ERRIFGOTO(result, _EXIT);

	result = UKUEMProtocol_SetResultMessage(hProtocol, ERR_UEMPROTOCOL_NOERROR, nDataNum);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result handleRequestFromReader(HUEMProtocol hProtocol, SChannel *pstChannel)
{
	uem_result result = ERR_UEM_UNKNOWN;
	EMessageType enMessageType = MESSAGE_TYPE_NONE;
	int nParamNum = 0;
	int *panParam = NULL;
	STCPSocketChannel *pstTCPChannel = NULL;
	int nDataToRead = 0;
	int nChunkIndex = 0;
	int nDataRead = 0;

	pstTCPChannel = (STCPSocketChannel *) pstChannel->pChannelStruct;

	result = UKUEMProtocol_GetRequestFromReceivedData(hProtocol, &enMessageType, &nParamNum, &panParam);
	ERRIFGOTO(result, _EXIT);

	switch(enMessageType)
	{
	case MESSAGE_TYPE_READ_QUEUE:
		result = handleReadQueue(pstChannel, panParam, nParamNum, hProtocol);
		ERRIFGOTO(result, _EXIT);
		break;
	case MESSAGE_TYPE_READ_BUFFER:
		result = handleReadBuffer(pstChannel, panParam, nParamNum, hProtocol);
		ERRIFGOTO(result, _EXIT);
		break;
	case MESSAGE_TYPE_AVAILABLE_INDEX:
		result = handleAvailableIndex(pstChannel, panParam, nParamNum, hProtocol);
		ERRIFGOTO(result, _EXIT);
		break;
	case MESSAGE_TYPE_AVAILABLE_DATA:
		result = handleAvailableData(pstChannel, panParam, nParamNum, hProtocol);
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


static void *channelReceiverHandlingThread(void *pData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SChannel *pstChannel = NULL;
	STCPSocketChannel *pstTCPChannel = NULL;
	HUEMProtocol hProtocol = NULL;

	pstChannel = (SChannel *) pData;
	pstTCPChannel = (STCPSocketChannel *) pstChannel->pChannelStruct;

	hProtocol = pstTCPChannel->pstCommunicationInfo->hProtocol;

	while(pstTCPChannel->bChannelExit == FALSE)
	{
		result = UKUEMProtocol_Receive(hProtocol);
		if(result == ERR_UEM_NET_TIMEOUT)
		{
			continue;
		}
		ERRIFGOTO(result, _EXIT);

		result = handleRequestFromReader(hProtocol, pstChannel);
		if(result != ERR_UEM_NOERROR)
		{
			result = UKUEMProtocol_SetResultMessage(hProtocol, ERR_UEMPROTOCOL_INTERNAL, 0);
		}
		ERRIFGOTO(result, _EXIT);

		result = UKUEMProtocol_Send(hProtocol);
		ERRIFGOTO(result, _EXIT);
	}

_EXIT:
	if(result != ERR_UEM_NOERROR && pstTCPChannel->bChannelExit == FALSE)
	{
		UEM_DEBUG_PRINT("channelReceiverHandlingThread is exited with error: %d\n", result);
	}
	return NULL;
}


static uem_result createReceiverThread(SChannel *pstChannel)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STCPSocketChannel *pstTCPChannel = NULL;

	pstTCPChannel = (STCPSocketChannel *) pstChannel->pChannelStruct;

	result = UCThread_Create(channelReceiverHandlingThread, (void *) pstChannel, &(pstTCPChannel->hReceivingThread));
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result destroyReceiverThread(SChannel *pstChannel)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STCPSocketChannel *pstTCPChannel = NULL;

	pstTCPChannel = (STCPSocketChannel *) pstChannel->pChannelStruct;

	result = UCThread_Destroy(&(pstTCPChannel->hReceivingThread), FALSE, 0);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static disconnectToServer(STCPSocketChannel *pstTCPChannel)
{
	uem_result result = ERR_UEM_UNKNOWN;

	if(pstTCPChannel->pstCommunicationInfo->hProtocol != NULL)
	{
		UKUEMProtocol_Destroy(&(pstTCPChannel->pstCommunicationInfo->hProtocol));
	}

	if(pstTCPChannel->pstCommunicationInfo->hSocket != NULL)
	{
		UCDynamicSocket_Destroy(&(pstTCPChannel->pstCommunicationInfo->hSocket));
	}

	result = ERR_UEM_NOERROR;

	return result;
}


static uem_result connectToServer(STCPSocketChannel *pstTCPChannel)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSocketInfo stSocketInfo;
	int nRetryCount = 0;
	HUEMProtocol hProtocol = NULL;
	HSocket hSocket = NULL;

	stSocketInfo.enSocketType = SOCKET_TYPE_TCP;
	stSocketInfo.nPort = pstTCPChannel->pstClientInfo->nPort;
	stSocketInfo.pszSocketPath = pstTCPChannel->pstClientInfo->pszIPAddress;

	result = UCDynamicSocket_Create(&stSocketInfo, FALSE, &hSocket);
	ERRIFGOTO(result, _EXIT);

	while(nRetryCount < CONNECT_RETRY_COUNT)
	{
		if(pstTCPChannel->bChannelExit == FALSE)
		{
			ERRASSIGNGOTO(result, ERR_UEM_SUSPEND, _EXIT);
		}

		result = UCDynamicSocket_Connect(hSocket, CONNECT_TIMEOUT);
		if(result == ERR_UEM_NET_TIMEOUT || result == ERR_UEM_CONNECT_ERROR)
		{
			nRetryCount++;
			UCTime_Sleep(CONNECT_TIMEOUT*SECOND_IN_MILLISECOND);
			continue;
		}
		ERRIFGOTO(result, _EXIT);
	}

	result = UKUEMProtocol_Create(&hProtocol);
	ERRIFGOTO(result, _EXIT);

	result = UKUEMProtocol_SetSocket(hProtocol, hSocket);
	ERRIFGOTO(result, _EXIT);

	result = UKUEMProtocol_HandShake(hProtocol, 0, pstTCPChannel->pstCommunicationInfo->nChannelId);
	ERRIFGOTO(result, _EXIT);

	pstTCPChannel->pstCommunicationInfo->hSocket = hSocket;
	pstTCPChannel->pstCommunicationInfo->hProtocol = hProtocol;

	hSocket = NULL;
	hProtocol = NULL;

	result = ERR_UEM_NOERROR;
_EXIT:
	if(hProtocol != NULL)
	{
		UKUEMProtocol_Destroy(&hProtocol);
	}

	if(hSocket != NULL)
	{
		UCDynamicSocket_Destroy(&hSocket);
	}
	return result;
}


uem_result UKTCPSocketChannel_Initialize(SChannel *pstChannel)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STCPSocketChannel *pstTCPChannel = NULL;
	int nLoop = 0;

	pstTCPChannel = (STCPSocketChannel *) pstChannel->pChannelStruct;

	pstTCPChannel->pBuffer = NULL;
	pstTCPChannel->nBufLen = 0;
	pstTCPChannel->hReceivingThread = NULL;
	//pstTCPChannel->pstCommunicationInfo->
	//pstTCPChannel->pstClientInfo;
	result = UCThreadMutex_Create(&(pstTCPChannel->hMutex));
	ERRIFGOTO(result, _EXIT);

	if(pstTCPChannel->pstCommunicationInfo == NULL)
	{
		for(nLoop = 0 ; nLoop < g_nExternalCommunicationInfoNum; nLoop++)
		{
			if(g_astExternalCommunicationInfo[nLoop].nChannelId == pstChannel->nChannelIndex)
			{
				pstTCPChannel->pstCommunicationInfo = &g_astExternalCommunicationInfo[nLoop];
				break;
			}
		}
	}

	switch(pstChannel->enType)
	{
	case COMMUNICATION_TYPE_TCP_CLIENT_READER:
		// connect
		result = connectToServer(pstTCPChannel);
		ERRIFGOTO(result, _EXIT);
		break;
	case COMMUNICATION_TYPE_TCP_CLIENT_WRITER:
		// connect and create receive thread
		result = connectToServer(pstTCPChannel);
		ERRIFGOTO(result, _EXIT);

		result = UKChannelMemory_Initialize(pstChannel, pstTCPChannel->pstInternalChannel);
		ERRIFGOTO(result, _EXIT);

		result = createReceiverThread(pstChannel);
		ERRIFGOTO(result, _EXIT);
		break;
	case COMMUNICATION_TYPE_TCP_SERVER_READER:
		// do nothing
		break;
	case COMMUNICATION_TYPE_TCP_SERVER_WRITER:
		result = UKChannelMemory_Initialize(pstChannel, pstTCPChannel->pstInternalChannel);
		ERRIFGOTO(result, _EXIT);

		// create receive thread
		result = createReceiverThread(pstChannel);
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

static uem_result sendAndCheckResult(STCPSocketChannel *pstTCPChannel, HUEMProtocol hProtocol, int *pnReturnValue)
{
	uem_result result = ERR_UEM_UNKNOWN;
	EProtocolError enErrorCode = ERR_UEMPROTOCOL_INTERNAL;

	result = UKUEMProtocol_Send(hProtocol);
	ERRIFGOTO(result, _EXIT);

	do
	{
		if(pstTCPChannel->bChannelExit == TRUE)
		{
			ERRASSIGNGOTO(result, ERR_UEM_SUSPEND, _EXIT);
		}
		result = UKUEMProtocol_Receive(hProtocol);
		ERRIFGOTO(result, _EXIT);
	}while(result == ERR_UEM_NET_TIMEOUT);

	result = UKUEMProtocol_GetResultFromReceivedData(hProtocol, &enErrorCode, pnReturnValue);
	ERRIFGOTO(result, _EXIT);

	if(enErrorCode != ERR_UEMPROTOCOL_NOERROR)
	{
		UEM_DEBUG_PRINT("Error is retrieved from TCP writer\n");
		ERRASSIGNGOTO(result, ERR_UEM_INTERNAL_FAIL, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKTCPSocketChannel_ReadFromQueue(SChannel *pstChannel, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STCPSocketChannel *pstTCPChannel = NULL;
	HUEMProtocol hProtocol = NULL;
	int nDataRead = 0;
	void *pBody = NULL;

	pstTCPChannel = (STCPSocketChannel *) pstChannel->pChannelStruct;

	hProtocol = pstTCPChannel->pstCommunicationInfo->hProtocol;

	result = UKUEMProtocol_SetReadQueueRequest(hProtocol, nChunkIndex, nDataToRead);
	ERRIFGOTO(result, _EXIT);

	result = sendAndCheckResult(pstTCPChannel, hProtocol, NULL);
	ERRIFGOTO(result, _EXIT);

	result = UKUEMProtocol_GetBodyDataFromReceivedData(hProtocol, &nDataRead, &pBody);
	ERRIFGOTO(result, _EXIT);

	// TODO: Using memory system is needed
	UC_memcpy(pBuffer, pBody, nDataRead);

	*pnDataRead = nDataRead;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKTCPSocketChannel_ReadFromBuffer(SChannel *pstChannel, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STCPSocketChannel *pstTCPChannel = NULL;
	HUEMProtocol hProtocol = NULL;
	int nDataRead = 0;
	void *pBody = NULL;

	pstTCPChannel = (STCPSocketChannel *) pstChannel->pChannelStruct;

	hProtocol = pstTCPChannel->pstCommunicationInfo->hProtocol;

	result = UKUEMProtocol_SetReadBufferRequest(hProtocol, nChunkIndex, nDataToRead);
	ERRIFGOTO(result, _EXIT);

	result = sendAndCheckResult(hProtocol, NULL);
	ERRIFGOTO(result, _EXIT);

	result = UKUEMProtocol_GetBodyDataFromReceivedData(hProtocol, &nDataRead, &pBody);
	ERRIFGOTO(result, _EXIT);

	// TODO: Using memory system is needed
	UC_memcpy(pBuffer, pBody, nDataRead);

	*pnDataRead = nDataRead;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKTCPSocketChannel_GetAvailableChunk (SChannel *pstChannel, OUT int *pnChunkIndex)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STCPSocketChannel *pstTCPChannel = NULL;
	HUEMProtocol hProtocol = NULL;
	int nAvailableIndex = 0;

	pstTCPChannel = (STCPSocketChannel *) pstChannel->pChannelStruct;

	hProtocol = pstTCPChannel->pstCommunicationInfo->hProtocol;

	result = UKUEMProtocol_SetAvailableIndexRequest(hProtocol);
	ERRIFGOTO(result, _EXIT);

	result = sendAndCheckResult(hProtocol, &nAvailableIndex);
	ERRIFGOTO(result, _EXIT);

	*pnChunkIndex = nAvailableIndex;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKTCPSocketChannel_GetNumOfAvailableData (SChannel *pstChannel, IN int nChunkIndex, OUT int *pnDataNum)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STCPSocketChannel *pstTCPChannel = NULL;
	HUEMProtocol hProtocol = NULL;
	int nDataNum = 0;

	pstTCPChannel = (STCPSocketChannel *) pstChannel->pChannelStruct;

	hProtocol = pstTCPChannel->pstCommunicationInfo->hProtocol;

	result = UKUEMProtocol_SetAvailableDataRequest(hProtocol, nChunkIndex);
	ERRIFGOTO(result, _EXIT);

	result = sendAndCheckResult(hProtocol, &nDataNum);
	ERRIFGOTO(result, _EXIT);

	*pnDataNum = nDataNum;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


// same to shared memory channel write
uem_result UKTCPSocketChannel_WriteToQueue (SChannel *pstChannel, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STCPSocketChannel *pstTCPChannel = NULL;

	pstTCPChannel = (STCPSocketChannel *) pstChannel->pChannelStruct;

	result = UKChannelMemory_WriteToQueue(pstChannel, pstTCPChannel->pstInternalChannel, pBuffer, nDataToWrite, nChunkIndex, pnDataWritten);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


// same to shared memory channel write
uem_result UKTCPSocketChannel_WriteToBuffer (SChannel *pstChannel, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STCPSocketChannel *pstTCPChannel = NULL;

	pstTCPChannel = (STCPSocketChannel *) pstChannel->pChannelStruct;

	result = UKChannelMemory_WriteToBuffer(pstChannel, pstTCPChannel->pstInternalChannel, pBuffer, nDataToWrite, nChunkIndex, pnDataWritten);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKTCPSocketChannel_SetExit(SChannel *pstChannel, int nExitFlag)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STCPSocketChannel *pstTCPChannel = NULL;

	pstTCPChannel = (STCPSocketChannel *) pstChannel->pChannelStruct;

	pstTCPChannel->bChannelExit = TRUE;

	switch(pstChannel->enType)
	{
	case COMMUNICATION_TYPE_TCP_CLIENT_READER:
	case COMMUNICATION_TYPE_TCP_SERVER_READER:
		// do nothing
		break;
	case COMMUNICATION_TYPE_TCP_SERVER_WRITER:
	case COMMUNICATION_TYPE_TCP_CLIENT_WRITER:
		result = UKChannelMemory_SetExit(pstChannel, pstTCPChannel->pstInternalChannel, nExitFlag);
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


uem_result UKTCPSocketChannel_ClearExit(SChannel *pstChannel, int nExitFlag)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STCPSocketChannel *pstTCPChannel = NULL;

	pstTCPChannel = (STCPSocketChannel *) pstChannel->pChannelStruct;

	pstTCPChannel->bChannelExit = FALSE;

	switch(pstChannel->enType)
	{
	case COMMUNICATION_TYPE_TCP_CLIENT_READER:
	case COMMUNICATION_TYPE_TCP_SERVER_READER:
		// do nothing
		break;
	case COMMUNICATION_TYPE_TCP_SERVER_WRITER:
	case COMMUNICATION_TYPE_TCP_CLIENT_WRITER:
		result = UKChannelMemory_ClearExit(pstChannel, pstTCPChannel->pstInternalChannel, nExitFlag);
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


uem_result UKTCPSocketChannel_Finalize(SChannel *pstChannel)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STCPSocketChannel *pstTCPChannel = NULL;

	pstTCPChannel = (STCPSocketChannel *) pstChannel->pChannelStruct;

	//UKTCPSocketChannel_SetExit(pstChannel, EXIT_FLAG_READ | EXIT_FLAG_WRITE);

	switch(pstChannel->enType)
	{
	case COMMUNICATION_TYPE_TCP_CLIENT_READER:
		// disconnect
		result = disconnectToServer(pstTCPChannel);
		ERRIFGOTO(result, _EXIT);
		break;
	case COMMUNICATION_TYPE_TCP_CLIENT_WRITER:
		// destroy receiver thread
		result = destroyReceiverThread(pstChannel);
		ERRIFGOTO(result, _EXIT);

		result = UKChannelMemory_Finalize(pstChannel, pstTCPChannel->pstInternalChannel);
		ERRIFGOTO(result, _EXIT);

		result = disconnectToServer(pstTCPChannel);
		ERRIFGOTO(result, _EXIT);
		break;
	case COMMUNICATION_TYPE_TCP_SERVER_READER:
		// do nothing
		break;
	case COMMUNICATION_TYPE_TCP_SERVER_WRITER:
		// destroy receiver thread
		result = destroyReceiverThread(pstChannel);
		ERRIFGOTO(result, _EXIT);

		result = UKChannelMemory_Finalize(pstChannel, pstTCPChannel->pstInternalChannel);
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

