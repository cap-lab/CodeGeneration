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
#define CONNECT_RETRY_COUNT (5)
#define SECOND_IN_MILLISECOND (1000)

uem_result UKTCPSocketChannel_Clear(SChannel *pstChannel)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STCPSocketChannel *pstTCPChannel = NULL;

	pstTCPChannel = (STCPSocketChannel *) pstChannel->pChannelStruct;

	result = UKChannelMemory_Clear(pstChannel, pstTCPChannel->pstInternalChannel);
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

	pstTCPChannel = (STCPSocketChannel *) pstChannel->pChannelStruct;

	result = UKUEMProtocol_GetRequestFromReceivedData(hProtocol, &enMessageType, &nParamNum, &panParam);
	ERRIFGOTO(result, _EXIT);

	switch(enMessageType)
	{
	case MESSAGE_TYPE_READ_QUEUE:

		break;
	case MESSAGE_TYPE_READ_BUFFER:
		panParam[READ_BUFFER_CHUNK_INDEX_INDEX];
		panParam[READ_BUFFER_SIZE_TO_READ_INDEX];
		result = UKChannelMemory_ReadFromBuffer(pstChannel, pstTCPChannel->pstInternalChannel, pBuffer, nDataToRead, nChunkIndex, pnDataRead)
		break;
	case MESSAGE_TYPE_AVAILABLE_INDEX:
		break;
	case MESSAGE_TYPE_AVAILABLE_DATA:
		break;
	default:
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

	while(g_bSystemExit == FALSE)
	{
		result = UKUEMProtocol_Receive(hProtocol);
		if(result == ERR_UEM_NET_TIMEOUT)
		{
			continue;
		}
		ERRIFGOTO(result, _EXIT);


	}

_EXIT:
	if(result != ERR_UEM_NOERROR && g_bSystemExit == FALSE)
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

	pstTCPChannel = (STCPSocketChannel *) pstChannel->pChannelStruct;

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
		break;
	case COMMUNICATION_TYPE_TCP_SERVER_READER:
		// do nothing
		break;
	case COMMUNICATION_TYPE_TCP_SERVER_WRITER:
		// create receive thread
		break;
	}
	// pstTCPChannel->

	result = UKChannelMemory_Initialize(pstChannel, pstTCPChannel->pstInternalChannel);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKTCPSocketChannel_ReadFromQueue(SChannel *pstChannel, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STCPSocketChannel *pstTCPChannel = NULL;

	pstTCPChannel = (STCPSocketChannel *) pstChannel->pChannelStruct;

	result = UKChannelMemory_ReadFromQueue(pstChannel, pstTCPChannel->pstInternalChannel, pBuffer, nDataToRead, nChunkIndex, pnDataRead);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKTCPSocketChannel_ReadFromBuffer(SChannel *pstChannel, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STCPSocketChannel *pstTCPChannel = NULL;

	pstTCPChannel = (STCPSocketChannel *) pstChannel->pChannelStruct;

	result = UKChannelMemory_ReadFromBuffer(pstChannel, pstTCPChannel->pstInternalChannel, pBuffer, nDataToRead, nChunkIndex, pnDataRead);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


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


uem_result UKTCPSocketChannel_GetAvailableChunk (SChannel *pstChannel, OUT int *pnChunkIndex)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STCPSocketChannel *pstTCPChannel = NULL;

	pstTCPChannel = (STCPSocketChannel *) pstChannel->pChannelStruct;

	result = UKChannelMemory_GetAvailableChunk(pstChannel, pstTCPChannel->pstInternalChannel, pnChunkIndex);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKTCPSocketChannel_GetNumOfAvailableData (SChannel *pstChannel, IN int nChunkIndex, OUT int *pnDataNum)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STCPSocketChannel *pstTCPChannel = NULL;

	pstTCPChannel = (STCPSocketChannel *) pstChannel->pChannelStruct;

	result = UKChannelMemory_GetNumOfAvailableData(pstChannel, pstTCPChannel->pstInternalChannel, nChunkIndex, pnDataNum);
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

	result = UKChannelMemory_SetExit(pstChannel, pstTCPChannel->pstInternalChannel, nExitFlag);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKTCPSocketChannel_ClearExit(SChannel *pstChannel, int nExitFlag)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STCPSocketChannel *pstTCPChannel = NULL;

	pstTCPChannel = (STCPSocketChannel *) pstChannel->pChannelStruct;

	result = UKChannelMemory_ClearExit(pstChannel, pstTCPChannel->pstInternalChannel, nExitFlag);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKTCPSocketChannel_Finalize(SChannel *pstChannel)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STCPSocketChannel *pstTCPChannel = NULL;

	pstTCPChannel = (STCPSocketChannel *) pstChannel->pChannelStruct;

	result = UKChannelMemory_Finalize(pstChannel, pstTCPChannel->pstInternalChannel);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

