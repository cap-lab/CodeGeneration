/*
 * UKSerialChannel.c
 *
 *  Created on: 2019. 02. 18., modified from UKSerialChannel.c
 *      Author: dowhan1128
 */


#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <UCTime.h>
#include <UCAlloc.h>

//#include <uem_serial_data.h>
#include <uem_bluetooth_data.h>

#include <UKChannelMemory.h>

#include <UKSerialCommunicationManager.h>
#include <UKSerialChannel.h>

#define SERIAL_COMMUNICATION_CHANNEL_QUEUE_SIZE (1)
#define REMOTE_REQUEST_WAIT_SLEEP_TIME (100)
#define REMOTE_REQUEST_WAIT_RETRY_COUNT (30)

uem_result UKSerialChannel_Clear(SChannel *pstChannel)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSerialWriterChannel *pstWriterChannel = NULL;

	switch(pstChannel->enType)
	{
	case COMMUNICATION_TYPE_SERIAL_MASTER_READER:
	case COMMUNICATION_TYPE_SERIAL_SLAVE_READER:
		// do nothing
		break;
	case COMMUNICATION_TYPE_SERIAL_MASTER_WRITER:
	case COMMUNICATION_TYPE_SERIAL_SLAVE_WRITER:
		pstWriterChannel = (SSerialWriterChannel *) pstChannel->pChannelStruct;
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


static uem_result getChannelQueueFromSerialCommunicationManager(IN uem_bool *pbExitFlag, SSerialInfo *pstSerialInfo, int nChannelId, OUT HFixedSizeQueue *phRecevingQueue)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = UKSerialCommunicationManager_SetChannel(pstSerialInfo->hManager, nChannelId);
	ERRIFGOTO(result, _EXIT);

	result = UKSerialCommunicationManager_GetChannelQueue(pstSerialInfo->hManager, nChannelId, phRecevingQueue);
	ERRIFGOTO(result, _EXIT);

	// wait until serial manager is initialized
	while(pstSerialInfo->bInitialized == FALSE)
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


static uem_result reallocTempBuffer(SSerialWriterChannel *pstWriterChannel, int nTargetSize)
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

static uem_result handleReadQueue(SChannel *pstChannel, SCommunicationQueueItem *pstReceivedItem)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSerialWriterChannel *pstWriterChannel = NULL;
	SSerialInfo *pstSerialInfo = NULL;
	SCommunicationQueueItem stItemToSend;
	int nDataToRead = 0;
	int nDataRead = 0;

	pstWriterChannel = (SSerialWriterChannel *) pstChannel->pChannelStruct;
	pstSerialInfo = (SSerialInfo *) pstWriterChannel->pConnectionInfo;

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

	result = UKSerialCommunicationManager_PutItemToSend(pstSerialInfo->hManager, &stItemToSend);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static uem_result handleReadBuffer(SChannel *pstChannel, SCommunicationQueueItem *pstReceivedItem)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSerialWriterChannel *pstWriterChannel = NULL;
	SSerialInfo *pstSerialInfo = NULL;
	SCommunicationQueueItem stItemToSend;
	int nDataToRead = 0;
	int nDataRead = 0;

	pstWriterChannel = (SSerialWriterChannel *) pstChannel->pChannelStruct;
	pstSerialInfo = (SSerialInfo *) pstWriterChannel->pConnectionInfo;

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

	result = UKSerialCommunicationManager_PutItemToSend(pstSerialInfo->hManager, &stItemToSend);
	ERRIFGOTO(result, _EXIT);


	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result notifyAndPushError(SChannel *pstChannel, SCommunicationQueueItem *pstReceivedItem)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSerialWriterChannel *pstWriterChannel = NULL;
	SSerialInfo *pstSerialInfo = NULL;
	SCommunicationQueueItem stItemToSend;
	SCommunicationQueueItem *pstItem = NULL;

	pstItem = &stItemToSend;

	pstWriterChannel = (SSerialWriterChannel *) pstChannel->pChannelStruct;
	pstSerialInfo = (SSerialInfo *) pstWriterChannel->pConnectionInfo;

	pstItem->enMessageType = MESSAGE_TYPE_NONE;
	pstItem->nChannelId = pstChannel->nChannelIndex;
	pstItem->uDetailItem.stResponse.enRequestMessageType = pstReceivedItem->enMessageType;
	pstItem->uDetailItem.stResponse.nReturnValue = 0;
	pstItem->uDetailItem.stResponse.pData = NULL;
	pstItem->uDetailItem.stResponse.nDataSize = 0;

	result = UKSerialCommunicationManager_PutItemToSend(pstSerialInfo->hManager, pstItem);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static uem_result handleAvailableData(SChannel *pstChannel, SCommunicationQueueItem *pstReceivedItem)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSerialWriterChannel *pstWriterChannel = NULL;
	SSerialInfo *pstSerialInfo = NULL;
	int nDataNum = 0;
	SCommunicationQueueItem stItemToSend;

	pstWriterChannel = (SSerialWriterChannel *) pstChannel->pChannelStruct;
	pstSerialInfo = (SSerialInfo*) pstWriterChannel->pConnectionInfo;

	stItemToSend.enMessageType = MESSAGE_TYPE_RESULT;
	stItemToSend.nChannelId = pstChannel->nChannelIndex;
	stItemToSend.uDetailItem.stResponse.enRequestMessageType = pstReceivedItem->enMessageType;
	stItemToSend.uDetailItem.stResponse.nReturnValue = 0;
	stItemToSend.uDetailItem.stResponse.pData = NULL;
	stItemToSend.uDetailItem.stResponse.nDataSize = 0;

	result = UKChannelMemory_GetNumOfAvailableData(pstChannel, pstWriterChannel->pstInternalChannel, 0, &nDataNum);
	ERRIFGOTO(result, _EXIT);

	stItemToSend.uDetailItem.stResponse.nReturnValue = nDataNum;

	result = UKSerialCommunicationManager_PutItemToSend(pstSerialInfo->hManager, &stItemToSend);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result handleRequestFromReader(SCommunicationQueueItem *pstReceivedItem, SChannel *pstChannel)
{
	uem_result result = ERR_UEM_UNKNOWN;

	if(pstChannel->nChannelIndex != pstReceivedItem->nChannelId)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
	}

	switch(pstReceivedItem->enMessageType)
	{
	case MESSAGE_TYPE_READ_QUEUE:
		result = handleReadQueue(pstChannel, pstReceivedItem);
		ERRIFGOTO(result, _EXIT);
		break;
	case MESSAGE_TYPE_READ_BUFFER:
		result = handleReadBuffer(pstChannel, pstReceivedItem);
		ERRIFGOTO(result, _EXIT);
		break;
	case MESSAGE_TYPE_AVAILABLE_DATA:
		result = handleAvailableData(pstChannel, pstReceivedItem);
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

static void *channelReceiverHandlingThread(void *pData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SChannel *pstChannel = NULL;
	SSerialWriterChannel *pstWriterChannel = NULL;
	SCommunicationQueueItem stReceivedItem;
	int nElementSize;

	pstChannel = (SChannel *) pData;
	pstWriterChannel = (SSerialWriterChannel *) pstChannel->pChannelStruct;

	while(pstWriterChannel->bChannelExit == FALSE)
	{
		result = UCFixedSizeQueue_GetItem(pstWriterChannel->hRequestQueue, (void *)&stReceivedItem, &nElementSize);
		if(result == ERR_UEM_TIME_EXPIRED)
		{
			continue;
		}
		ERRIFGOTO(result, _EXIT);

		result = handleRequestFromReader(&stReceivedItem, pstChannel);
		if(result != ERR_UEM_NOERROR)
		{
			result = notifyAndPushError(pstChannel, &stReceivedItem);
			ERRIFGOTO(result, _EXIT);
		}
		ERRIFGOTO(result, _EXIT);
	}

_EXIT:
	if(result != ERR_UEM_NOERROR && pstWriterChannel->bChannelExit == FALSE)
	{
		UEM_DEBUG_PRINT("channelReceiverHandlingThread is exited with error: %08x\n", result);
	}
	return NULL;
}


static uem_result createReceiverThread(SChannel *pstChannel)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSerialWriterChannel *pstWriterChannel = NULL;

	pstWriterChannel = (SSerialWriterChannel *) pstChannel->pChannelStruct;

	result = UCThread_Create(channelReceiverHandlingThread, (void *) pstChannel, &(pstWriterChannel->hReceivingThread));
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result destroyReceiverThread(SChannel *pstChannel)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSerialWriterChannel *pstWriterChannel = NULL;

	pstWriterChannel = (SSerialWriterChannel *) pstChannel->pChannelStruct;

	result = UCThread_Destroy(&(pstWriterChannel->hReceivingThread), FALSE, 0);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKSerialChannel_Initialize(SChannel *pstChannel)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSerialWriterChannel *pstWriterChannel = NULL;
	SSerialReaderChannel *pstReaderChannel = NULL;
	SSerialInfo *pstSerialInfo = NULL;

	switch(pstChannel->enType)
	{
	case COMMUNICATION_TYPE_SERIAL_MASTER_READER:
	case COMMUNICATION_TYPE_SERIAL_SLAVE_READER:
		// connect
		pstReaderChannel = (SSerialReaderChannel *) pstChannel->pChannelStruct;
		pstSerialInfo = (SSerialInfo *) pstReaderChannel->pConnectionInfo;

		result = getChannelQueueFromSerialCommunicationManager(&(pstReaderChannel->bChannelExit), pstSerialInfo, pstChannel->nChannelIndex, &(pstReaderChannel->hResponseQueue));
		ERRIFGOTO(result, _EXIT);
		break;
	case COMMUNICATION_TYPE_SERIAL_MASTER_WRITER:
	case COMMUNICATION_TYPE_SERIAL_SLAVE_WRITER:
		// connect and create receive thread
		pstWriterChannel = (SSerialWriterChannel *) pstChannel->pChannelStruct;
		pstSerialInfo = (SSerialInfo *) pstWriterChannel->pConnectionInfo;

		result = UKChannelMemory_Initialize(pstChannel, pstWriterChannel->pstInternalChannel);
		ERRIFGOTO(result, _EXIT);

		result = getChannelQueueFromSerialCommunicationManager(&(pstWriterChannel->bChannelExit), pstSerialInfo, pstChannel->nChannelIndex, &(pstWriterChannel->hRequestQueue));
		ERRIFGOTO(result, _EXIT);

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


uem_result UKSerialChannel_ReadFromQueue(SChannel *pstChannel, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSerialReaderChannel *pstSerialReaderChannel = NULL;
	int nDataRead = 0;
	void *pBody = NULL;
	SSerialInfo *pstSerialInfo = NULL;
	SCommunicationQueueItem stItem;
	SCommunicationQueueItem stResponseItem;
	int nElementSize = 0;

	pstSerialReaderChannel = (SSerialReaderChannel *) pstChannel->pChannelStruct;
	pstSerialInfo = (SSerialInfo *) pstSerialReaderChannel->pConnectionInfo;

	stItem.enMessageType = MESSAGE_TYPE_READ_QUEUE;
	stItem.nChannelId = pstChannel->nChannelIndex;
	stItem.uDetailItem.stRequest.nRequestDataSize = nDataToRead;

	result = UKSerialCommunicationManager_PutItemToSend(pstSerialInfo->hManager, &stItem);
	ERRIFGOTO(result, _EXIT);

	// blocking happened here
	do {
		result = UCFixedSizeQueue_GetItem(pstSerialReaderChannel->hResponseQueue, &stResponseItem, &nElementSize);
	}while(result == ERR_UEM_TIME_EXPIRED && pstSerialReaderChannel->bChannelExit == FALSE);
	ERRIFGOTO(result, _EXIT);

	if(stResponseItem.enMessageType != MESSAGE_TYPE_RESULT ||
		stResponseItem.nChannelId != pstChannel->nChannelIndex ||
		stResponseItem.uDetailItem.stResponse.enRequestMessageType != MESSAGE_TYPE_READ_QUEUE)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
	}

	pBody = stResponseItem.uDetailItem.stResponse.pData;
	nDataRead = stResponseItem.uDetailItem.stResponse.nDataSize;

	if(pBuffer != NULL)
	{
		result = pstSerialReaderChannel->pstReaderAccess->fnCopyFromMemory(pBuffer, pBody, nDataRead);
		ERRIFGOTO(result, _EXIT);
	}

	*pnDataRead = nDataRead;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKSerialChannel_ReadFromBuffer(SChannel *pstChannel, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSerialReaderChannel *pstSerialReaderChannel = NULL;
	int nDataRead = 0;
	void *pBody = NULL;
	SSerialInfo *pstSerialInfo = NULL;
	SCommunicationQueueItem stItem;
	SCommunicationQueueItem stResponseItem;
	int nElementSize = 0;

	pstSerialReaderChannel = (SSerialReaderChannel *) pstChannel->pChannelStruct;
	pstSerialInfo = (SSerialInfo *) pstSerialReaderChannel->pConnectionInfo;

	stItem.enMessageType = MESSAGE_TYPE_READ_BUFFER;
	stItem.nChannelId = pstChannel->nChannelIndex;
	stItem.uDetailItem.stRequest.nRequestDataSize = nDataToRead;

	result = UKSerialCommunicationManager_PutItemToSend(pstSerialInfo->hManager, &stItem);
	ERRIFGOTO(result, _EXIT);

	// blocking happened here
	do {
		result = UCFixedSizeQueue_GetItem(pstSerialReaderChannel->hResponseQueue, &stResponseItem, &nElementSize);
	}while(result == ERR_UEM_TIME_EXPIRED && pstSerialReaderChannel->bChannelExit == FALSE);
	ERRIFGOTO(result, _EXIT);

	if(stResponseItem.enMessageType != MESSAGE_TYPE_RESULT ||
		stResponseItem.nChannelId != pstChannel->nChannelIndex ||
		stResponseItem.uDetailItem.stResponse.enRequestMessageType != MESSAGE_TYPE_READ_BUFFER)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
	}

	pBody = stResponseItem.uDetailItem.stResponse.pData;
	nDataRead = stResponseItem.uDetailItem.stResponse.nDataSize;

	result = pstSerialReaderChannel->pstReaderAccess->fnCopyFromMemory(pBuffer, pBody, nDataRead);
	ERRIFGOTO(result, _EXIT);

	*pnDataRead = nDataRead;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKSerialChannel_GetAvailableChunk (SChannel *pstChannel, OUT int *pnChunkIndex)
{
	uem_result result = ERR_UEM_UNKNOWN;

	// TODO: support later?
	// always returns "0" for UEMLiteProtocol
	*pnChunkIndex = 0;

	result = ERR_UEM_NOERROR;

	return result;
}

uem_result UKSerialChannel_GetNumOfAvailableData (SChannel *pstChannel, IN int nChunkIndex, OUT int *pnDataNum)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSerialReaderChannel *pstSerialReaderChannel = NULL;
	SSerialWriterChannel *pstWriterChannel = NULL;
	SSerialInfo *pstSerialInfo = NULL;
	SCommunicationQueueItem stItem;
	SCommunicationQueueItem stResponseItem;
	int nElementSize = 0;

	switch(pstChannel->enType)
	{
	case COMMUNICATION_TYPE_SERIAL_MASTER_READER:
	case COMMUNICATION_TYPE_SERIAL_SLAVE_READER:
		pstSerialReaderChannel = (SSerialReaderChannel *) pstChannel->pChannelStruct;
		pstSerialInfo = (SSerialInfo *) pstSerialReaderChannel->pConnectionInfo;

		stItem.enMessageType = MESSAGE_TYPE_AVAILABLE_DATA;
		stItem.nChannelId = pstChannel->nChannelIndex;
		stItem.uDetailItem.stRequest.nRequestDataSize = 0; // not used

		result = UKSerialCommunicationManager_PutItemToSend(pstSerialInfo->hManager, &stItem);
		ERRIFGOTO(result, _EXIT);

		// blocking happened here
		do {
			result = UCFixedSizeQueue_GetItem(pstSerialReaderChannel->hResponseQueue, &stResponseItem, &nElementSize);
		}while(result == ERR_UEM_TIME_EXPIRED && pstSerialReaderChannel->bChannelExit == FALSE);
		ERRIFGOTO(result, _EXIT);

		if(stResponseItem.enMessageType != MESSAGE_TYPE_RESULT ||
			stResponseItem.nChannelId != pstChannel->nChannelIndex ||
			stResponseItem.uDetailItem.stResponse.enRequestMessageType != MESSAGE_TYPE_AVAILABLE_DATA)
		{
			ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
		}

		*pnDataNum = stResponseItem.uDetailItem.stResponse.nReturnValue;
		break;
	case COMMUNICATION_TYPE_SERIAL_MASTER_WRITER:
	case COMMUNICATION_TYPE_SERIAL_SLAVE_WRITER:
		pstWriterChannel = (SSerialWriterChannel *) pstChannel->pChannelStruct;

		result = UKChannelMemory_GetNumOfAvailableData(pstChannel, pstWriterChannel->pstInternalChannel, nChunkIndex, pnDataNum);
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


// same to shared memory channel write
uem_result UKSerialChannel_WriteToQueue (SChannel *pstChannel, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSerialWriterChannel *pstWriterChannel = NULL;

	pstWriterChannel = (SSerialWriterChannel *) pstChannel->pChannelStruct;

	result = UKChannelMemory_WriteToQueue(pstChannel, pstWriterChannel->pstInternalChannel, pBuffer, nDataToWrite, nChunkIndex, pnDataWritten);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKSerialChannel_FillInitialData(SChannel *pstChannel)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSerialWriterChannel *pstWriterChannel = NULL;

	pstWriterChannel = (SSerialWriterChannel *) pstChannel->pChannelStruct;

	result = UKChannelMemory_FillInitialData(pstChannel, pstWriterChannel->pstInternalChannel);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


// same to shared memory channel write
uem_result UKSerialChannel_WriteToBuffer (SChannel *pstChannel, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSerialWriterChannel *pstWriterChannel = NULL;

	pstWriterChannel = (SSerialWriterChannel *) pstChannel->pChannelStruct;

	result = UKChannelMemory_WriteToBuffer(pstChannel, pstWriterChannel->pstInternalChannel, pBuffer, nDataToWrite, nChunkIndex, pnDataWritten);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKSerialChannel_SetExit(SChannel *pstChannel, int nExitFlag)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSerialWriterChannel *pstWriterChannel = NULL;
	SSerialReaderChannel *pstReaderChannel = NULL;

	switch(pstChannel->enType)
	{
	case COMMUNICATION_TYPE_SERIAL_MASTER_READER:
	case COMMUNICATION_TYPE_SERIAL_SLAVE_READER:
		pstReaderChannel = (SSerialReaderChannel *) pstChannel->pChannelStruct;
		pstReaderChannel->bChannelExit = TRUE;
		break;
	case COMMUNICATION_TYPE_SERIAL_MASTER_WRITER:
	case COMMUNICATION_TYPE_SERIAL_SLAVE_WRITER:
		pstWriterChannel = (SSerialWriterChannel *) pstChannel->pChannelStruct;
		pstWriterChannel->bChannelExit = TRUE;
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


uem_result UKSerialChannel_ClearExit(SChannel *pstChannel, int nExitFlag)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSerialWriterChannel *pstWriterChannel = NULL;
	SSerialReaderChannel *pstReaderChannel = NULL;

	switch(pstChannel->enType)
	{
	case COMMUNICATION_TYPE_SERIAL_MASTER_READER:
	case COMMUNICATION_TYPE_SERIAL_SLAVE_READER:
		pstReaderChannel = (SSerialReaderChannel *) pstChannel->pChannelStruct;
		pstReaderChannel->bChannelExit = TRUE;
		break;
	case COMMUNICATION_TYPE_SERIAL_MASTER_WRITER:
	case COMMUNICATION_TYPE_SERIAL_SLAVE_WRITER:
		pstWriterChannel = (SSerialWriterChannel *) pstChannel->pChannelStruct;
		pstWriterChannel->bChannelExit = FALSE;
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


uem_result UKSerialChannel_Finalize(SChannel *pstChannel)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSerialWriterChannel *pstWriterChannel = NULL;
	SSerialReaderChannel *pstReaderChannel = NULL;
	SSerialInfo *pstSerialInfo = NULL;
	int nRetryCount = 0;

	UKSerialChannel_SetExit(pstChannel, EXIT_FLAG_READ | EXIT_FLAG_WRITE);

	switch(pstChannel->enType)
	{
	case COMMUNICATION_TYPE_SERIAL_MASTER_READER:
	case COMMUNICATION_TYPE_SERIAL_SLAVE_READER:
		pstReaderChannel = (SSerialReaderChannel *) pstChannel->pChannelStruct;
		pstSerialInfo = (SSerialInfo *) pstReaderChannel->pConnectionInfo;
		result = UKSerialCommunicationManager_ReleaseChannel(pstSerialInfo->hManager, pstChannel->nChannelIndex);
		ERRIFGOTO(result, _EXIT);
		break;
	case COMMUNICATION_TYPE_SERIAL_MASTER_WRITER:
	case COMMUNICATION_TYPE_SERIAL_SLAVE_WRITER:
		pstWriterChannel = (SSerialWriterChannel *) pstChannel->pChannelStruct;
		pstSerialInfo = (SSerialInfo *) pstWriterChannel->pConnectionInfo;

		// wait until the reader consumes the data for several seconds
		while(pstWriterChannel->pstInternalChannel->nDataLen > 0 &&
				nRetryCount < REMOTE_REQUEST_WAIT_RETRY_COUNT)
		{
			nRetryCount++;
			UCTime_Sleep(REMOTE_REQUEST_WAIT_SLEEP_TIME);
		}

		result = UKSerialCommunicationManager_ReleaseChannel(pstSerialInfo->hManager, pstChannel->nChannelIndex);
		ERRIFGOTO(result, _EXIT);

		// destroy receiver thread
		result = destroyReceiverThread(pstChannel);
		ERRIFGOTO(result, _EXIT);

		result = UKChannelMemory_Finalize(pstChannel, pstWriterChannel->pstInternalChannel);
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


