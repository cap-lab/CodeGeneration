/*
 * UKSerialChannel.c
 *
 *  Created on: 2018. 10. 25.
 *      Author: jej
 */


#ifdef HAVE_CONFIG_H
#include <config.h>
#endif


#include <uem_common.h>

#include <uem_bluetooth_data.h>

#include <UKChannelMemory.h>

#include <UKSerialModule.h>
#include <UKUEMLiteProtocol.h>


#define SERIAL_COMMUNICATION_CHANNEL_QUEUE_SIZE (1)
#define REMOTE_REQUEST_WAIT_SLEEP_TIME (100)
#define REMOTE_REQUEST_WAIT_RETRY_COUNT (30)

uem_result UKSerialChannel_Clear(SChannel *pstChannel)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSerialChannel *pstSerialChannel = NULL;

	switch(pstChannel->enType)
	{
	case COMMUNICATION_TYPE_REMOTE_WRITER:
	case COMMUNICATION_TYPE_REMOTE_READER:
		pstSerialChannel = (SSerialChannel *) pstChannel->pChannelStruct;
		result = UKChannelMemory_Clear(pstChannel, pstSerialChannel->pstInternalChannel);
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


uem_result UKSerialChannel_Initialize(SChannel *pstChannel)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSerialChannel *pstSerialChannel = NULL;

	switch(pstChannel->enType)
	{
	case COMMUNICATION_TYPE_REMOTE_WRITER:
	case COMMUNICATION_TYPE_REMOTE_READER:
		pstSerialChannel = (SSerialChannel *) pstChannel->pChannelStruct;

		result = UKChannelMemory_Initialize(pstChannel, pstSerialChannel->pstInternalChannel);
		ERRIFGOTO(result, _EXIT);

		result = UKSerialModule_SetChannel(pstSerialChannel->pstSerialInfo, pstChannel);
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
	SSerialChannel *pstSerialChannel = NULL;
	int nDataCanRead = 0;

	pstSerialChannel = (SSerialChannel *) pstChannel->pChannelStruct;

	result = UKChannelMemory_GetNumOfAvailableData(pstChannel, pstSerialChannel->pstInternalChannel, nChunkIndex, &nDataCanRead);
	ERRIFGOTO(result, _EXIT);

	if(nDataCanRead >= nDataToRead)
	{
		result = UKChannelMemory_ReadFromQueue(pstChannel, pstSerialChannel->pstInternalChannel, pBuffer, nDataToRead, nChunkIndex, pnDataRead);
		ERRIFGOTO(result, _EXIT);
	}
	else
	{
		// if there is no pending request message, add a request
		if(pstSerialChannel->stRequestInfo.enMessageType == MESSAGE_TYPE_NONE)
		{
			pstSerialChannel->stRequestInfo.enMessageType = MESSAGE_TYPE_READ_QUEUE;
			pstSerialChannel->stRequestInfo.nDataToRead = nDataToRead;
		}

		*pnDataRead = 0;
		UEMASSIGNGOTO(result, ERR_UEM_READ_BLOCK, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKSerialChannel_ClearRequest(SChannel *pstChannel)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSerialChannel *pstSerialChannel = NULL;
	pstSerialChannel = (SSerialChannel *) pstChannel->pChannelStruct;

	pstSerialChannel->stRequestInfo.enMessageType = MESSAGE_TYPE_NONE;
	pstSerialChannel->stRequestInfo.nDataToRead = 0;

	result = ERR_UEM_NOERROR;

	return result;
}



uem_result UKSerialChannel_PendReadQueueRequest(SChannel *pstChannel, int nDataToRead)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSerialChannel *pstSerialChannel = NULL;
	pstSerialChannel = (SSerialChannel *) pstChannel->pChannelStruct;

	pstSerialChannel->stRequestInfo.enMessageType = MESSAGE_TYPE_READ_QUEUE;
	pstSerialChannel->stRequestInfo.nDataToRead = nDataToRead;

	result = ERR_UEM_NOERROR;

	return result;
}



uem_result UKSerialChannel_PendReadBufferRequest(SChannel *pstChannel, int nDataToRead)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSerialChannel *pstSerialChannel = NULL;
	pstSerialChannel = (SSerialChannel *) pstChannel->pChannelStruct;

	pstSerialChannel->stRequestInfo.enMessageType = MESSAGE_TYPE_READ_BUFFER;
	pstSerialChannel->stRequestInfo.nDataToRead = nDataToRead;

	result = ERR_UEM_NOERROR;

	return result;
}


static uem_result makeRequestToSend(SChannel *pstChannel)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSerialChannel *pstSerialChannel = NULL;

	pstSerialChannel = (SSerialChannel *) pstChannel->pChannelStruct;

	switch(pstSerialChannel->stRequestInfo.enMessageType)
	{
	case MESSAGE_TYPE_AVAILABLE_DATA:
		result = UKUEMLiteProtocol_SetAvailableDataRequest(pstChannel->nChannelIndex);
		ERRIFGOTO(result, _EXIT);
		break;
	case MESSAGE_TYPE_READ_BUFFER:
		result = UKUEMLiteProtocol_SetReadBufferRequest(pstChannel->nChannelIndex, pstSerialChannel->stRequestInfo.nDataToRead);
		ERRIFGOTO(result, _EXIT);
		break;
	case MESSAGE_TYPE_READ_QUEUE:
		result = UKUEMLiteProtocol_SetReadQueueRequest(pstChannel->nChannelIndex, pstSerialChannel->stRequestInfo.nDataToRead);
		ERRIFGOTO(result, _EXIT);
		break;
	case MESSAGE_TYPE_RESULT: // waiting for result
	case MESSAGE_TYPE_NONE:
		result = ERR_UEM_SKIP_THIS;
		break;
	default:
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
		break;
	}

	if(pstSerialChannel->stRequestInfo.enMessageType != MESSAGE_TYPE_NONE)
	{
		// set message type to result because it needs to wait for the result and avoid duplicate requests
		pstSerialChannel->stRequestInfo.enMessageType = MESSAGE_TYPE_RESULT;
	}
_EXIT:
	return result;
}



static uem_result handleReadQueueRequest(SChannel *pstChannel, IN OUT unsigned char *pBuffer, int nDataToRead)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSerialChannel *pstSerialChannel = NULL;
	int nDataCanRead = 0;
	int nDataRead = 0;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(pBuffer, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	pstSerialChannel = (SSerialChannel *) pstChannel->pChannelStruct;

	result = UKChannelMemory_GetNumOfAvailableData(pstChannel, pstSerialChannel->pstInternalChannel, 0, &nDataCanRead);
	ERRIFGOTO(result, _EXIT);

	if(nDataCanRead >= nDataToRead)
	{
		result = UKChannelMemory_ReadFromQueue(pstChannel, pstSerialChannel->pstInternalChannel, pBuffer, nDataToRead, 0, &nDataRead);
		ERRIFGOTO(result, _EXIT);

		// clear request
		pstSerialChannel->stRequestInfo.enMessageType = MESSAGE_TYPE_NONE;
	}
	else
	{
		UEMASSIGNGOTO(result, ERR_UEM_NOT_REACHED_YET, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result handleReadBufferRequest(SChannel *pstChannel, IN OUT unsigned char *pBuffer, int nDataToRead)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSerialChannel *pstSerialChannel = NULL;
	int nDataRead = 0;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(pBuffer, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	pstSerialChannel = (SSerialChannel *) pstChannel->pChannelStruct;

	result = UKChannelMemory_ReadFromBuffer(pstChannel, pstSerialChannel->pstInternalChannel, pBuffer, nDataToRead, 0, &nDataRead);
	ERRIFGOTO(result, _EXIT);

	// clear request
	pstSerialChannel->stRequestInfo.enMessageType = MESSAGE_TYPE_NONE;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}



static uem_result handleGetAvailableDataRequest(SChannel *pstChannel, OUT int *pnDataCanRead)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSerialChannel *pstSerialChannel = NULL;
	int nDataCanRead = 0;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(pnDataCanRead, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	pstSerialChannel = (SSerialChannel *) pstChannel->pChannelStruct;

	result = UKChannelMemory_GetNumOfAvailableData(pstChannel, pstSerialChannel->pstInternalChannel, 0, &nDataCanRead);
	ERRIFGOTO(result, _EXIT);

	// clear request
	pstSerialChannel->stRequestInfo.enMessageType = MESSAGE_TYPE_NONE;

	*pnDataCanRead = nDataCanRead;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}



static uem_result makeResultToSend(SChannel *pstChannel)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSerialChannel *pstSerialChannel = NULL;
	unsigned char *pBuffer = NULL;
	int nBufferSize = 0;
	int nDataCanRead = 0;

	pstSerialChannel = (SSerialChannel *) pstChannel->pChannelStruct;

	result = UKUEMLiteProtocol_GetResultBufferToSend(&pBuffer, &nBufferSize);
	ERRIFGOTO(result, _EXIT);

	switch(pstSerialChannel->stRequestInfo.enMessageType)
	{
	case MESSAGE_TYPE_AVAILABLE_DATA:
		result = handleGetAvailableDataRequest(pstChannel, &nDataCanRead);
		ERRIFGOTO(result, _EXIT);
		result = UKUEMLiteProtocol_SetResultMessage(MESSAGE_TYPE_AVAILABLE_DATA, pstChannel->nChannelIndex,
														ERR_UEMPROTOCOL_NOERROR, nDataCanRead);
		ERRIFGOTO(result, _EXIT);
		break;
	case MESSAGE_TYPE_READ_BUFFER:
		result = handleReadBufferRequest(pstChannel, pBuffer, pstSerialChannel->stRequestInfo.nDataToRead);
		ERRIFGOTO(result, _EXIT);
		if(result == ERR_UEM_NOERROR)
		{
			result = UKUEMLiteProtocol_SetResultMessageHeaderUsingBuffer(MESSAGE_TYPE_READ_BUFFER, pstChannel->nChannelIndex,
														ERR_UEMPROTOCOL_NOERROR, pstSerialChannel->stRequestInfo.nDataToRead);
			ERRIFGOTO(result, _EXIT);
		}
		break;
	case MESSAGE_TYPE_READ_QUEUE:
		result = handleReadQueueRequest(pstChannel, pBuffer, pstSerialChannel->stRequestInfo.nDataToRead);
		ERRIFGOTO(result, _EXIT);
		if(result == ERR_UEM_NOERROR)
		{
			result = UKUEMLiteProtocol_SetResultMessageHeaderUsingBuffer(MESSAGE_TYPE_READ_QUEUE, pstChannel->nChannelIndex,
														ERR_UEMPROTOCOL_NOERROR, pstSerialChannel->stRequestInfo.nDataToRead);
			//UEM_DEBUG_PRINT("result check, %d\n", result);
			ERRIFGOTO(result, _EXIT);
		}
		break;
	case MESSAGE_TYPE_NONE:
		// do nothing (no request is arrived here)
		result = ERR_UEM_SKIP_THIS;
		break;
	default:
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
		break;
	}

	// not set ERR_UEM_NOERROR here to preserve specific uem_result (ERR_UEM_NOT_REACHED_YET)
_EXIT:
	return result;
}





uem_result UKSerialChannel_HandleRequest(SChannel *pstChannel)
{
	uem_result result = ERR_UEM_UNKNOWN;

	switch(pstChannel->enType)
	{
	case COMMUNICATION_TYPE_REMOTE_READER:
		result = makeRequestToSend(pstChannel);
		ERRIFGOTO(result, _EXIT);
		break;
	case COMMUNICATION_TYPE_REMOTE_WRITER:
		result = makeResultToSend(pstChannel);
		ERRIFGOTO(result, _EXIT);
		break;
	default:
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
		break;
	}

	// skip setting ERR_UEM_NOERROR to preserve non-error uem_result
_EXIT:
	return result;
}



uem_result UKSerialChannel_PendGetAvailableDataRequest(SChannel *pstChannel)
{
	uem_result result = ERR_UEM_UNKNOWN;

	SSerialChannel *pstSerialChannel = NULL;
	pstSerialChannel = (SSerialChannel *) pstChannel->pChannelStruct;

	pstSerialChannel->stRequestInfo.enMessageType = MESSAGE_TYPE_AVAILABLE_DATA;
	pstSerialChannel->stRequestInfo.nDataToRead = 0;

	result = ERR_UEM_NOERROR;

	return result;
}



uem_result UKSerialChannel_ReadFromBuffer(SChannel *pstChannel, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSerialChannel *pstSerialChannel = NULL;

	pstSerialChannel = (SSerialChannel *) pstChannel->pChannelStruct;

	result = UKChannelMemory_ReadFromBuffer(pstChannel, pstSerialChannel->pstInternalChannel, pBuffer, nDataToRead, nChunkIndex, pnDataRead);
	ERRIFGOTO(result, _EXIT);

	// if there is no pending request message, add a request
	if(pstSerialChannel->stRequestInfo.enMessageType == MESSAGE_TYPE_NONE)
	{
		pstSerialChannel->stRequestInfo.enMessageType = MESSAGE_TYPE_READ_BUFFER;
		pstSerialChannel->stRequestInfo.nDataToRead = nDataToRead;
	}

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
	SSerialChannel *pstSerialChannel = NULL;
	int nDataCanRead = 0;

	pstSerialChannel = (SSerialChannel *) pstChannel->pChannelStruct;

	result = UKChannelMemory_GetNumOfAvailableData(pstChannel, pstSerialChannel->pstInternalChannel, nChunkIndex, &nDataCanRead);
	ERRIFGOTO(result, _EXIT);

	switch(pstChannel->enType)
	{
	case COMMUNICATION_TYPE_REMOTE_READER:
		if(pstSerialChannel->stRequestInfo.enMessageType == MESSAGE_TYPE_NONE)
		{
			pstSerialChannel->stRequestInfo.enMessageType = MESSAGE_TYPE_AVAILABLE_DATA;
			pstSerialChannel->stRequestInfo.nDataToRead = 0;
		}
		break;
	case COMMUNICATION_TYPE_REMOTE_WRITER:
		break;
	default:
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
		break;
	}

	*pnDataNum = nDataCanRead;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


// same to shared memory channel write
uem_result UKSerialChannel_WriteToQueue (SChannel *pstChannel, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSerialChannel *pstSerialChannel = NULL;

	pstSerialChannel = (SSerialChannel *) pstChannel->pChannelStruct;

	result = UKChannelMemory_WriteToQueue(pstChannel, pstSerialChannel->pstInternalChannel, pBuffer, nDataToWrite, nChunkIndex, pnDataWritten);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKSerialChannel_FillInitialData(SChannel *pstChannel)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSerialChannel *pstSerialChannel = NULL;

	pstSerialChannel = (SSerialChannel *) pstChannel->pChannelStruct;

	result = UKChannelMemory_FillInitialData(pstChannel, pstSerialChannel->pstInternalChannel);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


// same to shared memory channel write
uem_result UKSerialChannel_WriteToBuffer (SChannel *pstChannel, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSerialChannel *pstSerialChannel = NULL;

	pstSerialChannel = (SSerialChannel *) pstChannel->pChannelStruct;

	result = UKChannelMemory_WriteToBuffer(pstChannel, pstSerialChannel->pstInternalChannel, pBuffer, nDataToWrite, nChunkIndex, pnDataWritten);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKSerialChannel_SetExit(SChannel *pstChannel, int nExitFlag)
{
	return ERR_UEM_NOERROR;
}


uem_result UKSerialChannel_ClearExit(SChannel *pstChannel, int nExitFlag)
{
	return ERR_UEM_NOERROR;
}


uem_result UKSerialChannel_Finalize(SChannel *pstChannel)
{
	return ERR_UEM_NOERROR;
}


