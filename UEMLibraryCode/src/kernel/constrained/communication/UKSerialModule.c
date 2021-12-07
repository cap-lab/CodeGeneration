/*
 * UKSerialModule.c
 *
 *  Created on: 2018. 10. 23.
 *      Author: jej
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <UCSerial.h>
#include <UCTime.h>


#include <uem_bluetooth_data.h>
#include <uem_lite_protocol_data.h>

#include <UKUEMLiteProtocol.h>
#include <UKChannel.h>

#include <UKSerialChannel.h>

#define WAIT_SLEEP_TIME (50)

#define SLEEP_TIME_DECISION_CONSTANT (1152000)

#ifndef SERIAL_COMMUNICATION_DELAY_TIME
	#define SERIAL_COMMUNICATION_DELAY_TIME (SLEEP_TIME_DECISION_CONSTANT/DATA_SERIAL_DEFAULT_BAUD_RATE)
#endif

void clearChannelList(SSerialInfo *pstSerialInfo)
{
	int nLoop = 0;

	for(nLoop = 0; nLoop < pstSerialInfo->nMaxChannelAccessNum ; nLoop++)
	{
		pstSerialInfo->ppastAccessChannelList[nLoop] = NULL;
	}
}

uem_result UKSerialModule_Initialize()
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;
	int nAvailableSize = 0;
	EMessageType enMessageType;
	int nParamNum;
	short *pasParam = NULL;
	int nChannelId = INVALID_CHANNEL_ID;

	for(nLoop = 0 ; nLoop < g_nSerialSlaveNum ; nLoop++)
	{
		clearChannelList(&(g_astSerialSlaveInfo[nLoop]));
	}

	for(nLoop = 0 ; nLoop < g_nSerialMasterNum ; nLoop++)
	{
		clearChannelList(&(g_astSerialMasterInfo[nLoop]));
	}

	for(nLoop = 0; nLoop < g_nSerialMasterNum ; nLoop++)
	{
		UCSerial_Initialize(g_astSerialMasterInfo[nLoop].hSerial);

		result = UKUEMLiteProtocol_SetEncryptionKey(g_astSerialMasterInfo[nLoop].pstEncKeyInfo);
		ERRIFGOTO(result, _EXIT);

		result = UKUEMLiteProtocol_SetHandShakeRequest(0);
		ERRIFGOTO(result, _EXIT);

		result = UKUEMLiteProtocol_Send(g_astSerialMasterInfo[nLoop].hSerial);
		ERRIFGOTO(result, _EXIT);
	}

	for(nLoop = 0 ; nLoop < g_nSerialSlaveNum ; nLoop++)
	{
		UCSerial_Initialize(g_astSerialSlaveInfo[nLoop].hSerial);

		result = UKUEMLiteProtocol_SetEncryptionKey(g_astSerialSlaveInfo[nLoop].pstEncKeyInfo);
		ERRIFGOTO(result, _EXIT);

		nAvailableSize = 0;

		while(nAvailableSize == 0)
		{
			result = UCSerial_Available(g_astSerialSlaveInfo[nLoop].hSerial, &nAvailableSize);
			ERRIFGOTO(result, _EXIT);

			UCTime_Sleep(WAIT_SLEEP_TIME);
		}

		result = UKUEMLiteProtocol_Receive(g_astSerialSlaveInfo[nLoop].hSerial);
		ERRIFGOTO(result, _EXIT);

		result = UKUEMLiteProtocol_GetHeaderFromReceivedData(&nChannelId, &enMessageType, &nParamNum, &pasParam);
		ERRIFGOTO(result, _EXIT);

		if(enMessageType != MESSAGE_TYPE_HANDSHAKE)
		{
			ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
		}

		result = UKUEMLiteProtocol_SetResultMessage(MESSAGE_TYPE_HANDSHAKE, 0, ERR_UEMPROTOCOL_NOERROR, 0);
		ERRIFGOTO(result, _EXIT);

		result = UKUEMLiteProtocol_Send(g_astSerialSlaveInfo[nLoop].hSerial);
		ERRIFGOTO(result, _EXIT);
	}

	for(nLoop = 0; nLoop < g_nSerialMasterNum ; nLoop++)
	{
		nAvailableSize = 0;

		while(nAvailableSize == 0)
		{
			result = UCSerial_Available(g_astSerialMasterInfo[nLoop].hSerial, &nAvailableSize);
			ERRIFGOTO(result, _EXIT);

			UCTime_Sleep(WAIT_SLEEP_TIME);
		}

		result = UKUEMLiteProtocol_Receive(g_astSerialMasterInfo[nLoop].hSerial);
		ERRIFGOTO(result, _EXIT);

		result = UKUEMLiteProtocol_GetHeaderFromReceivedData(&nChannelId, &enMessageType, &nParamNum, &pasParam);
		ERRIFGOTO(result, _EXIT);

		if(enMessageType != MESSAGE_TYPE_RESULT || MESSAGE_TYPE_HANDSHAKE != (EMessageType) pasParam[RESULT_REQUEST_PACKET_INDEX])
		{
			ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
		}
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static uem_result findChannelFromAccessChannelList(SSerialInfo *pstSerialInfo, int nChannelId, OUT SChannel **ppstChannel)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;

	for(nLoop = 0 ; nLoop < pstSerialInfo->nSetChannelAccessNum ; nLoop++)
	{
		if(pstSerialInfo->ppastAccessChannelList[nLoop]->nChannelIndex == nChannelId)
		{
			*ppstChannel = pstSerialInfo->ppastAccessChannelList[nLoop];
			break;
		}
	}

	if(nLoop == pstSerialInfo->nSetChannelAccessNum)
	{
		ERRASSIGNGOTO(result, ERR_UEM_NOT_FOUND, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}



static uem_result pendReadBufferRequest(SSerialInfo *pstSerialInfo, int nChannelId, short *pasParam)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SChannel *pstChannel = NULL;
	int nDataToRead;

	nDataToRead = (int) pasParam[READ_BUFFER_SIZE_TO_READ_INDEX];

	result = findChannelFromAccessChannelList(pstSerialInfo, nChannelId, &pstChannel);
	ERRIFGOTO(result, _EXIT);

	result = UKSerialChannel_PendReadBufferRequest(pstChannel, nDataToRead);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}



static uem_result pendReadQueueRequest(SSerialInfo *pstSerialInfo, int nChannelId, short *pasParam)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SChannel *pstChannel = NULL;
	int nDataToRead;

	nDataToRead = (int) pasParam[READ_QUEUE_SIZE_TO_READ_INDEX];

	result = findChannelFromAccessChannelList(pstSerialInfo, nChannelId, &pstChannel);
	ERRIFGOTO(result, _EXIT);

	result = UKSerialChannel_PendReadQueueRequest(pstChannel, nDataToRead);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result pendAvailableDataRequest(SSerialInfo *pstSerialInfo, int nChannelId, short *pasParam)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SChannel *pstChannel = NULL;

	nChannelId = (int) pasParam[AVAILABLE_DATA_CHANNEL_ID_INDEX];

	result = findChannelFromAccessChannelList(pstSerialInfo, nChannelId, &pstChannel);
	ERRIFGOTO(result, _EXIT);

	result = UKSerialChannel_PendGetAvailableDataRequest(pstChannel);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static uem_result handleReceivedResult(SSerialInfo *pstSerialInfo, int nChannelId, short *pasParam)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SChannel *pstChannel = NULL;
	EMessageType enRequestType = MESSAGE_TYPE_NONE;
	int nBodySize = 0;
	int nReturnValue = 0;
	void *pData = NULL;
	int nDataWritten = 0;
	int nCurrentDataNum = 0;

	enRequestType = (EMessageType) pasParam[RESULT_REQUEST_PACKET_INDEX];

	result = findChannelFromAccessChannelList(pstSerialInfo, nChannelId, &pstChannel);
	ERRIFGOTO(result, _EXIT);

	switch(enRequestType)
	{
	case MESSAGE_TYPE_AVAILABLE_DATA:
		nReturnValue = pasParam[RESULT_RETURN_VALUE_INDEX];
		result = UKSerialChannel_GetNumOfAvailableData(pstChannel, 0, &nCurrentDataNum);
		ERRIFGOTO(result, _EXIT);
		if(nReturnValue > 0 && pstChannel->nBufSize >= nCurrentDataNum + nReturnValue)
		{
			result = UKSerialChannel_PendReadQueueRequest(pstChannel, nReturnValue);
			ERRIFGOTO(result, _EXIT);
		}
		else
		{
			result = UKSerialChannel_ClearRequest(pstChannel);
			ERRIFGOTO(result, _EXIT);
		}
		break;
	case MESSAGE_TYPE_READ_BUFFER:
		result = UKUEMLiteProtocol_GetBodyDataFromReceivedData(&nBodySize, &pData);
		ERRIFGOTO(result, _EXIT);
		result = UKSerialChannel_WriteToBuffer(pstChannel, pData, nBodySize, 0, &nDataWritten);
		ERRIFGOTO(result, _EXIT);
		result = UKSerialChannel_ClearRequest(pstChannel);
		ERRIFGOTO(result, _EXIT);
		break;
	case MESSAGE_TYPE_READ_QUEUE:
		result = UKUEMLiteProtocol_GetBodyDataFromReceivedData(&nBodySize, &pData);
		ERRIFGOTO(result, _EXIT);
		result = UKSerialChannel_WriteToQueue(pstChannel, pData, nBodySize, 0, &nDataWritten);
		ERRIFGOTO(result, _EXIT);
		result = UKSerialChannel_ClearRequest(pstChannel);
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


static uem_result handleReceivedData(SSerialInfo *pstSerialInfo)
{
	uem_result result = ERR_UEM_UNKNOWN;
	EMessageType enMessageType;
	int nChannelId = INVALID_CHANNEL_ID;
	int nParamNum;
	short *pasParam = NULL;

	result = UKUEMLiteProtocol_GetHeaderFromReceivedData(&nChannelId, &enMessageType, &nParamNum, &pasParam);
	ERRIFGOTO(result, _EXIT);

	switch(enMessageType)
	{
	case MESSAGE_TYPE_READ_QUEUE:
		result = pendReadQueueRequest(pstSerialInfo, nChannelId, pasParam);
		ERRIFGOTO(result, _EXIT);
		break;
	case MESSAGE_TYPE_READ_BUFFER:
		result = pendReadBufferRequest(pstSerialInfo, nChannelId, pasParam);
		ERRIFGOTO(result, _EXIT);
		break;
	case MESSAGE_TYPE_AVAILABLE_DATA:
		result = pendAvailableDataRequest(pstSerialInfo, nChannelId, pasParam);
		ERRIFGOTO(result, _EXIT);
		break;
	case MESSAGE_TYPE_RESULT:
		result = handleReceivedResult(pstSerialInfo, nChannelId, pasParam);
		ERRIFGOTO(result, _EXIT);
		break;
	case MESSAGE_TYPE_AVAILABLE_INDEX:
	case MESSAGE_TYPE_NONE:
	case MESSAGE_TYPE_HANDSHAKE:
	default:
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
		break;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static uem_result handleRequests(SSerialInfo *pstSerialInfo)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;
	SChannel *pstChannel = NULL;

	for(nLoop = 0 ; nLoop < pstSerialInfo->nSetChannelAccessNum ; nLoop++)
	{
		pstChannel = pstSerialInfo->ppastAccessChannelList[nLoop];

		result = UKSerialChannel_HandleRequest(pstChannel);
		if(result == ERR_UEM_NOERROR)
		{
			result = UKUEMLiteProtocol_Send(pstSerialInfo->hSerial);
			ERRIFGOTO(result, _EXIT);
		}
		else if(result == ERR_UEM_NOT_REACHED_YET || result == ERR_UEM_SKIP_THIS)
		{
			// skip sending, not ready to send or no send request
		}
		else
		{
			goto _EXIT;/*
			result = UKUEMLiteProtocol_SetResultMessage(MESSAGE_TYPE_NONE, pstChannel->nChannelIndex, ERR_UEMPROTOCOL_ERROR, 0);
			ERRIFGOTO(result, _EXIT);
			result = UKUEMLiteProtocol_Send(pstSerialInfo->hSerial);
			ERRIFGOTO(result, _EXIT);*/
		}
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

// mq request => mq_request => mq result => read_queue request

uem_result UKSerialModule_Run()
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;
	int nAvailableSize = 0;

	for(nLoop = 0 ; nLoop < g_nSerialMasterNum ; nLoop++)
	{
		nAvailableSize = 0;

		result = UCSerial_Available(g_astSerialMasterInfo[nLoop].hSerial, &nAvailableSize);
		ERRIFGOTO(result, _EXIT);

		while(nAvailableSize > 0)
		{
			UCTime_Sleep(SERIAL_COMMUNICATION_DELAY_TIME);
			result = UKUEMLiteProtocol_Receive(g_astSerialMasterInfo[nLoop].hSerial);
			ERRIFGOTO(result, _EXIT);

			result = handleReceivedData(&(g_astSerialMasterInfo[nLoop]));
			ERRIFGOTO(result, _EXIT);

			result = UCSerial_Available(g_astSerialMasterInfo[nLoop].hSerial, &nAvailableSize);
			ERRIFGOTO(result, _EXIT);
		}

		result = handleRequests(&(g_astSerialMasterInfo[nLoop]));
		//UEM_DEBUG_PRINT("result check: %d\n", result);
		ERRIFGOTO(result, _EXIT);
	}

	for(nLoop = 0 ; nLoop < g_nSerialSlaveNum ; nLoop++)
	{
		nAvailableSize = 0;

		result = UCSerial_Available(g_astSerialSlaveInfo[nLoop].hSerial, &nAvailableSize);
		ERRIFGOTO(result, _EXIT);

		while(nAvailableSize > 0)
		{
			UCTime_Sleep(SERIAL_COMMUNICATION_DELAY_TIME);
			result = UKUEMLiteProtocol_Receive(g_astSerialSlaveInfo[nLoop].hSerial);
			ERRIFGOTO(result, _EXIT);

			result = handleReceivedData(&(g_astSerialSlaveInfo[nLoop]));
			ERRIFGOTO(result, _EXIT);

			result = UCSerial_Available(g_astSerialSlaveInfo[nLoop].hSerial, &nAvailableSize);
			ERRIFGOTO(result, _EXIT);
		}

		result = handleRequests(&(g_astSerialSlaveInfo[nLoop]));
		//UEM_DEBUG_PRINT("result check: %d\n", result);
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKSerialModule_SetChannel(SSerialInfo *pstSerialInfo, SChannel *pstChannel)
{
	uem_result result = ERR_UEM_UNKNOWN;

	if(pstSerialInfo->nMaxChannelAccessNum <= pstSerialInfo->nSetChannelAccessNum)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
	}

	pstSerialInfo->ppastAccessChannelList[pstSerialInfo->nSetChannelAccessNum] = pstChannel;
	pstSerialInfo->nSetChannelAccessNum++;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKSerialModule_Finalize()
{
	uem_result result = ERR_UEM_UNKNOWN;

	// do nothing

	result = ERR_UEM_NOERROR;

	return result;
}


