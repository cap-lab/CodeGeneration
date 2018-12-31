/*
 * UKUEMLiteProtocol.c
 *
 *  Created on: 2018. 10. 25.
 *      Author: jej
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <UCBasic.h>
#include <UCEndian.h>

#include <UKUEMLiteProtocol.h>

#include <uem_lite_protocol_data.h>

#define MAX_PROTOCOL_LENGTH (64)

#define MAX_BODY_LENGTH (MAX_PROTOCOL_LENGTH - MAX_HEADER_LENGTH)

typedef struct _SUEMLiteProtocolData {
	unsigned char byMessagePacket;
	short asMessageParam[MAX_MESSAGE_PARAMETER];
	int nParamNum;
	unsigned char abyBodyData[MAX_BODY_LENGTH];
	unsigned char byBodyLen; // Max 64 byte
} SUEMLiteProtocolData;

typedef struct _SUEMLiteProtocol {
	SUEMLiteProtocolData stReceiveData;
	SUEMLiteProtocolData stSendData;
	unsigned long unKey;
} SUEMLiteProtocol;


static SUEMLiteProtocol s_stProtocol = {
	{
		MESSAGE_TYPE_NONE,
		{0, },
		0,
		{0, },
		0,
	},
	{
		MESSAGE_TYPE_NONE,
		{0, },
		0,
		{0, },
		0,
	},
	0,
};


static uem_result getMessageParamNumByMessageType(EMessageType enMessageType, int *pnParamNum)
{
	uem_result result = ERR_UEM_NOERROR;
	int nParamNum = 0;
	switch(enMessageType)
	{
	case MESSAGE_TYPE_HANDSHAKE:
	case MESSAGE_TYPE_AVAILABLE_DATA:
		nParamNum = 1;
		break;
	case MESSAGE_TYPE_READ_QUEUE:
	case MESSAGE_TYPE_READ_BUFFER:
		nParamNum = 2;
		break;
	case MESSAGE_TYPE_RESULT:
		nParamNum = 4;
		break;
	case MESSAGE_TYPE_AVAILABLE_INDEX:
	default:
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
		break;
	}

	*pnParamNum = nParamNum;
_EXIT:
	return result;
}




static uem_result receiveHeader(HSerial hSerial, char *pHeaderBuffer, int nBufferLen, OUT int *pnHeaderLength)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nDataReceived = 0;
	int nTotalDataReceived = 0;
	int nDataToRead = PRE_HEADER_LENGTH;
	unsigned char byHeaderLength = 0;
	int nIndex = 0;

	while(nTotalDataReceived < nDataToRead)
	{
		result = UCSerial_Receive(hSerial, pHeaderBuffer+nTotalDataReceived, nDataToRead - nTotalDataReceived, &nDataReceived);
		ERRIFGOTO(result, _EXIT);

		nTotalDataReceived += nDataReceived;
	}

	byHeaderLength = pHeaderBuffer[nIndex];

	if(byHeaderLength < MIN_HEADER_LENGTH || byHeaderLength > MAX_HEADER_LENGTH)
	{
		//UEM_DEBUG_PRINT("data corruption error\n");
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
	}

	nIndex += PRE_HEADER_LENGTH;

	nDataToRead = (int) byHeaderLength;
	nTotalDataReceived = 0;

	while(nTotalDataReceived < nDataToRead)
	{
		result = UCSerial_Receive(hSerial, pHeaderBuffer + nIndex + nTotalDataReceived, nDataToRead - nTotalDataReceived, &nDataReceived);
		ERRIFGOTO(result, _EXIT);

		nTotalDataReceived += nDataReceived;
	}

	*pnHeaderLength = (int) byHeaderLength;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static uem_result parseHeader(SUEMLiteProtocolData *pstDataReceived, char *pHeader, int nHeaderLength)
{
	uem_result result = ERR_UEM_UNKNOWN;
	uem_bool bConverted = FALSE;
	int nIndex = 0;
	int nLoop = 0;

	pstDataReceived->byMessagePacket = pHeader[nIndex];

	result = getMessageParamNumByMessageType((EMessageType) pstDataReceived->byMessagePacket, &(pstDataReceived->nParamNum));
	ERRIFGOTO(result, _EXIT);

	nIndex += MESSAGE_PACKET_SIZE;

	for(nLoop = 0 ; nLoop < pstDataReceived->nParamNum ; nLoop++)
	{
		bConverted = UCEndian_LittleEndianCharToSystemShort(pHeader + nIndex, nHeaderLength - nIndex, &(pstDataReceived->asMessageParam[nLoop]));
		IFVARERRASSIGNGOTO(bConverted, FALSE, result, ERR_UEM_ILLEGAL_DATA, _EXIT);

		nIndex += MESSAGE_PARAMETER_SIZE;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result receiveData(HSerial hSerial, SUEMLiteProtocolData *pstDataReceived)
{
	uem_result result = ERR_UEM_UNKNOWN;
	char abyHeader[PRE_HEADER_LENGTH+MAX_HEADER_LENGTH];
	int nHeaderLength = 0;

	result = receiveHeader(hSerial, abyHeader, PRE_HEADER_LENGTH+MAX_HEADER_LENGTH, &nHeaderLength);
	ERRIFGOTO(result, _EXIT);

	result = parseHeader(pstDataReceived, abyHeader+PRE_HEADER_LENGTH, nHeaderLength);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	/*int nTest = 0;
	//UEM_DEBUG_PRINT("Debug on data receive error\n");
	UCSerial_Available(hSerial, &nTest);
	while(nTest > 0)
	{
		UCSerial_Receive(hSerial, abyHeader, MIN(nTest, PRE_HEADER_LENGTH+MAX_HEADER_LENGTH), &nTest);

		UCSerial_Available(hSerial, &nTest);
	}*/

	return result;
}



static uem_result receiveBody(HSerial hSerial, SUEMLiteProtocolData *pstDataReceived)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nBodySize;
	int nReceivedSize = 0;
	int nTotalReceivedSize = 0;

	nBodySize = pstDataReceived->asMessageParam[RESULT_BODY_SIZE_INDEX];

	if(nBodySize > MAX_BODY_LENGTH)
	{
		ERRASSIGNGOTO(result, ERR_UEM_NOT_SUPPORTED, _EXIT);
	}

	while(nTotalReceivedSize < nBodySize)
	{
		result = UCSerial_Receive(hSerial, (char *) (pstDataReceived->abyBodyData + nTotalReceivedSize),
										nBodySize - nTotalReceivedSize, &nReceivedSize);
		ERRIFGOTO(result, _EXIT);

		nTotalReceivedSize += nReceivedSize;
	}

	pstDataReceived->byBodyLen = (unsigned char) nBodySize;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}



uem_result UKUEMLiteProtocol_Receive(HSerial hSerial)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SUEMLiteProtocol *pstProtocol = NULL;
	EMessageType enReceivedMessageType;
	EMessageType enSentMessageType;

	pstProtocol = &s_stProtocol;

	result = receiveData(hSerial, &(pstProtocol->stReceiveData));
	ERRIFGOTO(result, _EXIT);

	enSentMessageType = (EMessageType) pstProtocol->stReceiveData.asMessageParam[RESULT_REQUEST_PACKET_INDEX];
	enReceivedMessageType = (EMessageType) pstProtocol->stReceiveData.byMessagePacket;


	if(enReceivedMessageType == MESSAGE_TYPE_RESULT &&
		(enSentMessageType == MESSAGE_TYPE_READ_QUEUE || enSentMessageType == MESSAGE_TYPE_READ_BUFFER))
	{
		result = receiveBody(hSerial, &(pstProtocol->stReceiveData));
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result clearData(SUEMLiteProtocolData *pstProtocolData)
{
	int nLoop = 0;

	for(nLoop = 0 ; nLoop < MAX_MESSAGE_PARAMETER ; nLoop++)
	{
		pstProtocolData->asMessageParam[nLoop] = 0;
	}

	pstProtocolData->byBodyLen = 0;

	return ERR_UEM_NOERROR;
}



static uem_result setBasicSendInfo(SUEMLiteProtocolData *pstProtocolData, EMessageType enMessageType)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nParamNum = 0;

	result = clearData(pstProtocolData);
	ERRIFGOTO(result, _EXIT);

	result = getMessageParamNumByMessageType(enMessageType, &nParamNum);
	ERRIFGOTO(result, _EXIT);

	pstProtocolData->byMessagePacket = (unsigned char) enMessageType;
	pstProtocolData->nParamNum = nParamNum;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKUEMLiteProtocol_SetReadQueueRequest(int nChannelId, int nSizeToRead)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SUEMLiteProtocol *pstProtocol = NULL;

	pstProtocol = &s_stProtocol;

	result = setBasicSendInfo(&(pstProtocol->stSendData), MESSAGE_TYPE_READ_QUEUE);
	ERRIFGOTO(result, _EXIT);

	pstProtocol->stSendData.asMessageParam[READ_QUEUE_CHANNEL_ID_INDEX] = (short) nChannelId;
	pstProtocol->stSendData.asMessageParam[READ_QUEUE_SIZE_TO_READ_INDEX] = (short) nSizeToRead;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKUEMLiteProtocol_SetReadBufferRequest(int nChannelId, int nSizeToRead)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SUEMLiteProtocol *pstProtocol = NULL;

	pstProtocol = &s_stProtocol;

	result = setBasicSendInfo(&(pstProtocol->stSendData), MESSAGE_TYPE_READ_BUFFER);
	ERRIFGOTO(result, _EXIT);

	pstProtocol->stSendData.asMessageParam[READ_BUFFER_CHANNEL_ID_INDEX] = (short) nChannelId;
	pstProtocol->stSendData.asMessageParam[READ_BUFFER_SIZE_TO_READ_INDEX] = (short) nSizeToRead;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKUEMLiteProtocol_SetAvailableDataRequest(int nChannelId)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SUEMLiteProtocol *pstProtocol = NULL;

	pstProtocol = &s_stProtocol;

	result = setBasicSendInfo(&(pstProtocol->stSendData), MESSAGE_TYPE_AVAILABLE_DATA);
	ERRIFGOTO(result, _EXIT);

	pstProtocol->stSendData.asMessageParam[AVAILABLE_DATA_CHANNEL_ID_INDEX] = (short) nChannelId;
	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKUEMLiteProtocol_SetResultMessage(EMessageType enRequestType, int nChannelId, EProtocolError enErrorCode, int nReturnValue)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SUEMLiteProtocol *pstProtocol = NULL;

	pstProtocol = &s_stProtocol;

	result = setBasicSendInfo(&(pstProtocol->stSendData), MESSAGE_TYPE_RESULT);
	ERRIFGOTO(result, _EXIT);

	pstProtocol->stSendData.asMessageParam[RESULT_CHANNEL_ID_INDEX] = (short) nChannelId;
	pstProtocol->stSendData.asMessageParam[RESULT_REQUEST_PACKET_INDEX] = (short) enRequestType;
	pstProtocol->stSendData.asMessageParam[RESULT_ERROR_CODE_INDEX] = (short) enErrorCode;
	pstProtocol->stSendData.asMessageParam[RESULT_RETURN_VALUE_INDEX] = (short) nReturnValue;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKUEMLiteProtocol_GetResultBufferToSend(OUT unsigned char **ppbyBuffer, OUT int *pnBufferSize)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SUEMLiteProtocol *pstProtocol = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(ppbyBuffer, FALSE, result, ERR_UEM_INVALID_PARAM, _EXIT);
	IFVARERRASSIGNGOTO(pnBufferSize, FALSE, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	pstProtocol = &s_stProtocol;

	*ppbyBuffer = pstProtocol->stSendData.abyBodyData;
	*pnBufferSize = MAX_BODY_LENGTH;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKUEMLiteProtocol_SetResultMessageHeaderUsingBuffer(EMessageType enRequestType, int nChannelId, EProtocolError enErrorCode, int nDataSize)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SUEMLiteProtocol *pstProtocol = NULL;
#ifdef ARGUMENT_CHECK
	if(nDataSize > MAX_BODY_LENGTH)
	{
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
#endif
	pstProtocol = &s_stProtocol;

	result = setBasicSendInfo(&(pstProtocol->stSendData), MESSAGE_TYPE_RESULT);
	ERRIFGOTO(result, _EXIT);

	pstProtocol->stSendData.asMessageParam[RESULT_CHANNEL_ID_INDEX] = (short) nChannelId;
	pstProtocol->stSendData.asMessageParam[RESULT_REQUEST_PACKET_INDEX] = (short) enRequestType;
	pstProtocol->stSendData.asMessageParam[RESULT_ERROR_CODE_INDEX] = (short) enErrorCode;
	pstProtocol->stSendData.asMessageParam[RESULT_BODY_SIZE_INDEX] = (short) nDataSize;

	// Actual buffer is set by getting pointer from UKUEMLiteProtocol_GetResultBufferToSend function
	//UC_memcpy(pstProtocol->stSendData.abyBodyData, pData, nDataSize);
	pstProtocol->stSendData.byBodyLen = (unsigned char) nDataSize;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result sendPreHeader(HSerial hSerial, SUEMLiteProtocolData *pstDataToSend)
{
	uem_result result = ERR_UEM_UNKNOWN;
	unsigned char byHeaderLength = 0;
	int nSentSize = 0;

	byHeaderLength = sizeof(pstDataToSend->byMessagePacket) + pstDataToSend->nParamNum * sizeof(short);

	result = UCSerial_Send(hSerial, (char *) &byHeaderLength, PRE_HEADER_LENGTH, &nSentSize);
	ERRIFGOTO(result, _EXIT);

	if(nSentSize < PRE_HEADER_LENGTH)
	{
		ERRASSIGNGOTO(result, ERR_UEM_NET_SEND_ERROR, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static uem_result sendHeader(HSerial hSerial, SUEMLiteProtocolData *pstDataToSend)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nSentSize = 0;
	int nLoop = 0;
	uem_bool bConverted = FALSE;
	char acParamVal[MESSAGE_PARAMETER_SIZE];
	int nTotalDataSent = 0;

	// send message packet type
	result = UCSerial_Send(hSerial, (char *) &(pstDataToSend->byMessagePacket), MESSAGE_PACKET_SIZE, &nSentSize);
	ERRIFGOTO(result, _EXIT);

	if(nSentSize < MESSAGE_PACKET_SIZE)
	{
		ERRASSIGNGOTO(result, ERR_UEM_NET_SEND_ERROR, _EXIT);
	}

	// send message packet parameter
	for(nLoop = 0 ; nLoop < pstDataToSend->nParamNum; nLoop++)
	{
		bConverted = UCEndian_SystemShortToLittleEndianChar(pstDataToSend->asMessageParam[nLoop], acParamVal, MESSAGE_PARAMETER_SIZE);
		IFVARERRASSIGNGOTO(bConverted, FALSE, result, ERR_UEM_ILLEGAL_DATA, _EXIT);
		nTotalDataSent = 0;

		while(nTotalDataSent < MESSAGE_PARAMETER_SIZE)
		{
			result = UCSerial_Send(hSerial, acParamVal + nTotalDataSent, MESSAGE_PARAMETER_SIZE - nTotalDataSent, &nSentSize);
			ERRIFGOTO(result, _EXIT);
			nTotalDataSent += nSentSize;
		}
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result sendBody(HSerial hSerial, SUEMLiteProtocolData *pstDataToSend)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nTotalDataSent = 0;
	int nSentSize = 0;
	int nDataToSend = 0;

	nDataToSend = (int) pstDataToSend->byBodyLen;

	while(nTotalDataSent < nDataToSend)
	{
		result = UCSerial_Send(hSerial, (char *) (pstDataToSend->abyBodyData + nTotalDataSent), nDataToSend - nTotalDataSent, &nSentSize);
		ERRIFGOTO(result, _EXIT);
		nTotalDataSent += nSentSize;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result sendData(HSerial hSerial, SUEMLiteProtocolData *pstDataToSend)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = sendPreHeader(hSerial, pstDataToSend);
	ERRIFGOTO(result, _EXIT);

	result = sendHeader(hSerial, pstDataToSend);
	ERRIFGOTO(result, _EXIT);

	if(pstDataToSend->byBodyLen > 0)
	{
		result = sendBody(hSerial, pstDataToSend);
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}



uem_result UKUEMLiteProtocol_Send(HSerial hSerial)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SUEMLiteProtocol *pstProtocol = NULL;

	pstProtocol = &s_stProtocol;

	result = sendData(hSerial, &(pstProtocol->stSendData));
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result getChannelIdFromReceivedHeader(EMessageType enMessageType, short asMessageParam[], int nParamNum, OUT int *pnChannelId)
{
	uem_result result = ERR_UEM_UNKNOWN;

	switch(enMessageType)
	{
	case MESSAGE_TYPE_HANDSHAKE:
		*pnChannelId = 0; // not using channel ID
		break;
	case MESSAGE_TYPE_READ_QUEUE:
		*pnChannelId = asMessageParam[READ_QUEUE_CHANNEL_ID_INDEX];
		break;
	case MESSAGE_TYPE_READ_BUFFER:
		*pnChannelId = asMessageParam[READ_BUFFER_CHANNEL_ID_INDEX];
		break;
	case MESSAGE_TYPE_AVAILABLE_DATA:
		*pnChannelId = asMessageParam[AVAILABLE_DATA_CHANNEL_ID_INDEX];
		break;
	case MESSAGE_TYPE_RESULT:
		*pnChannelId = asMessageParam[RESULT_CHANNEL_ID_INDEX];
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


uem_result UKUEMLiteProtocol_GetHeaderFromReceivedData(OUT int *pnChannelId, OUT EMessageType *penMessageType, OUT int *pnParamNum, OUT short **ppasParam)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SUEMLiteProtocol *pstProtocol = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(penMessageType, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	pstProtocol = &s_stProtocol;

	*penMessageType = (EMessageType) pstProtocol->stReceiveData.byMessagePacket;

	if(pnChannelId != NULL)
	{
		result = getChannelIdFromReceivedHeader(*penMessageType, pstProtocol->stReceiveData.asMessageParam, pstProtocol->stReceiveData.nParamNum, pnChannelId);
		ERRIFGOTO(result, _EXIT);
	}

	if(pnParamNum != NULL)
	{
		*pnParamNum = pstProtocol->stReceiveData.nParamNum;
	}

	if(ppasParam != NULL)
	{
		*ppasParam = pstProtocol->stReceiveData.asMessageParam;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKUEMLiteProtocol_GetBodyDataFromReceivedData(OUT int *pnBodySize, OUT void **ppBody)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SUEMLiteProtocol *pstProtocol = NULL;

	pstProtocol = &s_stProtocol;

	if(pnBodySize != NULL)
	{
		*pnBodySize = (int) pstProtocol->stReceiveData.byBodyLen;
	}

	if(ppBody != NULL)
	{
		*ppBody = pstProtocol->stReceiveData.abyBodyData;
	}

	result = ERR_UEM_NOERROR;

	return result;
}



