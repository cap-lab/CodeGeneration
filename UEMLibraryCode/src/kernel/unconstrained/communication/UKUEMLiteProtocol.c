/*
 * UKUEMLiteProtocol.c
 *
 *  Created on: 2018. 10. 5.
 *      Author: chjej202
 */


#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <UCBasic.h>
#include <UCAlloc.h>
#include <UCEndian.h>

#include <UKConnector.h>
#include <UKUEMLiteProtocol.h>

#include <uem_lite_protocol_data.h>

#define MAX_ALLOWED_DATA (32767)


typedef struct _SUEMLiteProtocolData {
	unsigned int unKey;
	unsigned char byMessagePacket;
	short asMessageParam[MAX_MESSAGE_PARAMETER];
	int nParamNum;
	void *pBodyData;
	short sBodyBufLen;
	short sBodyLen;
	char *pFullMessage;
	short sFullMessageLen;
	short sFullMessageBufLen;
	int nChannelId;
} SUEMLiteProtocolData;

typedef struct _SUEMLiteProtocol {
	uem_bool bDataHandled;
	SUEMLiteProtocolData stData;
	HConnector hConnector;
	int unKey;
} SUEMLiteProtocol;


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


uem_result UKUEMLiteProtocol_SetConnector(HUEMLiteProtocol hProtocol, HConnector hConnector)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SUEMLiteProtocol *pstProtocol = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(hProtocol, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	IFVARERRASSIGNGOTO(hConnector, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	pstProtocol = hProtocol;

	pstProtocol->hConnector = hConnector;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKUEMLiteProtocol_Create(OUT HUEMLiteProtocol *phProtocol)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SUEMLiteProtocol *pstProtocol = NULL;
	int nLoop = 0;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(phProtocol, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	pstProtocol = UCAlloc_malloc(sizeof(struct _SUEMLiteProtocol));
	ERRMEMGOTO(pstProtocol, result, _EXIT);

	pstProtocol->hConnector = NULL;
	pstProtocol->bDataHandled = FALSE;
	pstProtocol->unKey = 0;

	for(nLoop = 0; nLoop < MAX_MESSAGE_PARAMETER ; nLoop++)
	{
		pstProtocol->stData.asMessageParam[nLoop] = 0;
	}

	pstProtocol->stData.byMessagePacket = 0;
	pstProtocol->stData.nChannelId = INVALID_CHANNEL_ID;
	pstProtocol->stData.nParamNum = 0;
	pstProtocol->stData.pBodyData = NULL;
	pstProtocol->stData.pFullMessage = NULL;
	pstProtocol->stData.sBodyBufLen = 0;
	pstProtocol->stData.sBodyLen = 0;
	pstProtocol->stData.sFullMessageBufLen = 0;
	pstProtocol->stData.sFullMessageLen = 0;
	pstProtocol->stData.unKey = 0;

	*phProtocol = pstProtocol;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result clearData(struct _SUEMLiteProtocol *pstProtocol)
{
	int nLoop = 0;

	pstProtocol->bDataHandled = FALSE;

	for(nLoop = 0 ; nLoop < MAX_MESSAGE_PARAMETER ; nLoop++)
	{
		pstProtocol->stData.asMessageParam[nLoop] = 0;
	}

	pstProtocol->stData.nChannelId = INVALID_CHANNEL_ID;
	pstProtocol->stData.pBodyData = NULL;
	pstProtocol->stData.sBodyLen = 0;

	return ERR_UEM_NOERROR;
}

static uem_result setBasicSendInfo(struct _SUEMLiteProtocol *pstProtocol, EMessageType enMessageType)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nParamNum = 0;

	result = clearData(pstProtocol);
	ERRIFGOTO(result, _EXIT);

	result = getMessageParamNumByMessageType(enMessageType, &nParamNum);
	ERRIFGOTO(result, _EXIT);

	pstProtocol->stData.unKey = pstProtocol->unKey;
	pstProtocol->stData.byMessagePacket = (unsigned char) enMessageType;
	pstProtocol->stData.nParamNum = nParamNum;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKUEMLiteProtocol_SetHandShakeRequest(HUEMLiteProtocol hProtocol, unsigned int unDeviceKey)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SUEMLiteProtocol *pstProtocol = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(hProtocol, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	pstProtocol = hProtocol;

	result = setBasicSendInfo(pstProtocol, MESSAGE_TYPE_HANDSHAKE);
	ERRIFGOTO(result, _EXIT);

	pstProtocol->stData.asMessageParam[HANDSHAKE_DEVICE_KEY_LSB_INDEX] = (short) (unDeviceKey & 0xffff);
	pstProtocol->stData.asMessageParam[HANDSHAKE_DEVICE_KEY_MSB_INDEX] = (short) ((unDeviceKey >> 16) & 0xffff);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKUEMLiteProtocol_SetReadQueueRequest(HUEMLiteProtocol hProtocol, int nChannelId, int nSizeToRead)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SUEMLiteProtocol *pstProtocol = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(hProtocol, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	if(nSizeToRead <= 0)
	{
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
#endif
	pstProtocol = hProtocol;

	result = setBasicSendInfo(pstProtocol, MESSAGE_TYPE_READ_QUEUE);
	ERRIFGOTO(result, _EXIT);

	pstProtocol->stData.nChannelId = nChannelId;

	pstProtocol->stData.asMessageParam[READ_QUEUE_CHANNEL_ID_INDEX] = (short) nChannelId;
	pstProtocol->stData.asMessageParam[READ_QUEUE_SIZE_TO_READ_INDEX] = (short) nSizeToRead;
	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKUEMLiteProtocol_SetReadBufferRequest(HUEMLiteProtocol hProtocol, int nChannelId, int nSizeToRead)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SUEMLiteProtocol *pstProtocol = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(hProtocol, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	if(nSizeToRead <= 0)
	{
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
#endif
	pstProtocol = hProtocol;

	result = setBasicSendInfo(pstProtocol, MESSAGE_TYPE_READ_BUFFER);
	ERRIFGOTO(result, _EXIT);

	pstProtocol->stData.nChannelId = nChannelId;

	pstProtocol->stData.asMessageParam[READ_BUFFER_CHANNEL_ID_INDEX] = (short) nChannelId;
	pstProtocol->stData.asMessageParam[READ_BUFFER_SIZE_TO_READ_INDEX] = (short) nSizeToRead;
	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKUEMLiteProtocol_SetAvailableDataRequest(HUEMLiteProtocol hProtocol, int nChannelId)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SUEMLiteProtocol *pstProtocol = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(hProtocol, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	pstProtocol = hProtocol;

	result = setBasicSendInfo(pstProtocol, MESSAGE_TYPE_AVAILABLE_DATA);
	ERRIFGOTO(result, _EXIT);

	pstProtocol->stData.nChannelId = nChannelId;

	pstProtocol->stData.asMessageParam[AVAILABLE_DATA_CHANNEL_ID_INDEX] = (short) nChannelId;
	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKUEMLiteProtocol_SetResultMessage(HUEMLiteProtocol hProtocol, EMessageType enRequestType, int nChannelId, EProtocolError enErrorCode, int nReturnValue)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SUEMLiteProtocol *pstProtocol = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(hProtocol, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	pstProtocol = hProtocol;

	result = setBasicSendInfo(pstProtocol, MESSAGE_TYPE_RESULT);
	ERRIFGOTO(result, _EXIT);

	pstProtocol->stData.nChannelId = nChannelId;

	pstProtocol->stData.asMessageParam[RESULT_CHANNEL_ID_INDEX] = (short) nChannelId;
	pstProtocol->stData.asMessageParam[RESULT_REQUEST_PACKET_INDEX] = (short) enRequestType;
	pstProtocol->stData.asMessageParam[RESULT_ERROR_CODE_INDEX] = (short) enErrorCode;
	pstProtocol->stData.asMessageParam[RESULT_RETURN_VALUE_INDEX] = (short) nReturnValue;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKUEMLiteProtocol_SetResultMessageHeaderUsingBuffer(HUEMLiteProtocol hProtocol, EMessageType enRequestType, int nChannelId, EProtocolError enErrorCode, int nDataSize, void *pData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SUEMLiteProtocol *pstProtocol = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(hProtocol, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	if(nDataSize >= MAX_ALLOWED_DATA)
	{
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
#endif
	pstProtocol = hProtocol;

	result = setBasicSendInfo(pstProtocol, MESSAGE_TYPE_RESULT);
	ERRIFGOTO(result, _EXIT);

	pstProtocol->stData.nChannelId = nChannelId;

	pstProtocol->stData.asMessageParam[RESULT_CHANNEL_ID_INDEX] = (short) nChannelId;
	pstProtocol->stData.asMessageParam[RESULT_REQUEST_PACKET_INDEX] = (short) enRequestType;
	pstProtocol->stData.asMessageParam[RESULT_ERROR_CODE_INDEX] = (short) enErrorCode;
	pstProtocol->stData.asMessageParam[RESULT_BODY_SIZE_INDEX] = (short) nDataSize;
	pstProtocol->stData.pBodyData = pData;
	pstProtocol->stData.sBodyLen = (short) nDataSize;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result makeFullMessage(SUEMLiteProtocolData *pstProtocolData, unsigned char byHeaderLength, short sTotalDataSize)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nIndex = 0;
	short sLeftBufferLen = 0;
	uem_bool bConverted = FALSE;
	int nLoop = 0;

	nIndex = 0;
	sLeftBufferLen = pstProtocolData->sFullMessageBufLen;

	pstProtocolData->pFullMessage[nIndex] = byHeaderLength;

	nIndex += PRE_HEADER_LENGTH;
	sLeftBufferLen -= PRE_HEADER_LENGTH;

	pstProtocolData->pFullMessage[nIndex] = pstProtocolData->byMessagePacket;

	nIndex += MESSAGE_PACKET_SIZE;
	sLeftBufferLen -= MESSAGE_PACKET_SIZE;

	for(nLoop = 0 ; nLoop < pstProtocolData->nParamNum; nLoop++)
	{
		bConverted = UCEndian_SystemShortToLittleEndianChar(pstProtocolData->asMessageParam[nLoop], pstProtocolData->pFullMessage + nIndex, sLeftBufferLen);
		IFVARERRASSIGNGOTO(bConverted, FALSE, result, ERR_UEM_ILLEGAL_DATA, _EXIT);

		nIndex += MESSAGE_PARAMETER_SIZE;
		sLeftBufferLen -= MESSAGE_PARAMETER_SIZE;
	}

	if(pstProtocolData->sBodyLen > 0)
	{
		UC_memcpy(pstProtocolData->pFullMessage + nIndex, pstProtocolData->pBodyData, pstProtocolData->sBodyLen);

		nIndex += pstProtocolData->sBodyLen;
		sLeftBufferLen -= pstProtocolData->sBodyLen;
	}

	pstProtocolData->sFullMessageLen = sTotalDataSize;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result makeSendingData(SUEMLiteProtocolData *pstProtocolData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	unsigned char byHeaderLength = 0;
	short sTotalDataSize = 0;

	byHeaderLength = sizeof(pstProtocolData->byMessagePacket) + pstProtocolData->nParamNum * sizeof(short);

	if( byHeaderLength < MIN_HEADER_LENGTH || byHeaderLength > MAX_HEADER_LENGTH)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
	}

	sTotalDataSize = PRE_HEADER_LENGTH + byHeaderLength + pstProtocolData->sBodyLen;

	if(pstProtocolData->sFullMessageBufLen < sTotalDataSize)
	{
		SAFEMEMFREE(pstProtocolData->pFullMessage);

		pstProtocolData->pFullMessage = UCAlloc_malloc(sTotalDataSize);
		ERRMEMGOTO(pstProtocolData->pFullMessage, result, _EXIT);

		pstProtocolData->sFullMessageBufLen = sTotalDataSize;
	}

	result = makeFullMessage(pstProtocolData, byHeaderLength, sTotalDataSize);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


// TODO: this is for only socket (for serial communication, we need to find a way to access)
static uem_result sendData(HConnector hConnector, SUEMLiteProtocolData *pstDataToSend)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nSentSize = 0;
	int nSizeToSend = pstDataToSend->sFullMessageLen;
	int nTotalSizeSent = 0;

	while(nSizeToSend > 0)
	{
		result = UKConnector_Send(hConnector, SEND_TIMEOUT, pstDataToSend->pFullMessage + nTotalSizeSent, nSizeToSend, &nSentSize);
		ERRIFGOTO(result, _EXIT);

		nSizeToSend -= nSentSize;
		nTotalSizeSent += nSentSize;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKUEMLiteProtocol_Send(HUEMLiteProtocol hProtocol)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SUEMLiteProtocol *pstProtocol = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(hProtocol, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	pstProtocol = hProtocol;

	if(pstProtocol->bDataHandled == TRUE)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ALREADY_DONE, _EXIT);
	}

	result = makeSendingData(&(pstProtocol->stData));
	ERRIFGOTO(result, _EXIT);

	result = sendData(pstProtocol->hConnector, &(pstProtocol->stData));
	ERRIFGOTO(result, _EXIT);

	//UEM_DEBUG_PRINT("pstProtocol send: %d\n", pstProtocol->stDataToSend.sMessagePacket);

	pstProtocol->bDataHandled = TRUE;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result receiveHeader(HConnector hConnector, char *pHeaderBuffer, int nBufferLen, OUT int *pnHeaderLength)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nDataReceived = 0;
	int nTotalDataReceived = 0;
	int nDataToRead = PRE_HEADER_LENGTH;
	unsigned char byHeaderLength = 0;
	int nIndex = 0;

	while(nTotalDataReceived < nDataToRead)
	{
		result = UKConnector_Receive(hConnector, RECEIVE_TIMEOUT, pHeaderBuffer+nTotalDataReceived, nDataToRead - nTotalDataReceived, &nDataReceived);
		ERRIFGOTO(result, _EXIT);

		nTotalDataReceived += nDataReceived;
	}

	byHeaderLength = pHeaderBuffer[nIndex];

	if(byHeaderLength < MIN_HEADER_LENGTH || byHeaderLength > MAX_HEADER_LENGTH)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
	}

	nIndex += PRE_HEADER_LENGTH;

	nDataToRead = (int) byHeaderLength;
	nTotalDataReceived = 0;

	while(nTotalDataReceived < nDataToRead)
	{
		result = UKConnector_Receive(hConnector, RECEIVE_TIMEOUT, pHeaderBuffer + nIndex + nTotalDataReceived, nDataToRead - nTotalDataReceived, &nDataReceived);
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


static uem_result receiveData(HConnector hConnector, SUEMLiteProtocolData *pstDataReceived)
{
	uem_result result = ERR_UEM_UNKNOWN;
	char abyHeader[PRE_HEADER_LENGTH+MAX_HEADER_LENGTH];
	int nHeaderLength = 0;

	result = receiveHeader(hConnector, abyHeader, PRE_HEADER_LENGTH+MAX_HEADER_LENGTH, &nHeaderLength);
	ERRIFGOTO(result, _EXIT);

	result = parseHeader(pstDataReceived, abyHeader+PRE_HEADER_LENGTH, nHeaderLength);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result receiveBody(HConnector hConnector, SUEMLiteProtocolData *pstDataReceived)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nBodySize;
	int nReceivedSize = 0;
	int nTotalReceivedSize = 0;
	int nBodyBufLen = 0;

	nBodySize = pstDataReceived->asMessageParam[RESULT_BODY_SIZE_INDEX];
	nBodyBufLen = (int) pstDataReceived->sBodyBufLen;

	if(nBodyBufLen < nBodySize)
	{
		SAFEMEMFREE(pstDataReceived->pBodyData);

		pstDataReceived->pBodyData = UCAlloc_malloc(nBodySize);
		ERRMEMGOTO(pstDataReceived->pBodyData, result, _EXIT);

		pstDataReceived->sBodyBufLen = (short) nBodySize;
	}

	while(nTotalReceivedSize < nBodySize)
	{
		result = UKConnector_Receive(hConnector, RECEIVE_TIMEOUT, pstDataReceived->pBodyData + nTotalReceivedSize,
										nBodySize - nTotalReceivedSize, &nReceivedSize);
		ERRIFGOTO(result, _EXIT);

		nTotalReceivedSize += nReceivedSize;
	}

	pstDataReceived->sBodyLen = (short) nBodySize;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}



uem_result UKUEMLiteProtocol_HandShake(HUEMLiteProtocol hProtocol, HUEMLiteProtocol hReceiveProtocol, unsigned int unDeviceKey)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nRetryCount = 0;
	EMessageType enMessageType;
	short *pasParam = NULL;
	int nParamNum = 0;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(hProtocol, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	IFVARERRASSIGNGOTO(hReceiveProtocol, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif

	result = UKUEMLiteProtocol_SetHandShakeRequest(hProtocol, unDeviceKey);
	ERRIFGOTO(result, _EXIT);

	result = UKUEMLiteProtocol_Send(hProtocol);
	ERRIFGOTO(result, _EXIT);

	do {
		result = UKUEMLiteProtocol_Receive(hReceiveProtocol);
		if(result == ERR_UEM_NET_TIMEOUT)
		{
			nRetryCount++;
			if(nRetryCount >= HANDSHAKE_RETRY_COUNT)
			{
				ERRASSIGNGOTO(result, ERR_UEM_CONNECT_ERROR, _EXIT);
			}
		}

	}while(result == ERR_UEM_NET_TIMEOUT);
	ERRIFGOTO(result, _EXIT);

	result = UKUEMLiteProtocol_GetHeaderFromReceivedData(hReceiveProtocol, NULL, &enMessageType, &nParamNum, &pasParam);
	ERRIFGOTO(result, _EXIT);

	if(enMessageType != MESSAGE_TYPE_RESULT || MESSAGE_TYPE_HANDSHAKE != (EMessageType) pasParam[RESULT_REQUEST_PACKET_INDEX])
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKUEMLiteProtocol_Receive(HUEMLiteProtocol hProtocol)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SUEMLiteProtocol *pstProtocol = NULL;
	EMessageType enReceivedMessageType;
	EMessageType enSentMessageType;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(hProtocol, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	pstProtocol = hProtocol;

	if(pstProtocol->bDataHandled == TRUE)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_CONTROL, _EXIT);
	}

	result = receiveData(pstProtocol->hConnector, &(pstProtocol->stData));
	ERRIFGOTO(result, _EXIT);

	enSentMessageType = (EMessageType) pstProtocol->stData.asMessageParam[RESULT_REQUEST_PACKET_INDEX];
	enReceivedMessageType = (EMessageType) pstProtocol->stData.byMessagePacket;

	if(enReceivedMessageType == MESSAGE_TYPE_RESULT &&
		(enSentMessageType == MESSAGE_TYPE_READ_QUEUE || enSentMessageType == MESSAGE_TYPE_READ_BUFFER))
	{
		result = receiveBody(pstProtocol->hConnector, &(pstProtocol->stData));
		ERRIFGOTO(result, _EXIT);
	}

	//UEM_DEBUG_PRINT("pstProtocol receive: %d\n", pstProtocol->stReceivedData.sMessagePacket);

	pstProtocol->bDataHandled = TRUE;
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


uem_result UKUEMLiteProtocol_GetHeaderFromReceivedData(HUEMLiteProtocol hProtocol, OUT int *pnChannelId, OUT EMessageType *penMessageType, OUT int *pnParamNum, OUT short **ppasParam)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SUEMLiteProtocol *pstProtocol = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(hProtocol, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	IFVARERRASSIGNGOTO(penMessageType, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	pstProtocol = hProtocol;

	if(pstProtocol->bDataHandled == FALSE)
	{
		ERRASSIGNGOTO(result, ERR_UEM_NO_DATA, _EXIT);
	}

	*penMessageType = (EMessageType) pstProtocol->stData.byMessagePacket;

	if(pnChannelId != NULL)
	{
		result = getChannelIdFromReceivedHeader(*penMessageType, pstProtocol->stData.asMessageParam, pstProtocol->stData.nParamNum, pnChannelId);
		ERRIFGOTO(result, _EXIT);
	}

	if(pnParamNum != NULL)
	{
		*pnParamNum = pstProtocol->stData.nParamNum;
	}

	if(ppasParam != NULL)
	{
		*ppasParam = pstProtocol->stData.asMessageParam;
	}

	pstProtocol->bDataHandled = FALSE;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKUEMLiteProtocol_GetBodyDataFromReceivedData(HUEMLiteProtocol hProtocol, OUT int *pnBodySize, OUT void **ppBody)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SUEMLiteProtocol *pstProtocol = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(hProtocol, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	pstProtocol = hProtocol;

	if(pnBodySize != NULL)
	{
		*pnBodySize = pstProtocol->stData.sBodyLen;
	}

	if(ppBody != NULL)
	{
		*ppBody = pstProtocol->stData.pBodyData;
	}

	pstProtocol->bDataHandled = FALSE;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKUEMLiteProtocol_Destroy(IN OUT HUEMLiteProtocol *phProtocol)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SUEMLiteProtocol *pstProtocol = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(phProtocol, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	IFVARERRASSIGNGOTO(*phProtocol, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	pstProtocol = *phProtocol;

	SAFEMEMFREE(pstProtocol->stData.pBodyData);
	SAFEMEMFREE(pstProtocol->stData.pFullMessage);

	SAFEMEMFREE(pstProtocol);

	*phProtocol = NULL;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}
