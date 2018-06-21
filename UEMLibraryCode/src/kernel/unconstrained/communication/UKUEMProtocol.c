/*
 * UKUEMProtocol.c
 *
 *  Created on: 2018. 6. 5.
 *      Author: jej
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <UCBasic.h>
#include <UCDynamicSocket.h>

#include <UKUEMProtocol.h>

#define MAX_MESSAGE_PARAMETER (3)

#define HANDSHAKE_TIMEOUT (3)
#define SEND_TIMEOUT (3)
#define RECEIVE_TIMEOUT (3)

#define MAX_HEADER_LENGTH (18)
#define MIN_HEADER_LENGTH (6)

#define PRE_HEADER_LENGTH (2)
#define HEADER_KEY_SIZE (4)
#define MESSAGE_PACKET_SIZE (2)
#define MESSAGE_PARAMETER_SIZE (4)
#define HANDSHAKE_RETRY_COUNT (3)

typedef struct _SUEMProtocolData {
	unsigned int unKey;
	short sMessagePacket;
	int anMessageParam[MAX_MESSAGE_PARAMETER];
	int nParamNum;
	void *pBodyData;
	int nBodyBufLen;
	int nBodyLen;
	char *pFullMessage;
	int nFullMessageLen;
	int nFullMessageBufLen;
} SUEMProtocolData;

typedef struct _SUEMProtocol {
	uem_bool bSent;
	SUEMProtocolData stDataToSend;
	uem_bool bReceived;
	SUEMProtocolData stReceivedData;
	HSocket hSocket;
	int nChannelId;
	int unKey;
} SUEMProtocol;


uem_result UKUEMProtocol_GetMessageParamNumByMessageType(EMessageType enMessageType, int *pnParamNum)
{
	uem_result result = ERR_UEM_NOERROR;
	int nParamNum = 0;
	switch(enMessageType)
	{
	case MESSAGE_TYPE_HANDSHAKE:
	case MESSAGE_TYPE_AVAILABLE_INDEX:
	case MESSAGE_TYPE_AVAILABLE_DATA:
	case MESSAGE_TYPE_RESULT:
		nParamNum = 2;
		break;
	case MESSAGE_TYPE_READ_QUEUE:
	case MESSAGE_TYPE_READ_BUFFER:
		nParamNum = 3;
		break;
	default:
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
		break;
	}

	*pnParamNum = nParamNum;
_EXIT:
	return result;
}

inline static uem_bool intToLittleEndianChar(int nValue, char *pBuffer, int nBufferLen)
{
    if(nBufferLen < sizeof(int))
    {
        return FALSE;
    }
#ifdef WORDS_BIGENDIAN
    pBuffer[0] = nValue & 0xFF;
    pBuffer[1] = (nValue >> 8) & 0xFF;
    pBuffer[2] = (nValue >> 16) & 0xFF;
    pBuffer[3] = (nValue >> 24) & 0xFF;
#else
    int *pnDst;
    pnDst = (int *) pBuffer;
    *pnDst = nValue;
#endif
    return TRUE;
}

inline static uem_bool shortToLittleEndianChar(short sValue, char *pBuffer, int nBufferLen)
{
    if(nBufferLen < sizeof(short))
    {
        return FALSE;
    }
#ifdef WORDS_BIGENDIAN
    pBuffer[0] = sValue & 0xFF;
    pBuffer[1] = (sValue >> 8) & 0xFF;
#else
    short *psDst;
    psDst = (short *) pBuffer;
    *psDst = sValue;
#endif

    return TRUE;
}


inline static uem_bool littleEndianCharToSystemInt(char *pBuffer, int nBufferLen, int *pnValue)
{
    if(nBufferLen < sizeof(int))
    {
        return FALSE;
    }
#ifdef WORDS_BIGENDIAN
    *pnValue =  pBuffer[0];
    *pnValue = *pnValue | ((int) pBuffer[1] << 8);
    *pnValue = *pnValue | ((int) pBuffer[2] << 16);
    *pnValue = *pnValue | ((int) pBuffer[3] << 24);
#else
    int *pnDst;
    pnDst = (int *) pBuffer;
    *pnValue = *pnDst;
#endif
    return TRUE;
}

inline static uem_bool littleEndianCharToSystemShort(char *pBuffer, int nBufferLen, short *psValue)
{
    if(nBufferLen < sizeof(short))
    {
        return FALSE;
    }
#ifdef WORDS_BIGENDIAN
    *psValue =  pBuffer[0];
    *psValue = *psValue | ((short) pBuffer[1] << 8);
#else
    short *psDst;
    psDst = (short *) pBuffer;
    *psValue = *psDst;
#endif
    return TRUE;
}



uem_result UKUEMProtocol_Create(OUT HUEMProtocol *phProtocol)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SUEMProtocol *pstProtocol = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(phProtocol, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	pstProtocol = UC_malloc(sizeof(struct _SUEMProtocol));
	ERRMEMGOTO(pstProtocol, result, _EXIT);

	pstProtocol->hSocket = NULL;
	pstProtocol->bSent = FALSE;
	pstProtocol->bReceived = FALSE;

	pstProtocol->stReceivedData.unKey = 0;
	pstProtocol->stReceivedData.pBodyData = NULL;
	pstProtocol->stReceivedData.nBodyBufLen = 0;
	pstProtocol->stReceivedData.pFullMessage = NULL;
	pstProtocol->stReceivedData.nFullMessageBufLen = 0;

	pstProtocol->stDataToSend.unKey = 0;
	pstProtocol->stDataToSend.pBodyData = NULL;
	pstProtocol->stDataToSend.nBodyBufLen = 0;
	pstProtocol->stDataToSend.pFullMessage = NULL;
	pstProtocol->stDataToSend.nFullMessageBufLen = 0;

	*phProtocol = pstProtocol;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKUEMProtocol_SetSocket(HUEMProtocol hProtocol, HSocket hSocket)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SUEMProtocol *pstProtocol = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(hProtocol, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	pstProtocol = hProtocol;

	pstProtocol->hSocket = hSocket;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result makeFullMessage(SUEMProtocolData *pstProtocolData, short sHeaderLength, int nTotalDataSize)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nIndex = 0;
	int nLeftBufferLen = 0;
	uem_bool bConverted = FALSE;
	int nLoop = 0;

	nIndex = 0;
	nLeftBufferLen = pstProtocolData->nFullMessageBufLen;

	bConverted = shortToLittleEndianChar(sHeaderLength, pstProtocolData->pFullMessage + nIndex, nLeftBufferLen);
	IFVARERRASSIGNGOTO(bConverted, FALSE, result, ERR_UEM_ILLEGAL_DATA, _EXIT);

	nIndex += PRE_HEADER_LENGTH;
	nLeftBufferLen -= PRE_HEADER_LENGTH;

	bConverted = intToLittleEndianChar(pstProtocolData->unKey, pstProtocolData->pFullMessage + nIndex, nLeftBufferLen);
	IFVARERRASSIGNGOTO(bConverted, FALSE, result, ERR_UEM_ILLEGAL_DATA, _EXIT);

	nIndex += HEADER_KEY_SIZE;
	nLeftBufferLen -= HEADER_KEY_SIZE;

	bConverted = shortToLittleEndianChar(pstProtocolData->sMessagePacket, pstProtocolData->pFullMessage + nIndex, nLeftBufferLen);
	IFVARERRASSIGNGOTO(bConverted, FALSE, result, ERR_UEM_ILLEGAL_DATA, _EXIT);

	nIndex += MESSAGE_PACKET_SIZE;
	nLeftBufferLen -= MESSAGE_PACKET_SIZE;

	for(nLoop = 0 ; nLoop < pstProtocolData->nParamNum; nLoop++)
	{
		bConverted = intToLittleEndianChar(pstProtocolData->anMessageParam[nLoop], pstProtocolData->pFullMessage + nIndex, nLeftBufferLen);
		IFVARERRASSIGNGOTO(bConverted, FALSE, result, ERR_UEM_ILLEGAL_DATA, _EXIT);

		nIndex += MESSAGE_PARAMETER_SIZE;
		nLeftBufferLen -= MESSAGE_PARAMETER_SIZE;
	}

	if(pstProtocolData->nBodyLen > 0)
	{
		UC_memcpy(pstProtocolData->pFullMessage + nIndex, pstProtocolData->pBodyData, pstProtocolData->nBodyLen);

		nIndex += pstProtocolData->nBodyLen;
		nLeftBufferLen -= pstProtocolData->nBodyLen;
	}

	pstProtocolData->nFullMessageLen = nTotalDataSize;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result makeSendingData(SUEMProtocolData *pstProtocolData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	short sHeaderLength = 0;
	int nTotalDataSize = 0;

	sHeaderLength = sizeof(pstProtocolData->unKey) + sizeof(pstProtocolData->sMessagePacket) +
					pstProtocolData->nParamNum * sizeof(int);

	if( sHeaderLength < MIN_HEADER_LENGTH || sHeaderLength > MAX_HEADER_LENGTH)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
	}

	nTotalDataSize = PRE_HEADER_LENGTH + sHeaderLength + pstProtocolData->nBodyLen;

	if(pstProtocolData->nFullMessageBufLen < nTotalDataSize)
	{
		SAFEMEMFREE(pstProtocolData->pFullMessage);

		pstProtocolData->pFullMessage = UC_malloc(nTotalDataSize);
		ERRMEMGOTO(pstProtocolData->pFullMessage, result, _EXIT);

		pstProtocolData->nFullMessageBufLen = nTotalDataSize;
	}

	result = makeFullMessage(pstProtocolData, sHeaderLength, nTotalDataSize);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static uem_result clearData(struct _SUEMProtocol *pstProtocol)
{
	int nLoop = 0;

	pstProtocol->bSent = FALSE;
	pstProtocol->bReceived = FALSE;

	for(nLoop = 0 ; nLoop < MAX_MESSAGE_PARAMETER ; nLoop++)
	{
		pstProtocol->stDataToSend.anMessageParam[nLoop] = 0;
	}

	pstProtocol->stDataToSend.pBodyData = NULL;
	pstProtocol->stDataToSend.nBodyLen = 0;

	return ERR_UEM_NOERROR;
}


static uem_result setBasicSendInfo(struct _SUEMProtocol *pstProtocol, EMessageType enMessageType)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nParamNum = 0;

	result = clearData(pstProtocol);
	ERRIFGOTO(result, _EXIT);

	result = UKUEMProtocol_GetMessageParamNumByMessageType(enMessageType, &nParamNum);
	ERRIFGOTO(result, _EXIT);

	pstProtocol->stDataToSend.unKey = pstProtocol->unKey;
	pstProtocol->stDataToSend.sMessagePacket = (short) enMessageType;
	pstProtocol->stDataToSend.nParamNum = nParamNum;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}



uem_result UKUEMProtocol_HandShake(HUEMProtocol hProtocol, unsigned int unDeviceKey, int nChannelId)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SUEMProtocol *pstProtocol = NULL;
	int nRetValue = 0;
	int nRetryCount = 0;
	EProtocolError enErrorCode = ERR_UEMPROTOCOL_ERROR;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(hProtocol, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	pstProtocol = hProtocol;

	// set channel information
	pstProtocol->nChannelId = nChannelId;

	result = setBasicSendInfo(pstProtocol, MESSAGE_TYPE_HANDSHAKE);
	ERRIFGOTO(result, _EXIT);

	pstProtocol->stDataToSend.anMessageParam[0] = unDeviceKey;
	pstProtocol->stDataToSend.anMessageParam[1] = nChannelId;

	result = UKUEMProtocol_Send(hProtocol);
	ERRIFGOTO(result, _EXIT);

	do
	{
		result = UKUEMProtocol_Receive(hProtocol);
		if(result == ERR_UEM_NET_TIMEOUT)
		{
			nRetryCount++;
			if(nRetryCount >= HANDSHAKE_RETRY_COUNT)
			{
				ERRASSIGNGOTO(result, ERR_UEM_CONNECT_ERROR, _EXIT);
			}
		}
	} while(result == ERR_UEM_NET_TIMEOUT);
	ERRIFGOTO(result, _EXIT);

	result = UKUEMProtocol_GetResultFromReceivedData(hProtocol, &enErrorCode, &nRetValue);
	ERRIFGOTO(result, _EXIT);

	if(enErrorCode != ERR_UEMPROTOCOL_NOERROR)
	{
		ERRASSIGNGOTO(result, ERR_UEM_INTERNAL_FAIL, _EXIT);
	}

	pstProtocol->unKey = (unsigned int) nRetValue;

	// This function is called because send-receive order can be changed depending on writer-reader role
	result = clearData(pstProtocol);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKUEMProtocol_SetReadQueueRequest(HUEMProtocol hProtocol, int nIndex, int nSizeToRead)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SUEMProtocol *pstProtocol = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(hProtocol, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	if(nIndex < 0 || nSizeToRead <= 0)
	{
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
#endif
	pstProtocol = hProtocol;

	result = setBasicSendInfo(pstProtocol, MESSAGE_TYPE_READ_QUEUE);
	ERRIFGOTO(result, _EXIT);

	pstProtocol->stDataToSend.anMessageParam[READ_QUEUE_CHANNEL_ID_INDEX] = pstProtocol->nChannelId;
	pstProtocol->stDataToSend.anMessageParam[READ_QUEUE_CHUNK_INDEX_INDEX] = nIndex;
	pstProtocol->stDataToSend.anMessageParam[READ_QUEUE_SIZE_TO_READ_INDEX] = nSizeToRead;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKUEMProtocol_SetReadBufferRequest(HUEMProtocol hProtocol, int nIndex, int nSizeToRead)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SUEMProtocol *pstProtocol = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(hProtocol, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	if(nIndex < 0 || nSizeToRead <= 0)
	{
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
#endif
	pstProtocol = hProtocol;

	result = setBasicSendInfo(pstProtocol, MESSAGE_TYPE_READ_BUFFER);
	ERRIFGOTO(result, _EXIT);

	pstProtocol->stDataToSend.anMessageParam[READ_BUFFER_CHANNEL_ID_INDEX] = pstProtocol->nChannelId;
	pstProtocol->stDataToSend.anMessageParam[READ_BUFFER_CHUNK_INDEX_INDEX] = nIndex;
	pstProtocol->stDataToSend.anMessageParam[READ_BUFFER_SIZE_TO_READ_INDEX] = nSizeToRead;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKUEMProtocol_SetAvailableIndexRequest(HUEMProtocol hProtocol)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SUEMProtocol *pstProtocol = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(hProtocol, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	pstProtocol = hProtocol;

	result = setBasicSendInfo(pstProtocol, MESSAGE_TYPE_AVAILABLE_INDEX);
	ERRIFGOTO(result, _EXIT);

	pstProtocol->stDataToSend.anMessageParam[AVAILABLE_INDEX_CHANNEL_ID_INDEX] = pstProtocol->nChannelId;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKUEMProtocol_SetAvailableDataRequest(HUEMProtocol hProtocol, int nIndex)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SUEMProtocol *pstProtocol = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(hProtocol, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	pstProtocol = hProtocol;

	result = setBasicSendInfo(pstProtocol, MESSAGE_TYPE_AVAILABLE_DATA);
	ERRIFGOTO(result, _EXIT);

	pstProtocol->stDataToSend.anMessageParam[AVAILABLE_DATA_CHANNEL_ID_INDEX] = pstProtocol->nChannelId;
	pstProtocol->stDataToSend.anMessageParam[AVAILABLE_DATA_CHUNK_INDEX_INDEX] = nIndex;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKUEMProtocol_SetResultMessage(HUEMProtocol hProtocol, EProtocolError enErrorCode, int nReturnValue)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SUEMProtocol *pstProtocol = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(hProtocol, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	pstProtocol = hProtocol;

	result = setBasicSendInfo(pstProtocol, MESSAGE_TYPE_RESULT);
	ERRIFGOTO(result, _EXIT);

	pstProtocol->stDataToSend.anMessageParam[RESULT_ERROR_CODE_INDEX] = (int) enErrorCode;
	pstProtocol->stDataToSend.anMessageParam[RESULT_RETURN_VALUE_INDEX] = nReturnValue;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKUEMProtocol_SetResultMessageWithBuffer(HUEMProtocol hProtocol, EProtocolError enErrorCode, int nDataSize, void *pData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SUEMProtocol *pstProtocol = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(hProtocol, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	pstProtocol = hProtocol;

	result = setBasicSendInfo(pstProtocol, MESSAGE_TYPE_RESULT);
	ERRIFGOTO(result, _EXIT);

	pstProtocol->stDataToSend.anMessageParam[RESULT_ERROR_CODE_INDEX] = (int) enErrorCode;
	pstProtocol->stDataToSend.anMessageParam[RESULT_BODY_SIZE_INDEX] = nDataSize;
	pstProtocol->stDataToSend.pBodyData = pData;
	pstProtocol->stDataToSend.nBodyLen = nDataSize;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static uem_result sendData(HSocket hSocket, SUEMProtocolData *pstDataToSend)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nSentSize = 0;
	int nSizeToSend = pstDataToSend->nFullMessageLen;
	int nTotalSizeSent = 0;

	while(nSizeToSend > 0)
	{
		result = UCDynamicSocket_Send(hSocket, SEND_TIMEOUT, pstDataToSend->pFullMessage + nTotalSizeSent, nSizeToSend, &nSentSize);
		ERRIFGOTO(result, _EXIT);

		nSizeToSend -= nSentSize;
		nTotalSizeSent += nSentSize;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKUEMProtocol_Send(HUEMProtocol hProtocol)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SUEMProtocol *pstProtocol = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(hProtocol, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	pstProtocol = hProtocol;

	if(pstProtocol->bSent == TRUE)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ALREADY_DONE, _EXIT);
	}

	result = makeSendingData(&(pstProtocol->stDataToSend));
	ERRIFGOTO(result, _EXIT);

	result = sendData(pstProtocol->hSocket, &(pstProtocol->stDataToSend));
	ERRIFGOTO(result, _EXIT);

	//UEM_DEBUG_PRINT("pstProtocol send: %d\n", pstProtocol->stDataToSend.sMessagePacket);

	pstProtocol->bSent = TRUE;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result receiveHeader(HSocket hSocket, char *pHeaderBuffer, int nBufferLen, OUT int *pnHeaderLength)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nDataReceived = 0;
	int nTotalDataReceived = 0;
	int nDataToRead = PRE_HEADER_LENGTH;
	uem_bool bConverted = FALSE;
	short sHeaderLength = 0;
	int nIndex = 0;

	while(nTotalDataReceived < nDataToRead)
	{
		result = UCDynamicSocket_Receive(hSocket, RECEIVE_TIMEOUT, pHeaderBuffer+nTotalDataReceived, nDataToRead - nTotalDataReceived, &nDataReceived);
		ERRIFGOTO(result, _EXIT);

		nTotalDataReceived += nDataReceived;
	}

	bConverted = littleEndianCharToSystemShort(pHeaderBuffer, PRE_HEADER_LENGTH, &sHeaderLength);
	IFVARERRASSIGNGOTO(bConverted, FALSE, result, ERR_UEM_ILLEGAL_DATA, _EXIT);

	if(sHeaderLength < MIN_HEADER_LENGTH || sHeaderLength > MAX_HEADER_LENGTH)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
	}

	nIndex += PRE_HEADER_LENGTH;

	nDataToRead = (int) sHeaderLength;
	nTotalDataReceived = 0;

	while(nTotalDataReceived < nDataToRead)
	{
		result = UCDynamicSocket_Receive(hSocket, RECEIVE_TIMEOUT, pHeaderBuffer + nIndex + nTotalDataReceived, nDataToRead - nTotalDataReceived, &nDataReceived);
		ERRIFGOTO(result, _EXIT);

		nTotalDataReceived += nDataReceived;
	}

	*pnHeaderLength = (int) sHeaderLength;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result parseHeader(SUEMProtocolData *pstDataReceived, char *pHeader, int nHeaderLength)
{
	uem_result result = ERR_UEM_UNKNOWN;
	uem_bool bConverted = FALSE;
	int nIndex = 0;
	int nLoop = 0;

	bConverted = littleEndianCharToSystemInt(pHeader, nHeaderLength, (int *) &(pstDataReceived->unKey));
	IFVARERRASSIGNGOTO(bConverted, FALSE, result, ERR_UEM_ILLEGAL_DATA, _EXIT);

	nIndex += HEADER_KEY_SIZE;

	bConverted = littleEndianCharToSystemShort(pHeader + nIndex, nHeaderLength - nIndex, &(pstDataReceived->sMessagePacket));
	IFVARERRASSIGNGOTO(bConverted, FALSE, result, ERR_UEM_ILLEGAL_DATA, _EXIT);

	result = UKUEMProtocol_GetMessageParamNumByMessageType((EMessageType) pstDataReceived->sMessagePacket, &(pstDataReceived->nParamNum));
	ERRIFGOTO(result, _EXIT);

	nIndex += MESSAGE_PACKET_SIZE;

	for(nLoop = 0 ; nLoop < pstDataReceived->nParamNum ; nLoop++)
	{
		bConverted = littleEndianCharToSystemInt(pHeader + nIndex, nHeaderLength - nIndex, &(pstDataReceived->anMessageParam[nLoop]));
		IFVARERRASSIGNGOTO(bConverted, FALSE, result, ERR_UEM_ILLEGAL_DATA, _EXIT);

		nIndex += MESSAGE_PARAMETER_SIZE;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result receiveData(HSocket hSocket, SUEMProtocolData *pstDataReceived)
{
	uem_result result = ERR_UEM_UNKNOWN;
	char abyHeader[PRE_HEADER_LENGTH+MAX_HEADER_LENGTH];
	int nHeaderLength = 0;

	result = receiveHeader(hSocket, abyHeader, PRE_HEADER_LENGTH+MAX_HEADER_LENGTH, &nHeaderLength);
	ERRIFGOTO(result, _EXIT);

	result = parseHeader(pstDataReceived, abyHeader+PRE_HEADER_LENGTH, nHeaderLength);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result receiveBody(HSocket hSocket, SUEMProtocolData *pstDataReceived)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nBodySize;
	int nReceivedSize = 0;
	int nTotalReceivedSize = 0;

	nBodySize = pstDataReceived->anMessageParam[RESULT_BODY_SIZE_INDEX];

	if(pstDataReceived->nBodyBufLen < nBodySize)
	{
		SAFEMEMFREE(pstDataReceived->pBodyData);

		pstDataReceived->pBodyData = UC_malloc(nBodySize);
		ERRMEMGOTO(pstDataReceived->pBodyData, result, _EXIT);

		pstDataReceived->nBodyBufLen = nBodySize;
	}

	while(nTotalReceivedSize < nBodySize)
	{
		result = UCDynamicSocket_Receive(hSocket, RECEIVE_TIMEOUT, pstDataReceived->pBodyData + nTotalReceivedSize,
										nBodySize - nTotalReceivedSize, &nReceivedSize);
		ERRIFGOTO(result, _EXIT);

		nTotalReceivedSize += nReceivedSize;
	}

	pstDataReceived->nBodyLen = nBodySize;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKUEMProtocol_Receive(HUEMProtocol hProtocol)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SUEMProtocol *pstProtocol = NULL;
	EMessageType enReceivedMessageType;
	EMessageType enSentMessageType;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(hProtocol, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	pstProtocol = hProtocol;

	if(pstProtocol->bReceived == TRUE)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_CONTROL, _EXIT);
	}

	result = receiveData(pstProtocol->hSocket, &(pstProtocol->stReceivedData));
	ERRIFGOTO(result, _EXIT);

	enSentMessageType = (EMessageType) pstProtocol->stDataToSend.sMessagePacket;
	enReceivedMessageType = (EMessageType) pstProtocol->stReceivedData.sMessagePacket;

	if(enReceivedMessageType == MESSAGE_TYPE_RESULT &&
		(enSentMessageType == MESSAGE_TYPE_READ_QUEUE || enSentMessageType == MESSAGE_TYPE_READ_BUFFER))
	{
		result = receiveBody(pstProtocol->hSocket, &(pstProtocol->stReceivedData));
		ERRIFGOTO(result, _EXIT);
	}

	//UEM_DEBUG_PRINT("pstProtocol receive: %d\n", pstProtocol->stReceivedData.sMessagePacket);

	pstProtocol->bReceived = TRUE;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKUEMProtocol_SetChannelId(HUEMProtocol hProtocol, int nChannelId)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SUEMProtocol *pstProtocol = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(hProtocol, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	if(nChannelId < 0)
	{
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
#endif
	pstProtocol = hProtocol;

	pstProtocol->nChannelId = nChannelId;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKUEMProtocol_GetRequestFromReceivedData(HUEMProtocol hProtocol, OUT EMessageType *penMessageType, OUT int *pnParamNum, OUT int **ppanParam)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SUEMProtocol *pstProtocol = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(hProtocol, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	IFVARERRASSIGNGOTO(penMessageType, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	pstProtocol = hProtocol;

	if(pstProtocol->bReceived == FALSE)
	{
		ERRASSIGNGOTO(result, ERR_UEM_NO_DATA, _EXIT);
	}

	*penMessageType = (EMessageType) pstProtocol->stReceivedData.sMessagePacket;

	if(pnParamNum != NULL)
	{
		*pnParamNum = pstProtocol->stReceivedData.nParamNum;
	}

	if(ppanParam != NULL)
	{
		*ppanParam = pstProtocol->stReceivedData.anMessageParam;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKUEMProtocol_GetResultFromReceivedData(HUEMProtocol hProtocol, OUT EProtocolError *penErrorCode, OUT int *pnReturnValue)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SUEMProtocol *pstProtocol = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(hProtocol, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	IFVARERRASSIGNGOTO(penErrorCode, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	pstProtocol = hProtocol;

	if(pstProtocol->bReceived == FALSE)
	{
		ERRASSIGNGOTO(result, ERR_UEM_NO_DATA, _EXIT);
	}

	*penErrorCode = (EProtocolError) pstProtocol->stReceivedData.anMessageParam[RESULT_ERROR_CODE_INDEX];

	if(pnReturnValue != NULL)
	{
		*pnReturnValue = pstProtocol->stReceivedData.anMessageParam[RESULT_RETURN_VALUE_INDEX];
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKUEMProtocol_GetBodyDataFromReceivedData(HUEMProtocol hProtocol, OUT int *pnBodySize, OUT void **ppBody)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SUEMProtocol *pstProtocol = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(hProtocol, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	pstProtocol = hProtocol;

	if(pstProtocol->bReceived == FALSE)
	{
		ERRASSIGNGOTO(result, ERR_UEM_NO_DATA, _EXIT);
	}

	if(pnBodySize != NULL)
	{
		*pnBodySize = pstProtocol->stReceivedData.nBodyLen;
	}

	if(ppBody != NULL)
	{
		*ppBody = pstProtocol->stReceivedData.pBodyData;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKUEMProtocol_Destroy(IN OUT HUEMProtocol *phProtocol)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SUEMProtocol *pstProtocol = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(phProtocol, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	IFVARERRASSIGNGOTO(*phProtocol, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	pstProtocol = *phProtocol;

	SAFEMEMFREE(pstProtocol->stDataToSend.pFullMessage);
	SAFEMEMFREE(pstProtocol->stReceivedData.pFullMessage);
	SAFEMEMFREE(pstProtocol->stReceivedData.pBodyData);

	SAFEMEMFREE(pstProtocol);

	*phProtocol = NULL;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}
