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
#include <UCAlloc.h>
#include <UCEndian.h>

#include <UKVirtualCommunication.h>
#include <UKVirtualEncryption.h>
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
	HVirtualSocket hSocket;
	SVirtualCommunicationAPI *pstAPI;
	int nChannelId;
	int unKey;
	HVirtualKey hKey;
	SVirtualEncryptionAPI *pstEncAPI;
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


uem_result UKUEMProtocol_Create(OUT HUEMProtocol *phProtocol)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SUEMProtocol *pstProtocol = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(phProtocol, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	pstProtocol = UCAlloc_malloc(sizeof(struct _SUEMProtocol));
	ERRMEMGOTO(pstProtocol, result, _EXIT);

	pstProtocol->hSocket = NULL;
	pstProtocol->pstAPI = NULL;
	pstProtocol->bSent = FALSE;
	pstProtocol->bReceived = FALSE;
	pstProtocol->nChannelId = INVALID_CHANNEL_ID;
	pstProtocol->unKey = 0; // not used now
	pstProtocol->pstEncAPI = NULL;
	pstProtocol->hKey = NULL;

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

uem_result UKUEMProtocol_SetSocket(HUEMProtocol hProtocol, HVirtualSocket hSocket, SVirtualCommunicationAPI *pstAPI)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SUEMProtocol *pstProtocol = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(hProtocol, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	IFVARERRASSIGNGOTO(pstAPI, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	pstProtocol = hProtocol;

	pstProtocol->hSocket = hSocket;
	pstProtocol->pstAPI = pstAPI;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKUEMProtocol_SetEncryptionKey(HUEMProtocol hProtocol, SEncryptionKeyInfo *pstEncKeyInfo)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SUEMProtocol *pstProtocol = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(hProtocol, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	pstProtocol = hProtocol;

	if(pstEncKeyInfo != NULL)
	{
		pstProtocol->pstEncAPI = pstEncKeyInfo->pstEncAPI;

		if(pstProtocol->pstEncAPI != NULL)
		{
			result = pstProtocol->pstEncAPI->fnInitialize(&(pstProtocol->hKey), pstEncKeyInfo);
			ERRIFGOTO(result, _EXIT);
		}
	}
	
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

	bConverted = UCEndian_SystemShortToLittleEndianChar(sHeaderLength, pstProtocolData->pFullMessage + nIndex, nLeftBufferLen);
	IFVARERRASSIGNGOTO(bConverted, FALSE, result, ERR_UEM_ILLEGAL_DATA, _EXIT);

	nIndex += PRE_HEADER_LENGTH;
	nLeftBufferLen -= PRE_HEADER_LENGTH;

	bConverted = UCEndian_SystemIntToLittleEndianChar(pstProtocolData->unKey, pstProtocolData->pFullMessage + nIndex, nLeftBufferLen);
	IFVARERRASSIGNGOTO(bConverted, FALSE, result, ERR_UEM_ILLEGAL_DATA, _EXIT);

	nIndex += HEADER_KEY_SIZE;
	nLeftBufferLen -= HEADER_KEY_SIZE;

	bConverted = UCEndian_SystemShortToLittleEndianChar(pstProtocolData->sMessagePacket, pstProtocolData->pFullMessage + nIndex, nLeftBufferLen);
	IFVARERRASSIGNGOTO(bConverted, FALSE, result, ERR_UEM_ILLEGAL_DATA, _EXIT);

	nIndex += MESSAGE_PACKET_SIZE;
	nLeftBufferLen -= MESSAGE_PACKET_SIZE;

	for(nLoop = 0 ; nLoop < pstProtocolData->nParamNum; nLoop++)
	{
		bConverted = UCEndian_SystemIntToLittleEndianChar(pstProtocolData->anMessageParam[nLoop], pstProtocolData->pFullMessage + nIndex, nLeftBufferLen);
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

		pstProtocolData->pFullMessage = UCAlloc_malloc(nTotalDataSize);
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

static uem_result sendData(HVirtualSocket hSocket, SVirtualCommunicationAPI *pstAPI, SUEMProtocolData *pstDataToSend)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nSentSize = 0;
	int nSizeToSend = pstDataToSend->nFullMessageLen;
	int nTotalSizeSent = 0;

	while(nSizeToSend > 0)
	{
		result = pstAPI->fnSend(hSocket, SEND_TIMEOUT, pstDataToSend->pFullMessage + nTotalSizeSent, nSizeToSend, &nSentSize);
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
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(pstProtocol->pstAPI, NULL, result, ERR_UEM_INVALID_SOCKET, _EXIT);
	IFVARERRASSIGNGOTO(pstProtocol->hSocket, NULL, result, ERR_UEM_INVALID_SOCKET, _EXIT);
#endif
	if(pstProtocol->bSent == TRUE)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ALREADY_DONE, _EXIT);
	}

	if(pstProtocol->pstEncAPI != NULL)
	{
		result = pstProtocol->pstEncAPI->fnEncode(pstProtocol->hKey, pstProtocol->stDataToSend.pBodyData, pstProtocol->stDataToSend.nBodyLen);
		ERRIFGOTO(result, _EXIT);
	}
		
	result = makeSendingData(&(pstProtocol->stDataToSend));
	ERRIFGOTO(result, _EXIT);
	
	result = sendData(pstProtocol->hSocket, pstProtocol->pstAPI, &(pstProtocol->stDataToSend));
	ERRIFGOTO(result, _EXIT);

	//UEM_DEBUG_PRINT("pstProtocol send: %d\n", pstProtocol->stDataToSend.sMessagePacket);

	pstProtocol->bSent = TRUE;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result receiveHeader(HVirtualSocket hSocket, SVirtualCommunicationAPI *pstAPI, char *pHeaderBuffer, int nBufferLen, OUT int *pnHeaderLength)
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
		result = pstAPI->fnReceive(hSocket, RECEIVE_TIMEOUT, pHeaderBuffer+nTotalDataReceived, nDataToRead - nTotalDataReceived, &nDataReceived);
		ERRIFGOTO(result, _EXIT);

		nTotalDataReceived += nDataReceived;
	}

	bConverted = UCEndian_LittleEndianCharToSystemShort(pHeaderBuffer, PRE_HEADER_LENGTH, &sHeaderLength);
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
		result = pstAPI->fnReceive(hSocket, RECEIVE_TIMEOUT, pHeaderBuffer + nIndex + nTotalDataReceived, nDataToRead - nTotalDataReceived, &nDataReceived);
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

	bConverted = UCEndian_LittleEndianCharToSystemInt(pHeader, nHeaderLength, (int *) &(pstDataReceived->unKey));
	IFVARERRASSIGNGOTO(bConverted, FALSE, result, ERR_UEM_ILLEGAL_DATA, _EXIT);

	nIndex += HEADER_KEY_SIZE;

	bConverted = UCEndian_LittleEndianCharToSystemShort(pHeader + nIndex, nHeaderLength - nIndex, &(pstDataReceived->sMessagePacket));
	IFVARERRASSIGNGOTO(bConverted, FALSE, result, ERR_UEM_ILLEGAL_DATA, _EXIT);

	result = UKUEMProtocol_GetMessageParamNumByMessageType((EMessageType) pstDataReceived->sMessagePacket, &(pstDataReceived->nParamNum));
	ERRIFGOTO(result, _EXIT);

	nIndex += MESSAGE_PACKET_SIZE;

	for(nLoop = 0 ; nLoop < pstDataReceived->nParamNum ; nLoop++)
	{
		bConverted = UCEndian_LittleEndianCharToSystemInt(pHeader + nIndex, nHeaderLength - nIndex, &(pstDataReceived->anMessageParam[nLoop]));
		IFVARERRASSIGNGOTO(bConverted, FALSE, result, ERR_UEM_ILLEGAL_DATA, _EXIT);

		nIndex += MESSAGE_PARAMETER_SIZE;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result receiveData(HVirtualSocket hSocket, SVirtualCommunicationAPI *pstAPI, SUEMProtocolData *pstDataReceived)
{
	uem_result result = ERR_UEM_UNKNOWN;
	char abyHeader[PRE_HEADER_LENGTH+MAX_HEADER_LENGTH];
	int nHeaderLength = 0;

	result = receiveHeader(hSocket, pstAPI, abyHeader, PRE_HEADER_LENGTH+MAX_HEADER_LENGTH, &nHeaderLength);
	ERRIFGOTO(result, _EXIT);

	result = parseHeader(pstDataReceived, abyHeader+PRE_HEADER_LENGTH, nHeaderLength);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result receiveBody(HVirtualSocket hSocket, SVirtualCommunicationAPI *pstAPI, SUEMProtocolData *pstDataReceived)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nBodySize;
	int nReceivedSize = 0;
	int nTotalReceivedSize = 0;

	nBodySize = pstDataReceived->anMessageParam[RESULT_BODY_SIZE_INDEX];

	if(pstDataReceived->nBodyBufLen < nBodySize)
	{
		SAFEMEMFREE(pstDataReceived->pBodyData);

		pstDataReceived->pBodyData = UCAlloc_malloc(nBodySize);
		ERRMEMGOTO(pstDataReceived->pBodyData, result, _EXIT);

		pstDataReceived->nBodyBufLen = nBodySize;
	}

	while(nTotalReceivedSize < nBodySize)
	{
		result = pstAPI->fnReceive(hSocket, RECEIVE_TIMEOUT, (char *)pstDataReceived->pBodyData + nTotalReceivedSize,
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
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(pstProtocol->hSocket, NULL, result, ERR_UEM_INVALID_SOCKET, _EXIT);
	IFVARERRASSIGNGOTO(pstProtocol->pstAPI, NULL, result, ERR_UEM_INVALID_SOCKET, _EXIT);
#endif
	if(pstProtocol->bReceived == TRUE)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_CONTROL, _EXIT);
	}

	result = receiveData(pstProtocol->hSocket, pstProtocol->pstAPI, &(pstProtocol->stReceivedData));
	ERRIFGOTO(result, _EXIT);

	enSentMessageType = (EMessageType) pstProtocol->stDataToSend.sMessagePacket;
	enReceivedMessageType = (EMessageType) pstProtocol->stReceivedData.sMessagePacket;

	if(enReceivedMessageType == MESSAGE_TYPE_RESULT &&
		(enSentMessageType == MESSAGE_TYPE_READ_QUEUE || enSentMessageType == MESSAGE_TYPE_READ_BUFFER))
	{
		result = receiveBody(pstProtocol->hSocket, pstProtocol->pstAPI, &(pstProtocol->stReceivedData));
		ERRIFGOTO(result, _EXIT);

		if(pstProtocol->pstEncAPI != NULL)
		{
			result = pstProtocol->pstEncAPI->fnDecode(pstProtocol->hKey, pstProtocol->stReceivedData.pBodyData, pstProtocol->stReceivedData.nBodyLen);
			ERRIFGOTO(result, _EXIT);
		}
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
	uem_result result = ERR_UEM_NOERROR;
	struct _SUEMProtocol *pstProtocol = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(phProtocol, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	IFVARERRASSIGNGOTO(*phProtocol, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	pstProtocol = *phProtocol;

	SAFEMEMFREE(pstProtocol->stDataToSend.pFullMessage);
	SAFEMEMFREE(pstProtocol->stReceivedData.pFullMessage);
	SAFEMEMFREE(pstProtocol->stReceivedData.pBodyData);

	if(pstProtocol->pstEncAPI != NULL)
	{
		result = pstProtocol->pstEncAPI->fnFinalize(&(pstProtocol->hKey));
	}

	SAFEMEMFREE(pstProtocol);

	*phProtocol = NULL;

_EXIT:
	return result;
}
