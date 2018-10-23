/*
 * UCSerial.cpp
 *
 *  Created on: 2018. 10. 23.
 *      Author: jej
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <Arduino.h>

#include <SoftwareSerial.h>

#include <uem_common.h>

#include <UCSerial.h>

typedef struct _SSerialHandle {
	SoftwareSerial *pclsHandle;
} SSerialHandle;


SoftwareSerial BT_Serial(7, 8);

SSerialHandle g_astSerialHandle[]  = {
	{ &BT_Serial },
};

#define DEFAULT_BAUD_RATE (9600)


#ifdef __cplusplus
extern "C"
{
#endif

void UCSerial_Initialize(HSerial hSerial)
{
	SSerialHandle *pstSerialHandle = NULL;

	pstSerialHandle = (SSerialHandle *) hSerial;

	pstSerialHandle->pclsHandle->begin(DEFAULT_BAUD_RATE);
}


uem_result UCSerial_Send(HSerial hSerial, IN char *pData, IN int nDataLen, OUT int *pnSentSize)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSerialHandle *pstSerialHandle = NULL;
	size_t unDataSent = 0;
	size_t unDataToSend = 0;
	size_t unTotalDataSent = 0;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(hSerial, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	IFVARERRASSIGNGOTO(pData, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	IFVARERRASSIGNGOTO(pnSentSize, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	if(nDataLen <= 0)
	{
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
#endif
	pstSerialHandle = (SSerialHandle *) hSerial;

	unDataToSend = (size_t) nDataLen;

	while(unDataToSend > 0)
	{
		unDataSent = pstSerialHandle->pclsHandle->write((const char *)pData + unTotalDataSent, unDataToSend);
		if(unDataSent <= 0)
		{
			break;
		}
		unTotalDataSent += unDataSent;
		unDataToSend -= unDataSent;
	}

	*pnSentSize = unTotalDataSent;

	if(unTotalDataSent == 0)
	{
		ERRASSIGNGOTO(result, ERR_UEM_NET_SEND_ERROR, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UCSerial_Receive(HSerial hSerial, IN OUT char *pBuffer, IN int nBufferLen, OUT int *pnReceivedSize)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSerialHandle *pstSerialHandle = NULL;
	size_t unDataCanBeRead = 0;
	size_t unDataRead = 0;
	size_t unDataToRead = 0;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(hSerial, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	IFVARERRASSIGNGOTO(pBuffer, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	IFVARERRASSIGNGOTO(pnReceivedSize, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	if(nBufferLen <= 0)
	{
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
#endif
	pstSerialHandle = (SSerialHandle *) hSerial;

	unDataCanBeRead = pstSerialHandle->pclsHandle->available();

	if(unDataCanBeRead > 0)
	{
		unDataToRead = MIN(unDataCanBeRead, nBufferLen);
		unDataRead = pstSerialHandle->pclsHandle->readBytes(pBuffer, unDataToRead);
		if(unDataRead <= 0)
		{
			ERRASSIGNGOTO(result, ERR_UEM_NET_RECEIVE_ERROR, _EXIT);
		}
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}



#ifdef __cplusplus
}
#endif


