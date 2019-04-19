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
#ifndef ARDUINO_OpenCR
	#include <SoftwareSerial.h>
#endif

#include <uem_common.h>

#include <UCSerial.h>
#include <UCSerial_data.hpp>


#ifdef __cplusplus
extern "C"
{
#endif

void UCSerial_Initialize(HSerial hSerial)
{
	SSerialHandle *pstSerialHandle = (SSerialHandle *) NULL;

	pstSerialHandle = (SSerialHandle *) hSerial;

	pstSerialHandle->pclsHandle->begin(DATA_SERIAL_DEFAULT_BAUD_RATE);
}


uem_result UCSerial_Send(HSerial hSerial, IN char *pData, IN int nDataLen, OUT int *pnSentSize)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSerialHandle *pstSerialHandle = (SSerialHandle *) NULL;
	size_t unDataSent = 0;
	size_t unDataToSend = 0;
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

	unDataSent = pstSerialHandle->pclsHandle->write((const char *)pData, unDataToSend);

	*pnSentSize = unDataSent;

	if(unDataSent == 0)
	{
		ERRASSIGNGOTO(result, ERR_UEM_NET_SEND_ERROR, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UCSerial_Available(HSerial hSerial, OUT int *pnAvailableSize)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSerialHandle *pstSerialHandle = (SSerialHandle *) NULL;
	size_t unDataCanBeRead = 0;

	pstSerialHandle = (SSerialHandle *) hSerial;

	unDataCanBeRead = pstSerialHandle->pclsHandle->available();

	*pnAvailableSize = unDataCanBeRead;

	result = ERR_UEM_NOERROR;

	return result;
}



uem_result UCSerial_Receive(HSerial hSerial, IN OUT char *pBuffer, IN int nBufferLen, OUT int *pnReceivedSize)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSerialHandle *pstSerialHandle = (SSerialHandle *) NULL;
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
		unDataToRead =  MIN(unDataCanBeRead, (size_t) nBufferLen);
		unDataRead = pstSerialHandle->pclsHandle->readBytes(pBuffer, unDataToRead);

		/*{
			int nLoop = 0;

			for(nLoop = 0 ; nLoop < unDataRead ; nLoop++)
			{
				UEM_DEBUG_PRINT("%d ", pBuffer[nLoop]);
			}
			UEM_DEBUG_PRINT("\n");
		}*/
	}

	*pnReceivedSize = unDataRead;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}



#ifdef __cplusplus
}
#endif


