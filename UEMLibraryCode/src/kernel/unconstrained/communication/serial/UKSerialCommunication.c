/*
 * UKSerialCommunication.c
 *
 *  Created on: 2019. 5. 23.
 *      Author: jej
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <UCSerialPort.h>

#include <UKVirtualCommunication.h>

#include <uem_serial_data.h>

uem_result UKSerialCommunication_Create(OUT HVirtualSocket *phSocket, void *pSocketInfo)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSerialAccessInfo *pstSerialInfo = NULL;
	SSerialPortInfo stSerialPortInfo;
	HSerialPort hSerialPort = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(phSocket, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	pstSerialInfo = (SSerialAccessInfo *) pSocketInfo;

	if(pstSerialInfo != NULL)
	{
		stSerialPortInfo.pszSerialPortPath = pstSerialInfo->pszSerialPortPath;

		result = UCSerialPort_Create(&stSerialPortInfo, &hSerialPort);
		ERRIFGOTO(result, _EXIT);
	}
	else // for serial communication, client socket for accept is not used
	{
		hSerialPort = NULL;
	}

	*phSocket = (HVirtualSocket *) hSerialPort;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKSerialCommunication_Destroy(HVirtualSocket *phSocket)
{
	uem_result result = ERR_UEM_UNKNOWN;
	HSerialPort hSerialPort = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(phSocket, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	IFVARERRASSIGNGOTO(*phSocket, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	hSerialPort = (HSerialPort) *phSocket;

	result = UCSerialPort_Destroy(&hSerialPort);
	ERRIFGOTO(result, _EXIT);

	*phSocket = NULL;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKSerialCommunication_Connect(HVirtualSocket hSocket, int nTimeout)
{
	uem_result result = ERR_UEM_UNKNOWN;
	HSerialPort hSerialPort = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(hSocket, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	hSerialPort = (HSerialPort) hSocket;

	// nTimeout is not used here

	result = UCSerialPort_Open(hSerialPort);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKSerialCommunication_Disconnect(HVirtualSocket hSocket)
{
	uem_result result = ERR_UEM_UNKNOWN;
	HSerialPort hSerialPort = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(hSocket, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	hSerialPort = (HSerialPort) hSocket;

	result = UCSerialPort_Close(hSerialPort);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKSerialCommunication_Listen(HVirtualSocket hSocket)
{
	uem_result result = ERR_UEM_UNKNOWN;
	HSerialPort hSerialPort = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(hSocket, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	hSerialPort = (HSerialPort) hSocket;

	result = UCSerialPort_Open(hSerialPort);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKSerialCommunication_Accept(HVirtualSocket hSocket, int nTimeout, IN OUT HVirtualSocket hAcceptedSocket)
{
	uem_result result = ERR_UEM_UNKNOWN;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(hSocket, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	// do nothing

	result = ERR_UEM_SKIP_THIS;
_EXIT:
	return result;
}


uem_result UKSerialCommunication_Send(HVirtualSocket hSocket, IN int nTimeout, IN char *pData, IN int nDataLen, OUT int *pnSentSize)
{
	uem_result result = ERR_UEM_UNKNOWN;
	HSerialPort hSerialPort = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(hSocket, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	hSerialPort = (HSerialPort) hSocket;

	result = UCSerialPort_Send(hSerialPort, nTimeout, pData, nDataLen, pnSentSize);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKSerialCommunication_Receive(HVirtualSocket hSocket, IN int nTimeout, IN OUT char *pBuffer, IN int nBufferLen, OUT int *pnReceivedSize)
{
	uem_result result = ERR_UEM_UNKNOWN;
	HSerialPort hSerialPort = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(hSocket, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	hSerialPort = (HSerialPort) hSocket;

	result = UCSerialPort_Receive(hSerialPort, nTimeout, pBuffer, nBufferLen, pnReceivedSize);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


