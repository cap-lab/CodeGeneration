/*
 * UKSerialModule.c
 *
 *  Created on: 2019. 02. 18., modified from UKBluetoothModule.c
 *      Author: dowhan1128
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <UCTime.h>
#include <UCThread.h>
#include <UCSerialPort.h>

//#include <uem_serial_data.h>
#include <uem_bluetooth_data.h>  //Use bluetooth data even in Serial communication.

#include <uem_data.h>

#define CONNECT_TIMEOUT (7)
#define DESTROY_TIMEOUT (3000)

#define CONNECT_RETRY_COUNT (100)
#define SECOND_IN_MILLISECOND (1000)
#define ACCEPT_TIMEOUT (3000)
#define OPEN_TIMEOUT (3000)

static uem_result serialSend(HConnector hConnector, HUserConnector hUserHandle, IN int nTimeout, IN char *pData, IN int nDataLen, OUT int *pnSentSize)
{
	uem_result result = ERR_UEM_UNKNOWN;
	HSerialPort hSerialPort = NULL;

	hSerialPort = (HSerialPort) hUserHandle;

	result = UCSerialPort_Send(hSerialPort, nTimeout, pData, nDataLen, pnSentSize);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result serialReceive(HConnector hConnector, HUserConnector hUserHandle, IN int nTimeout, IN OUT char *pBuffer, IN int nBufferLen, OUT int *pnReceivedSize)
{
	uem_result result = ERR_UEM_UNKNOWN;
	HSerialPort hSerialPort = NULL;

	hSerialPort = (HSerialPort) hUserHandle;

	result = UCSerialPort_Receive(hSerialPort, nTimeout, pBuffer, nBufferLen, pnReceivedSize);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static void *serialSlaveHandlingThread(void *pData)
{
	SSerialInfo *pstSerialInfo = NULL;
	uem_result result = ERR_UEM_UNKNOWN;
	int nRetryCount = 0;

	pstSerialInfo = (SSerialInfo *) pData;

	result = UCSerialPort_Open(pstSerialInfo->hSerialPort);

	result = UKConnector_SetConnector(pstSerialInfo->hConnector, pstSerialInfo->hSerialPort,
											serialSend, serialReceive);
	ERRIFGOTO(result, _EXIT);

	//02.17. error 발생지점
	while(nRetryCount < CONNECT_RETRY_COUNT)
	{
		if(g_bSystemExit == TRUE)
		{
			ERRASSIGNGOTO(result, ERR_UEM_SUSPEND, _EXIT);
		}

		result = UKSerialCommunicationManager_AcceptHandshake(pstSerialInfo->hManager);
		if(result == ERR_UEM_NET_TIMEOUT)
		{
			nRetryCount++;
			//FIXME : Timeout value is currently set as RECEIVE_TIMEOUT(=3 seconds) in UKSerialCommunicationManager_AcceptHandshake,
			// but the variable is not defined in this file.
			int RECEIVE_TIMEOUT = 3;
			UEM_DEBUG_PRINT("Timeout] Connection takes more than %d seconds.\n", RECEIVE_TIMEOUT);
			continue;
		}
		ERRIFGOTO(result, _EXIT);
		break;
	}

	pstSerialInfo->bInitialized = TRUE;

	result = UKSerialCommunicationManager_Run(pstSerialInfo->hManager);
	ERRIFGOTO(result, _EXIT);

_EXIT:
	if(result != ERR_UEM_NOERROR && g_bSystemExit == FALSE)
	{
		UEM_DEBUG_PRINT("serialSlaveHandlingThread is exited with error: %08x\n", result);
	}
	if (pstSerialInfo->hSerialPort != NULL)
	{
		UCSerialPort_Destroy(&(pstSerialInfo->hSerialPort));
	}
	return NULL;
}

static void *serialMasterHandlingThread(void *pData)
{
	SSerialInfo *pstSerialPortInfo = NULL;
	uem_result result = ERR_UEM_UNKNOWN;
	int nRetryCount = 0;

	pstSerialPortInfo = (SSerialInfo *) pData;

	while(nRetryCount < CONNECT_RETRY_COUNT)
	{
		if(g_bSystemExit == TRUE)
		{
			ERRASSIGNGOTO(result, ERR_UEM_SUSPEND, _EXIT);
		}

		result = UCSerialPort_Open(pstSerialPortInfo->hSerialPort);
		if(result != ERR_UEM_NOERROR)
		{
			UEM_DEBUG_PRINT("cannot open Serial port. retrying ... %08x\n",result);
			nRetryCount++;
			UCTime_Sleep(CONNECT_TIMEOUT*SECOND_IN_MILLISECOND);
			continue;
		}
		ERRIFGOTO(result, _EXIT);
		break;
	}

	result = UKConnector_SetConnector(pstSerialPortInfo->hConnector, pstSerialPortInfo->hSerialPort,
										serialSend, serialReceive);
	ERRIFGOTO(result, _EXIT);

	while(nRetryCount < CONNECT_RETRY_COUNT)
	{
		if(g_bSystemExit == TRUE)
		{
			ERRASSIGNGOTO(result, ERR_UEM_SUSPEND, _EXIT);
		}

		//retry until Handshake successfully done.
		//Handshake itself contains receive/timeout whileloop. Once it Send message, it waits to receive data
		//3 seconds x 3 times.
		result = UKSerialCommunicationManager_Handshake(pstSerialPortInfo->hManager);

		if(result == ERR_UEM_CONNECT_ERROR)
		{
			nRetryCount++;
			continue;
		}
		else if(result == ERR_UEM_NET_TIMEOUT)
		{
			UEM_DEBUG_PRINT("Timeout] Connection takes more than %d seconds.\n", CONNECT_TIMEOUT);
		}
		ERRIFGOTO(result, _EXIT);
		break;
	}

	pstSerialPortInfo->bInitialized = TRUE;

	result = UKSerialCommunicationManager_Run(pstSerialPortInfo->hManager);
	ERRIFGOTO(result, _EXIT);
_EXIT:
	if(result != ERR_UEM_NOERROR && g_bSystemExit == FALSE)
	{
		UEM_DEBUG_PRINT("serialMasterHandlingThread is exited with error: %08x\n", result);
		// exit forcedly
		g_bSystemExit = TRUE;
	}
	return NULL;
}

static uem_result createConnectorsAndManagers(int nModuleNum, SSerialInfo astModuleInfo[], uem_bool bIsSlave)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;
	SSerialPortInfo stSerialPortInfo;

	//serial통신에서 이걸 빼면 안된다.
	for(nLoop = 0 ; nLoop < nModuleNum ; nLoop++)
	{
		stSerialPortInfo.pszSerialPortPath = (char *) astModuleInfo[nLoop].pszSerialPortPath;

		result = UCSerialPort_Create(&stSerialPortInfo, FALSE, &(astModuleInfo[nLoop].hSerialPort));
		ERRIFGOTO(result, _EXIT);

		result = UKConnector_Create(&(astModuleInfo[nLoop].hConnector));
		ERRIFGOTO(result, _EXIT);

		result = UKSerialCommunicationManager_Create(astModuleInfo[nLoop].hConnector,
					astModuleInfo[nLoop].nMaxChannelAccessNum, &(astModuleInfo[nLoop].hManager));
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKSerialModule_Initialize()
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;

	result = createConnectorsAndManagers(g_nSerialSlaveInfoNum, g_astSerialSlaveInfo, TRUE);
	ERRIFGOTO(result, _EXIT);

	for(nLoop = 0 ; nLoop < g_nSerialSlaveInfoNum ; nLoop++)
	{
		result = UCThread_Create(serialSlaveHandlingThread, &g_astSerialSlaveInfo[nLoop], &(g_astSerialSlaveInfo[nLoop].hHandlingThread));
		ERRIFGOTO(result, _EXIT);
	}

	result = createConnectorsAndManagers(g_nSerialMasterInfoNum, g_astSerialMasterInfo, FALSE);
	ERRIFGOTO(result, _EXIT);

	for(nLoop = 0 ; nLoop < g_nSerialMasterInfoNum ; nLoop++)
	{
		result = UCThread_Create(serialMasterHandlingThread, &g_astSerialMasterInfo[nLoop], &(g_astSerialMasterInfo[nLoop].hHandlingThread));
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static void serialModuleFinalize(int nModuleNum, SSerialInfo astModuleInfo[])
{
	int nLoop = 0;

	for(nLoop = 0 ; nLoop < nModuleNum ; nLoop++)
	{
		astModuleInfo[nLoop].bInitialized = FALSE;

		// ignore error for destruction
		UCThread_Destroy(&(astModuleInfo[nLoop].hHandlingThread), FALSE, DESTROY_TIMEOUT);
	}

	for(nLoop = 0 ; nLoop < nModuleNum ; nLoop++)
	{
		// ignore error for destruction
		UKSerialCommunicationManager_Destroy(&(astModuleInfo[nLoop].hManager));
		// ignore error for destruction
		UKConnector_Destroy(&(astModuleInfo[nLoop].hConnector));
		// ignore error for destruction
		//UCSerialPort_Destroy(&(astModuleInfo[nLoop].hSerialPort));
		UCSerialPort_Close(astModuleInfo[nLoop].hSerialPort);
	}
}

uem_result UKSerialModule_Finalize()
{
	uem_result result = ERR_UEM_UNKNOWN;

	serialModuleFinalize(g_nSerialMasterInfoNum, g_astSerialMasterInfo);
	serialModuleFinalize(g_nSerialSlaveInfoNum, g_astSerialSlaveInfo);

	result = ERR_UEM_NOERROR;
//_EXIT:
	return result;
}

