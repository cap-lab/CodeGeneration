/*
 * UKBluetoothModule.c
 *
 *  Created on: 2018. 10. 8.
 *      Author: chjej202
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <UCTime.h>
#include <UCThread.h>
#include <UCDynamicSocket.h>

#include <uem_bluetooth_data.h>
#include <uem_data.h>

#define CONNECT_TIMEOUT (3)
#define DESTROY_TIMEOUT (3000)

#define CONNECT_RETRY_COUNT (100)
#define SECOND_IN_MILLISECOND (1000)
#define ACCEPT_TIMEOUT (3000)

static uem_result bluetoothSend(HConnector hConnector, HUserConnector hUserHandle, IN int nTimeout, IN char *pData, IN int nDataLen, OUT int *pnSentSize)
{
	uem_result result = ERR_UEM_UNKNOWN;
	HSocket hSocket = NULL;

	hSocket = (HSocket) hUserHandle;

	result = UCDynamicSocket_Send(hSocket, nTimeout, pData, nDataLen, pnSentSize);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result bluetoothReceive(HConnector hConnector, HUserConnector hUserHandle, IN int nTimeout, IN OUT char *pBuffer, IN int nBufferLen, OUT int *pnReceivedSize)
{
	uem_result result = ERR_UEM_UNKNOWN;
	HSocket hSocket = NULL;

	hSocket = (HSocket) hUserHandle;

	result = UCDynamicSocket_Receive(hSocket, nTimeout, pBuffer, nBufferLen, pnReceivedSize);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static void *bluetoothSlaveHandlingThread(void *pData)
{
	SBluetoothInfo *pstBluetoothInfo = NULL;
	uem_result result = ERR_UEM_UNKNOWN;
	SSocketInfo stSocketInfo;
	HSocket hServerSocket = NULL;

	pstBluetoothInfo = (SBluetoothInfo *) pData;

	stSocketInfo.enSocketType = SOCKET_TYPE_BLUETOOTH;
	stSocketInfo.nPort = 0;
	stSocketInfo.pszSocketPath = (char *) pstBluetoothInfo->pszTargetMacAddress;

	result = UCDynamicSocket_Create(&stSocketInfo, TRUE, &hServerSocket);
	ERRIFGOTO(result, _EXIT);

	result = UCDynamicSocket_Bind(hServerSocket);
	ERRIFGOTO(result, _EXIT);

	result = UCDynamicSocket_Listen(hServerSocket);
	ERRIFGOTO(result, _EXIT);

	result = UCDynamicSocket_Accept(hServerSocket, ACCEPT_TIMEOUT, pstBluetoothInfo->hSocket);
	ERRIFGOTO(result, _EXIT);

	result = UKConnector_SetConnector(pstBluetoothInfo->hConnector, pstBluetoothInfo->hSocket,
											bluetoothSend, bluetoothReceive);
	ERRIFGOTO(result, _EXIT);

	result = UKSerialCommunicationManager_AcceptHandshake(pstBluetoothInfo->hManager);
	ERRIFGOTO(result, _EXIT);

	pstBluetoothInfo->bInitialized = TRUE;

	result = UKSerialCommunicationManager_Run(pstBluetoothInfo->hManager);
	ERRIFGOTO(result, _EXIT);

_EXIT:
	if(result != ERR_UEM_NOERROR && g_bSystemExit == FALSE)
	{
		UEM_DEBUG_PRINT("bluetoothSlaveHandlingThread is exited with error: %08x\n", result);
	}
	if (hServerSocket != NULL)
	{
		UCDynamicSocket_Destroy(&hServerSocket);
	}
	return NULL;
}


static void *bluetoothMasterHandlingThread(void *pData)
{
	SBluetoothInfo *pstBluetoothInfo = NULL;
	uem_result result = ERR_UEM_UNKNOWN;
	int nRetryCount = 0;

	pstBluetoothInfo = (SBluetoothInfo *) pData;

	while(nRetryCount < CONNECT_RETRY_COUNT)
	{
		if(g_bSystemExit == TRUE)
		{
			ERRASSIGNGOTO(result, ERR_UEM_SUSPEND, _EXIT);
		}

		result = UCDynamicSocket_Connect(pstBluetoothInfo->hSocket, CONNECT_TIMEOUT);
		if(result == ERR_UEM_NET_TIMEOUT || result == ERR_UEM_CONNECT_ERROR)
		{
			nRetryCount++;
			UCTime_Sleep(CONNECT_TIMEOUT*SECOND_IN_MILLISECOND);
			continue;
		}
		ERRIFGOTO(result, _EXIT);
		break;
	}

	result = UKConnector_SetConnector(pstBluetoothInfo->hConnector, pstBluetoothInfo->hSocket,
										bluetoothSend, bluetoothReceive);
	ERRIFGOTO(result, _EXIT);

	// handshake first
	result = UKSerialCommunicationManager_Handshake(pstBluetoothInfo->hManager);
	ERRIFGOTO(result, _EXIT);

	pstBluetoothInfo->bInitialized = TRUE;

	result = UKSerialCommunicationManager_Run(pstBluetoothInfo->hManager);
	ERRIFGOTO(result, _EXIT);
_EXIT:
	if(result != ERR_UEM_NOERROR && g_bSystemExit == FALSE)
	{
		UEM_DEBUG_PRINT("bluetoothMasterHandlingThread is exited with error: %08x\n", result);
		// exit forcedly
		g_bSystemExit = TRUE;
	}
	return NULL;
}

static uem_result createConnectorsAndManagers(int nModuleNum, SBluetoothInfo astModuleInfo[], uem_bool bIsSlave)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;
	SSocketInfo stSocketInfo;

	for(nLoop = 0 ; nLoop < nModuleNum ; nLoop++)
	{
		stSocketInfo.enSocketType = SOCKET_TYPE_BLUETOOTH;
		stSocketInfo.nPort = 0;
		if(bIsSlave == TRUE)
		{
			stSocketInfo.pszSocketPath = NULL;
		}
		else
		{
			stSocketInfo.pszSocketPath = (char *) astModuleInfo[nLoop].pszTargetMacAddress;
		}

		result = UCDynamicSocket_Create(&stSocketInfo, FALSE, &(astModuleInfo[nLoop].hSocket));
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


uem_result UKBluetoothModule_Initialize()
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;

	result = createConnectorsAndManagers(g_nBluetoothSlaveNum, g_astBluetoothSlaveInfo, TRUE);
	ERRIFGOTO(result, _EXIT);

	for(nLoop = 0 ; nLoop < g_nBluetoothSlaveNum ; nLoop++)
	{
		result = UCThread_Create(bluetoothSlaveHandlingThread, &g_astBluetoothSlaveInfo[nLoop], &(g_astBluetoothSlaveInfo[nLoop].hHandlingThread));
		ERRIFGOTO(result, _EXIT);
	}

	result = createConnectorsAndManagers(g_nBluetoothMasterNum, g_astBluetoothMasterInfo, FALSE);
	ERRIFGOTO(result, _EXIT);

	for(nLoop = 0 ; nLoop < g_nBluetoothMasterNum ; nLoop++)
	{
		result = UCThread_Create(bluetoothMasterHandlingThread, &g_astBluetoothMasterInfo[nLoop], &(g_astBluetoothMasterInfo[nLoop].hHandlingThread));
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static void bluetoothModuleFinalize(int nModuleNum, SBluetoothInfo astModuleInfo[])
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
		UCDynamicSocket_Destroy(&(astModuleInfo[nLoop].hSocket));
	}
}

uem_result UKBluetoothModule_Finalize()
{
	uem_result result = ERR_UEM_UNKNOWN;

	bluetoothModuleFinalize(g_nBluetoothMasterNum, g_astBluetoothMasterInfo);
	bluetoothModuleFinalize(g_nBluetoothSlaveNum, g_astBluetoothSlaveInfo);

	result = ERR_UEM_NOERROR;
//_EXIT:
	return result;
}

