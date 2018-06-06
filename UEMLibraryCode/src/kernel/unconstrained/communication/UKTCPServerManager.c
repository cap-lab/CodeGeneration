/*
 * UKTCPServerManager.c
 *
 *  Created on: 2018. 6. 2.
 *      Author: jej
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <UCDynamicSocket.h>

typedef struct _STCPInfo {
	int nPort;
	HSocket hServerSocket;
	HThread hServerThread;
} STCPInfo;

#define SOCKET_ACCEPT_TIMEOUT (3)
#define SERVER_THREAD_DESTROY_TIMEOUT (3000)

STCPInfo g_astTCPInfo[] = {
	{
		7942,
		NULL,
		NULL,
	},
	{
		7952,
		NULL,
		NULL,
	},
};


uem_result UKTCPServerManager_Initialize()
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;
	SSocketInfo stSocketInfo;
	HSocket hSocket;

	for(nLoop = 0 ; nLoop < ARRAYLEN(g_astTCPInfo) ; nLoop++)
	{
		stSocketInfo.enSocketType = SOCKET_TYPE_TCP;
		stSocketInfo.nPort = g_astTCPInfo[nLoop].nPort;
		stSocketInfo.pszSocketPath = NULL;

		result = UCDynamicSocket_Create(&stSocketInfo, TRUE, &(g_astTCPInfo[nLoop].hServerSocket));
		ERRIFGOTO(result, _EXIT);
	}

	for(nLoop = 0 ; nLoop < ARRAYLEN(g_astTCPInfo) ; nLoop++)
	{
		result = UCDynamicSocket_Bind(g_astTCPInfo[nLoop].hServerSocket);
		ERRIFGOTO(result, _EXIT);

		result = UCDynamicSocket_Listen(g_astTCPInfo[nLoop].hServerSocket);
		ERRIFGOTO(result, _EXIT);
	}

	for(nLoop = 0 ; nLoop < ARRAYLEN(g_astTCPInfo) ; nLoop++)
	{
		result = UCThread_Create(tcpServerThread, &g_astTCPInfo[nLoop], &(g_astTCPInfo[nLoop].hServerThread));
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static void *tcpServerThread(void *pData)
{
	STCPInfo *pstInfo = (STCPInfo *) pData;
	HSocket hClientSocket = NULL;
	uem_result result;
	SSocketInfo stSocketInfo;

	stSocketInfo.enSocketType = SOCKET_TYPE_TCP;
	stSocketInfo.nPort = pstInfo->nPort;
	stSocketInfo.pszSocketPath = NULL;

	while(g_bSystemExit == FALSE)
	{
		if(hClientSocket == NULL)
		{
			result = UCDynamicSocket_Create(&stSocketInfo, FALSE, &hClientSocket);
			ERRIFGOTO(result, _EXIT);
		}

		result = UCDynamicSocket_Accept(pstInfo->hServerSocket, SOCKET_ACCEPT_TIMEOUT, hClientSocket);
		if(result == ERR_UEM_NET_TIMEOUT)
		{
			continue;
		}
		ERRIFGOTO(result, _EXIT);

		// do something (handshake)

		hClientSocket = NULL;
	}


_EXIT:
	if(hClientSocket != NULL)
	{
		UCDynamicSocket_Destroy(&hClientSocket);
	}
	if(result != ERR_UEM_NOERROR && g_bSystemExit == FALSE)
	{
		UEM_DEBUG_PRINT("tcpServerThread is exited with error: %d\n", result);
	}
	return NULL;
}


uem_result UKTCPServerManager_AddServer()
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKTCPServerManager_RunServers()
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKTCPServerManager_Finalize()
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;

	for(nLoop = 0 ; nLoop < ARRAYLEN(g_astTCPInfo) ; nLoop++)
	{
		result = UCThread_Destroy(&(g_astTCPInfo[nLoop].hServerThread), FALSE, SERVER_THREAD_DESTROY_TIMEOUT);
		ERRIFGOTO(result, _EXIT);
	}

	for(nLoop = 0 ; nLoop < ARRAYLEN(g_astTCPInfo) ; nLoop++)
	{
		result = UCDynamicSocket_Destroy(&(g_astTCPInfo[nLoop].hServerSocket));
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}
