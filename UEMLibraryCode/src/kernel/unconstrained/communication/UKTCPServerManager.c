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
#include <UKUEMProtocol.h>

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


SExternalCommunicationInfo g_astExternalCommunicationInfo[] = {
	{
		18,
		COMMUNICATION_TYPE_TCP_SERVER_READER,
		NULL,
		NULL,
	},
	{
		19,
		COMMUNICATION_TYPE_TCP_CLIENT_READER,
		NULL,
		NULL,
	},
	{
		20,
		COMMUNICATION_TYPE_TCP_SERVER_WRITER,
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




static uem_result handleHandshakeFromClient(HSocket hClientSocket)
{
	uem_result result = ERR_UEM_UNKNOWN;
	HUEMProtocol hProtocol = NULL;
	EMessageType enMessageType = MESSAGE_TYPE_NONE;
	int nParamNum = 0;
	int *panParam = NULL;
	int nLoop = 0;
	int nChannelId = INVALID_CHANNEL_ID;
	uem_bool bFound = FALSE;

	result = UKUEMProtocol_Create(&hProtocol);
	ERRIFGOTO(result, _EXIT);

	result = UKUEMProtocol_SetSocket(hProtocol, hClientSocket);
	ERRIFGOTO(result, _EXIT);

	// if it cannot receive anything until timeout, handle as an error
	result = UKUEMProtocol_Receive(hProtocol);
	ERRIFGOTO(result, _EXIT);

	result = UKUEMProtocol_GetRequestFromReceivedData(hProtocol, &enMessageType, &nParamNum, &panParam);
	ERRIFGOTO(result, _EXIT);

	if(enMessageType != MESSAGE_TYPE_HANDSHAKE)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_CONTROL, _EXIT);
	}

	nChannelId = panParam[HANDSHAKE_CHANNELD_ID_INDEX]; // channel_id

	for(nLoop = 0 ; nLoop < ARRAYLEN(g_astExternalCommunicationInfo) ; nLoop++)
	{
		if(g_astExternalCommunicationInfo[nLoop].nChannelId == nChannelId)
		{
			if(g_astExternalCommunicationInfo[nLoop].enType == COMMUNICATION_TYPE_TCP_SERVER_READER ||
					g_astExternalCommunicationInfo[nLoop].enType == COMMUNICATION_TYPE_TCP_SERVER_WRITER)
			{
				bFound = TRUE;
				break;
			}
		}
	}

	if(bFound == TRUE)
	{
		// correct case
		g_astExternalCommunicationInfo[nLoop].hSocket = hClientSocket;
		g_astExternalCommunicationInfo[nLoop].hProtocol = hProtocol;

		// TODO: check key value
		panParam[HANDSHAKE_DEVICE_KEY_INDEX]; // check device

		// TODO: set key value
		result = UKUEMProtocol_SetResultMessage(hProtocol, ERR_UEMPROTOCOL_NOERROR, 0);
		ERRIFGOTO(result, _EXIT);
	}
	else
	{
		// error case
		result = UKUEMProtocol_SetResultMessage(hProtocol, ERR_UEMPROTOCOL_ERROR, 0);
		ERRIFGOTO(result, _EXIT);
	}

	result = UKUEMProtocol_Send(hProtocol);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	if(result != ERR_UEM_NOERROR)
	{
		if(hProtocol != NULL)
		{
			UKUEMProtocol_Destroy(&hProtocol);
		}

		if(hClientSocket != NULL)
		{
			UCDynamicSocket_Destroy(&hClientSocket);
		}
	}
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
		result = handleHandshakeFromClient(hClientSocket);
		// skip error to preserve TCP server accept

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
