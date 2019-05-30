/*
 * UKServiceRunner.c
 *
 *  Created on: 2019. 5. 23.
 *      Author: jej
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <UCAlloc.h>
#include <UCTime.h>
#include <UCThread.h>

#include <UKUEMProtocol.h>
#include <UKVirtualCommunication.h>

#include <uem_remote_data.h>

#define ACCEPT_TIMEOUT (3)

#define DESTROY_TIMEOUT_IN_MS (3000)

#define CONNECT_TIMEOUT (7)
#define CONNECT_RETRY_COUNT (100)

#define SECOND_IN_MILLISECOND (1000)

struct _SAggregateServiceThreadData {
	SAggregateServiceInfo *pstServiceInfo;
	void *pSocketInfo;
};

static void *aggregateClientThread(void *pData)
{
	SAggregateServiceInfo *pstServiceInfo = NULL;
	uem_result result = ERR_UEM_UNKNOWN;
	SVirtualCommunicationAPI *pstAPI = NULL;
	int nRetryCount = 0;

	pstServiceInfo = (SAggregateServiceInfo *) pData;

	while(nRetryCount < CONNECT_RETRY_COUNT)
	{
		if(g_bSystemExit == TRUE)
		{
			ERRASSIGNGOTO(result, ERR_UEM_SUSPEND, _EXIT);
		}

		result = pstAPI->fnConnect(pstServiceInfo->hSocket, CONNECT_TIMEOUT);
		if(result == ERR_UEM_CONNECT_ERROR)
		{
			nRetryCount++;
			UCTime_Sleep(CONNECT_TIMEOUT*SECOND_IN_MILLISECOND);
			continue;
		}
		else if(result == ERR_UEM_NET_TIMEOUT)
		{
			UEM_DEBUG_PRINT("Timeout] Connection takes more than %d seconds.\n", CONNECT_TIMEOUT);
		}
		ERRIFGOTO(result, _EXIT);
		break;
	}

	result = UKSerialCommunicationManager_Create(pstServiceInfo->hSocket, pstServiceInfo->pstAPI, pstServiceInfo->nMaxChannelAccessNum, &(pstServiceInfo->hManager));
	ERRIFGOTO(result, _EXIT);

	// handshake first
	result = UKSerialCommunicationManager_Handshake(pstServiceInfo->hManager);
	ERRIFGOTO(result, _EXIT);

	pstServiceInfo->bInitialized = TRUE;

	result = UKSerialCommunicationManager_Run(pstServiceInfo->hManager);
	ERRIFGOTO(result, _EXIT);
_EXIT:
	if(result != ERR_UEM_NOERROR && g_bSystemExit == FALSE)
	{
		UEM_DEBUG_PRINT("aggregate client thread is exited with error: %08x\n", result);
		// exit forcedly
		g_bSystemExit = TRUE;
	}
	return NULL;
}


static void *aggregateServiceThread(void *pData)
{
	struct _SAggregateServiceThreadData *pstThreadData = NULL;
	SAggregateServiceInfo *pstServiceInfo = NULL;
	uem_result result = ERR_UEM_UNKNOWN;
	HVirtualSocket hServerSocket = NULL;
	SVirtualCommunicationAPI *pstAPI = NULL;

	pstThreadData = (struct _SAggregateServiceThreadData *) pData;

	pstServiceInfo = pstThreadData->pstServiceInfo;
	pstAPI = pstServiceInfo->pstAPI;

	result = pstAPI->fnCreate(&hServerSocket, pstThreadData->pSocketInfo);
	ERRIFGOTO(result, _EXIT);

	result = pstAPI->fnListen(hServerSocket);
	ERRIFGOTO(result, _EXIT);

	result = pstAPI->fnAccept(hServerSocket, ACCEPT_TIMEOUT, pstServiceInfo->hSocket);
	ERRIFGOTO(result, _EXIT);
	if(result == ERR_UEM_SKIP_THIS)
	{
		// client socket is not provided, use server socket instead for communication
		pstAPI->fnDestroy(&pstServiceInfo->hSocket);
		pstServiceInfo->hSocket = hServerSocket;
		hServerSocket = NULL;
	}
	else // result == ERR_UEM_NOERROR
	{
		// do nothing (use pstServiceInfo->hSocket)
	}

	result = UKSerialCommunicationManager_Create(pstServiceInfo->hSocket, pstServiceInfo->pstAPI, pstServiceInfo->nMaxChannelAccessNum, &(pstServiceInfo->hManager));
	ERRIFGOTO(result, _EXIT);

	result = UKSerialCommunicationManager_AcceptHandshake(pstServiceInfo->hManager);
	ERRIFGOTO(result, _EXIT);

	pstServiceInfo->bInitialized = TRUE;

	result = UKSerialCommunicationManager_Run(pstServiceInfo->hManager);
	ERRIFGOTO(result, _EXIT);

_EXIT:
	if(result != ERR_UEM_NOERROR && g_bSystemExit == FALSE)
	{
		UEM_DEBUG_PRINT("aggregate service thread is exited with error: %08x\n", result);
	}
	if(pstServiceInfo->hSocket != NULL)
	{
		pstAPI->fnDestroy(&pstServiceInfo->hSocket);
	}
	if (hServerSocket != NULL)
	{
		pstAPI->fnDestroy(&hServerSocket);
	}
	SAFEMEMFREE(pstThreadData);
	return NULL;
}


// slave
uem_result UKServiceRunner_StartAggregatedService(SAggregateServiceInfo *pstServiceInfo, void *pSocketInfo)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SVirtualCommunicationAPI *pstAPI = NULL;
	struct _SAggregateServiceThreadData *pstThreadData = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(pstServiceInfo, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	IFVARERRASSIGNGOTO(pSocketInfo, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	pstAPI = pstServiceInfo->pstAPI;

	pstThreadData = (struct _SAggregateServiceThreadData *) UCAlloc_malloc(sizeof(struct _SAggregateServiceThreadData));
	ERRMEMGOTO(pstThreadData, result, _EXIT);

	result = pstAPI->fnCreate(&(pstServiceInfo->hSocket), pSocketInfo);
	ERRIFGOTO(result, _EXIT);

	pstThreadData->pstServiceInfo = pstServiceInfo;
	pstThreadData->pSocketInfo = pSocketInfo;

	result = UCThread_Create(aggregateServiceThread, pstThreadData, &(pstServiceInfo->hServiceThread));
	ERRIFGOTO(result, _EXIT);

	pstThreadData = NULL;

	result = ERR_UEM_NOERROR;
_EXIT:
	if(result != ERR_UEM_NOERROR && pstAPI != NULL)
	{
		pstAPI->fnDestroy(&(pstServiceInfo->hSocket));
	}
	SAFEMEMFREE(pstThreadData);

	return result;
}

static uem_result destroyAggreagatedService(SAggregateServiceInfo *pstServiceInfo)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SVirtualCommunicationAPI *pstAPI = NULL;

	pstAPI = pstServiceInfo->pstAPI;

	pstServiceInfo->bInitialized = FALSE;

	UCThread_Destroy(&(pstServiceInfo->hServiceThread), FALSE, DESTROY_TIMEOUT_IN_MS);

	UKSerialCommunicationManager_Destroy(&(pstServiceInfo->hManager));

	pstAPI->fnDestroy(&(pstServiceInfo->hSocket));

	result = ERR_UEM_NOERROR;

	return result;
}

// slave
uem_result UKServiceRunner_StopAggregatedService(SAggregateServiceInfo *pstServiceInfo)
{
	uem_result result = ERR_UEM_UNKNOWN;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(pstServiceInfo, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif

	destroyAggreagatedService(pstServiceInfo);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


// master
uem_result UKServiceRunner_StartAggregatedClientService(SAggregateServiceInfo *pstServiceInfo, void *pSocketInfo)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SVirtualCommunicationAPI *pstAPI = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(pstServiceInfo, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	IFVARERRASSIGNGOTO(pSocketInfo, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	pstAPI = pstServiceInfo->pstAPI;

	result = pstAPI->fnCreate(&(pstServiceInfo->hSocket), pSocketInfo);
	ERRIFGOTO(result, _EXIT);

	result = UCThread_Create(aggregateClientThread, pstServiceInfo, &(pstServiceInfo->hServiceThread));
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	if(result != ERR_UEM_NOERROR && pstAPI != NULL)
	{
		pstAPI->fnDestroy(&(pstServiceInfo->hSocket));
	}
	return result;
}

// master
uem_result UKServiceRunner_StopAggregatedClientService(SAggregateServiceInfo *pstServiceInfo)
{
	uem_result result = ERR_UEM_UNKNOWN;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(pstServiceInfo, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	destroyAggreagatedService(pstServiceInfo);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result handleHandshakeFromClient(HVirtualSocket hClientSocket, SVirtualCommunicationAPI *pstAPI)
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

	result = UKUEMProtocol_SetSocket(hProtocol, hClientSocket, pstAPI);
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

	for(nLoop = 0 ; nLoop < g_nIndividualConnectionInfoNum ; nLoop++)
	{
		if(g_astIndividualConnectionInfo[nLoop].nChannelId == nChannelId)
		{
			if(g_astIndividualConnectionInfo[nLoop].enType == PAIR_TYPE_SERVER)
			{
				bFound = TRUE;
				break;
			}
		}
	}

	if(bFound == TRUE)
	{
		result = UKUEMProtocol_SetChannelId(hProtocol, nChannelId);
		ERRIFGOTO(result, _EXIT);

		// correct case
		g_astIndividualConnectionInfo[nLoop].hSocket = hClientSocket;
		g_astIndividualConnectionInfo[nLoop].hProtocol = hProtocol;

		// TODO: check key value
		//panParam[HANDSHAKE_DEVICE_KEY_INDEX]; // check device

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

		if(hClientSocket != NULL && pstAPI != NULL)
		{
			pstAPI->fnDestroy(&hClientSocket);
		}
	}
	return result;
}


static void *individualServiceThread(void *pData)
{
	SIndividualServiceInfo *pstServiceInfo = (HVirtualSocket) pData;
	HVirtualSocket hClientSocket = NULL;
	SVirtualCommunicationAPI *pstAPI = NULL;
	uem_result result = ERR_UEM_UNKNOWN;

	pstAPI = pstServiceInfo->pstAPI;

	while(g_bSystemExit == FALSE)
	{
		if(hClientSocket == NULL)
		{
			result = pstAPI->fnCreate(&hClientSocket, NULL);
			ERRIFGOTO(result, _EXIT);
		}

		result = pstAPI->fnAccept(pstServiceInfo->hSocket, ACCEPT_TIMEOUT, hClientSocket);
		if(result == ERR_UEM_NET_TIMEOUT)
		{
			continue;
		}
		ERRIFGOTO(result, _EXIT);

		// do something (handshake)
		result = handleHandshakeFromClient(hClientSocket, pstAPI);
		// skip error to preserve TCP server accept

		hClientSocket = NULL;
	}

_EXIT:
	if(hClientSocket != NULL)
	{
		pstAPI->fnDestroy(&hClientSocket);
	}
	if(result != ERR_UEM_NOERROR && g_bSystemExit == FALSE)
	{
		UEM_DEBUG_PRINT("tcpServerThread is exited with error: %08x\n", result);
	}
	return NULL;
}


uem_result UKServiceRunner_StartIndividualService(SIndividualServiceInfo *pstServiceInfo, void *pSocketInfo)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SVirtualCommunicationAPI *pstAPI = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(pstServiceInfo, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	IFVARERRASSIGNGOTO(pSocketInfo, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	IFVARERRASSIGNGOTO(pstServiceInfo->pstAPI, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	pstAPI = pstServiceInfo->pstAPI;

	result = pstAPI->fnCreate(&(pstServiceInfo->hSocket), pSocketInfo);
	ERRIFGOTO(result, _EXIT);

	result = pstAPI->fnListen(pstServiceInfo->hSocket);
	ERRIFGOTO(result, _EXIT);

	result = UCThread_Create(individualServiceThread, pstServiceInfo, &(pstServiceInfo->hServiceThread));
	ERRIFGOTO(result, _EXIT);


	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKServiceRunner_StopIndividualService(SIndividualServiceInfo *pstServiceInfo)
{
	uem_result result = ERR_UEM_UNKNOWN;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(pstServiceInfo, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	IFVARERRASSIGNGOTO(pstServiceInfo->pstAPI, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	result = UCThread_Destroy(&(pstServiceInfo->hServiceThread), FALSE, DESTROY_TIMEOUT_IN_MS);
	ERRIFGOTO(result, _EXIT);

	result = pstServiceInfo->pstAPI->fnDestroy(&(pstServiceInfo->hSocket));
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

