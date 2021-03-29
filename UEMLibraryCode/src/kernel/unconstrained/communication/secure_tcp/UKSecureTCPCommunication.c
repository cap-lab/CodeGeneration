/*
 * UKSecureTCPCommunication.c
 *
 *  Created on: 2020. 5. 21.
 *      Author: jrkim
 */


#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <UCSecureTCPSocket.h>

#include <UKVirtualCommunication.h>

#include <uem_secure_tcp_data.h>

#define UNUSED_PORT_NUM (1)


uem_result UKSecureTCPCommunication_Create(OUT HVirtualSocket *phSocket, void *pSocketInfo)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSecureTCPInfo *pstSSLTCPInfo = NULL;
	SSecureSocketInfo stSSLSocketInfo;
	uem_bool bIsServer = FALSE;
	HSSLSocket hSocket = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(phSocket, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	pstSSLTCPInfo = (SSecureTCPInfo *) pSocketInfo;

	stSSLSocketInfo.stSocketInfo.enSocketType = SOCKET_TYPE_TCP;

	if(pstSSLTCPInfo != NULL)
	{
		stSSLSocketInfo.pstKeyInfo = pstSSLTCPInfo->pstKeyInfo;

		stSSLSocketInfo.stSocketInfo.nPort = pstSSLTCPInfo->stTCPInfo.nPort;
		stSSLSocketInfo.stSocketInfo.pszSocketPath = pstSSLTCPInfo->stTCPInfo.pszIPAddress;

		switch(pstSSLTCPInfo->stTCPInfo.enType)
		{
		case PAIR_TYPE_SERVER:
			bIsServer = TRUE;
			break;
		case PAIR_TYPE_CLIENT:
			bIsServer = FALSE;
			break;
		default:
			ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
			break;
		}
	}
	else // client socket for accept
	{
		stSSLSocketInfo.pstKeyInfo = NULL;
		stSSLSocketInfo.stSocketInfo.nPort = UNUSED_PORT_NUM;
		stSSLSocketInfo.stSocketInfo.pszSocketPath = NULL;

		bIsServer = FALSE;
	}

	result = UCSecureTCPSocket_Create(&stSSLSocketInfo, bIsServer, &hSocket);
	ERRIFGOTO(result, _EXIT);

	*phSocket = (HVirtualSocket *) hSocket;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKSecureTCPCommunication_Destroy(HVirtualSocket *phSocket)
{
	uem_result result = ERR_UEM_UNKNOWN;
	HSSLSocket hSocket = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(phSocket, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	IFVARERRASSIGNGOTO(*phSocket, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	hSocket = (HSSLSocket) *phSocket;

	result = UCSecureTCPSocket_Destroy(&hSocket);
	ERRIFGOTO(result, _EXIT);

	*phSocket = NULL;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKSecureTCPCommunication_Connect(HVirtualSocket hSocket, int nTimeout)
{
	uem_result result = ERR_UEM_UNKNOWN;
	HSSLSocket hTCPSocket = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(hSocket, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	hTCPSocket = (HSSLSocket) hSocket;

	result = UCSecureTCPSocket_Connect(hTCPSocket, nTimeout);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKSecureTCPCommunication_Disconnect(HVirtualSocket hSocket)
{
	uem_result result = ERR_UEM_UNKNOWN;
	HSSLSocket hTCPSocket = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(hSocket, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	hTCPSocket = (HSSLSocket) hSocket;

	result = UCSecureTCPSocket_Disconnect(hTCPSocket);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKSecureTCPCommunication_Listen(HVirtualSocket hSocket)
{
	uem_result result = ERR_UEM_UNKNOWN;
	HSSLSocket hServerSocket = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(hSocket, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	hServerSocket = (HSSLSocket) hSocket;

	result = UCSecureTCPSocket_Bind(hServerSocket);
	ERRIFGOTO(result, _EXIT);

	result = UCSecureTCPSocket_Listen(hServerSocket);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKSecureTCPCommunication_Accept(HVirtualSocket hSocket, int nTimeout, IN OUT HVirtualSocket hAcceptedSocket)
{
	uem_result result = ERR_UEM_UNKNOWN;
	HSSLSocket hServerSocket = NULL;
	HSSLSocket hClientSocket = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(hSocket, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	IFVARERRASSIGNGOTO(hAcceptedSocket, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	hServerSocket = (HSSLSocket) hSocket;
	hClientSocket = (HSSLSocket) hAcceptedSocket;

	result = UCSecureTCPSocket_Accept(hServerSocket, nTimeout, hClientSocket);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKSecureTCPCommunication_Send(HVirtualSocket hSocket, IN int nTimeout, IN char *pData, IN int nDataLen, OUT int *pnSentSize)
{
	uem_result result = ERR_UEM_UNKNOWN;
	HSSLSocket hTCPSocket = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(hSocket, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	hTCPSocket = (HSSLSocket) hSocket;

	result = UCSecureTCPSocket_Send(hTCPSocket, nTimeout, pData, nDataLen, pnSentSize);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKSecureTCPCommunication_Receive(HVirtualSocket hSocket, IN int nTimeout, IN OUT char *pBuffer, IN int nBufferLen, OUT int *pnReceivedSize)
{
	uem_result result = ERR_UEM_UNKNOWN;
	HSSLSocket hTCPSocket = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(hSocket, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	hTCPSocket = (HSSLSocket) hSocket;

	result = UCSecureTCPSocket_Receive(hTCPSocket, nTimeout, pBuffer, nBufferLen, pnReceivedSize);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}
