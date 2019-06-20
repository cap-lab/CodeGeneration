/*
 * UKSocketCommunication.c
 *
 *  Created on: 2019. 5. 23.
 *      Author: jej
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <UCDynamicSocket.h>

#include <UKVirtualCommunication.h>


uem_result UKSocketCommunication_Destroy(HVirtualSocket *phSocket)
{
	uem_result result = ERR_UEM_UNKNOWN;
	HSocket hSocket = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(phSocket, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	IFVARERRASSIGNGOTO(*phSocket, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	hSocket = (HSocket) *phSocket;

	result = UCDynamicSocket_Destroy(&hSocket);
	ERRIFGOTO(result, _EXIT);

	*phSocket = NULL;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKSocketCommunication_Connect(HVirtualSocket hSocket, int nTimeout)
{
	uem_result result = ERR_UEM_UNKNOWN;
	HSocket hTCPSocket = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(hSocket, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	hTCPSocket = (HSocket) hSocket;

	result = UCDynamicSocket_Connect(hTCPSocket, nTimeout);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKSocketCommunication_Disconnect(HVirtualSocket hSocket)
{
	uem_result result = ERR_UEM_UNKNOWN;
	HSocket hTCPSocket = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(hSocket, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	hTCPSocket = (HSocket) hSocket;

	result = UCDynamicSocket_Disconnect(hTCPSocket);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKSocketCommunication_Listen(HVirtualSocket hSocket)
{
	uem_result result = ERR_UEM_UNKNOWN;
	HSocket hServerSocket = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(hSocket, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	hServerSocket = (HSocket) hSocket;

	result = UCDynamicSocket_Bind(hServerSocket);
	ERRIFGOTO(result, _EXIT);

	result = UCDynamicSocket_Listen(hServerSocket);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKSocketCommunication_Accept(HVirtualSocket hSocket, int nTimeout, IN OUT HVirtualSocket hAcceptedSocket)
{
	uem_result result = ERR_UEM_UNKNOWN;
	HSocket hServerSocket = NULL;
	HSocket hClientSocket = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(hSocket, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	IFVARERRASSIGNGOTO(hAcceptedSocket, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	hServerSocket = (HSocket) hSocket;
	hClientSocket = (HSocket) hAcceptedSocket;

	result = UCDynamicSocket_Accept(hServerSocket, nTimeout, hClientSocket);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKSocketCommunication_Send(HVirtualSocket hSocket, IN int nTimeout, IN char *pData, IN int nDataLen, OUT int *pnSentSize)
{
	uem_result result = ERR_UEM_UNKNOWN;
	HSocket hTCPSocket = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(hSocket, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	hTCPSocket = (HSocket) hSocket;

	result = UCDynamicSocket_Send(hTCPSocket, nTimeout, pData, nDataLen, pnSentSize);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKSocketCommunication_Receive(HVirtualSocket hSocket, IN int nTimeout, IN OUT char *pBuffer, IN int nBufferLen, OUT int *pnReceivedSize)
{
	uem_result result = ERR_UEM_UNKNOWN;
	HSocket hTCPSocket = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(hSocket, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	hTCPSocket = (HSocket) hSocket;

	result = UCDynamicSocket_Receive(hTCPSocket, nTimeout, pBuffer, nBufferLen, pnReceivedSize);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}
