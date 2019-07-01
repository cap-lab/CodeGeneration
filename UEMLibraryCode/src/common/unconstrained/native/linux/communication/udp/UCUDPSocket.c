/*
 * UCUDPSocket.c
 *
 *  Created on: 2019. 6. 20.
 *      Author: wecracy
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#ifndef WIN32
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#else
#include <winsock.h>
#endif

#include <uem_common.h>

#ifndef WIN32
	#ifdef DEBUG_PRINT
#include <errno.h>
#include <string.h>
	#endif
#endif

#include <UCBasic.h>

#include <UCDynamicSocket.h>

uem_result UCUDPSocket_Bind(HSocket hSocket)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SUCSocket *pstSocket = NULL;
    struct sockaddr_in stUDPServerAddr;
    int nBroadcast = 1;
    int nRet = 0;

	pstSocket = (SUCSocket *) hSocket;

   pstSocket->nSocketfd = socket(AF_INET, SOCK_DGRAM, 0);
   if(pstSocket->nSocketfd < 0)
   {
	   ERRASSIGNGOTO(result, ERR_UEM_SOCKET_ERROR, _EXIT);
   }

   nRet = setsockopt(pstSocket->nSocketfd, SOL_SOCKET, SO_BROADCAST, (void*)nBroadcast, sizeof(nBroadcast));
   if(nRet != 0)
   {
	   ERRASSIGNGOTO(result, ERR_UEM_SOCKET_ERROR, _EXIT);
   }

   UC_memset(&stUDPServerAddr, 0, sizeof(stUDPServerAddr));
   stUDPServerAddr.sin_family = AF_INET;
   stUDPServerAddr.sin_addr.s_addr = htonl(INADDR_BROADCAST);
   stUDPServerAddr.sin_port = htons(pstSocket->nPort);

   nRet = bind(pstSocket->nSocketfd, (struct sockaddr *)&stUDPServerAddr, sizeof(stUDPServerAddr));
   if(nRet != 0)
   {
	   UEM_DEBUG_PRINT("bind errno: %d, %s\n", errno, strerror(errno));
	   ERRASSIGNGOTO(result, ERR_UEM_BIND_ERROR, _EXIT);
   }

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UCUDPSocket_Sendto(HSocket hSocket, IN uint32_t unClientAddress, IN int nTimeout, IN char *pData, IN int nDataLen, OUT int *pnSentSize) {
	uem_result result = ERR_UEM_UNKNOWN;
	SUCSocket *pstSocket = NULL;
	struct sockaddr_in stUDPServerAddr;
	fd_set stWriteSet;
	int nDataSent = 0;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(pData, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	if (IS_VALID_HANDLE(hSocket, ID_UEM_SOCKET) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}

	if (nTimeout <= 0 || nDataLen <= 0) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
#endif
	pstSocket = (SUCSocket *) hSocket;
#ifdef ARGUMENT_CHECK
	if (pstSocket->bIsServer == TRUE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_SOCKET, _EXIT);
	}
#endif
	stUDPServerAddr.sin_family = AF_INET;
	stUDPServerAddr.sin_addr.s_addr = htonl(unClientAddress);
	stUDPServerAddr.sin_port = htons(pstSocket->nPort);
	result = selectTimeout(pstSocket->nSocketfd, NULL, &stWriteSet, NULL,nTimeout);
	ERRIFGOTO(result, _EXIT);

	nDataSent = sendto(pstSocket->nSocketfd, pData, nDataLen, 0);
	if (nDataSent <= 0)
	{
		ERRASSIGNGOTO(result, ERR_UEM_NET_SEND_ERROR, _EXIT);
	}

	if (pnSentSize != NULL)
	{
		*pnSentSize = nDataSent;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UCUDPSocket_RecvFrom(HSocket hSocket, IN char *pszClientAddress,  IN int nTimeout, IN int nBufferLen, OUT char *pBuffer, OUT int *pnRecvSize) {
	uem_result result = ERR_UEM_UNKNOWN;
	SUCSocket *pstSocket = NULL;
	struct sockaddr_in stUDPWriterAddr;
	fd_set stReadSet;
	int nDataReceived = 0;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(pBuffer, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	if (IS_VALID_HANDLE(hSocket, ID_UEM_SOCKET) == FALSE)
	{
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}

	if (nTimeout <= 0 || nBufferLen <= 0)
	{
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
#endif
	pstSocket = (SUCSocket *) hSocket;
#ifdef ARGUMENT_CHECK
	if (pstSocket->bIsServer == TRUE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_SOCKET, _EXIT);
	}
#endif
	stUDPWriterAddr.sin_family = AF_INET;
	inet_aton(pszClientAddress, &stUDPWriterAddr.sin_addr.s_addr);
	stUDPWriterAddr.sin_port = htons(pstSocket->nPort);
	result = selectTimeout(pstSocket->nSocketfd, &stReadSet, NULL, NULL, nTimeout);
	ERRIFGOTO(result, _EXIT);

	nDataReceived = recvfrom(pstSocket->nSocketfd, pBuffer, nBufferLen, 0, (struct sockaddr *)stUDPWriterAddr, sizeof(struct sockaddr_in));
	if (nDataReceived <= 0)
	{
		ERRASSIGNGOTO(result, ERR_UEM_NET_RECEIVE_ERROR, _EXIT);
	}

	if(pnRecvSize != NULL)
	{
		*pnRecvSize = nDataReceived;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}
