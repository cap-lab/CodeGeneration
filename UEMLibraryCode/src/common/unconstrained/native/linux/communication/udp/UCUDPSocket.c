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

static uem_result selectTimeout(int portfd, fd_set *pstReadSet, fd_set *pstWriteSet, fd_set *pstExceptSet, int nTimeout)
{
    uem_result result = ERR_UEM_UNKNOWN;
    struct timeval stTimeVal;
    int nRet = 0;

    if(pstReadSet != NULL)
    {
        FD_ZERO(pstReadSet);
        FD_SET(portfd, pstReadSet);
    }

    if(pstWriteSet != NULL)
    {
        FD_ZERO(pstWriteSet);
        FD_SET(portfd, pstWriteSet);
    }

    if(pstExceptSet != NULL)
   {
       FD_ZERO(pstExceptSet);
       FD_SET(portfd, pstExceptSet);
   }

    stTimeVal.tv_sec = 0;
    stTimeVal.tv_usec = nTimeout;

    nRet = select(portfd+1, pstReadSet, pstWriteSet, pstExceptSet, &stTimeVal);
    if(nRet < 0)
    {
        ERRASSIGNGOTO(result, ERR_UEM_SELECT_ERROR, _EXIT);
    }
    else if(nRet == 0)
    {
        ERRASSIGNGOTO(result, ERR_UEM_NET_TIMEOUT, _EXIT);
    }
    else
    {
        result = ERR_UEM_NOERROR;
    }
_EXIT:
    return result;
}

uem_result UCUDPSocket_Create(HSocket hSocket, SSocketInfo *pstSocketInfo, uem_bool bIsServer)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SUCSocket *pstSocket = NULL;
	int optval = 1;
	int nRet = 0;

	pstSocket = (SUCSocket *) hSocket;

	pstSocket->nSocketfd = socket(AF_INET, SOCK_DGRAM, 0);
	if(pstSocket->nSocketfd < 0)
	{
	    ERRASSIGNGOTO(result, ERR_UEM_SOCKET_ERROR, _EXIT);
	}

	nRet = setsockopt(pstSocket->nSocketfd, SOL_SOCKET, SO_BROADCAST, (void*)&optval, sizeof(optval));
	if(nRet != 0)
	{
	   ERRASSIGNGOTO(result, ERR_UEM_SOCKET_ERROR, _EXIT);
	}

	nRet = setsockopt(pstSocket->nSocketfd, SOL_SOCKET, SO_REUSEPORT, &optval, sizeof(optval));
	if(nRet != 0)
	{
	   ERRASSIGNGOTO(result, ERR_UEM_SOCKET_ERROR, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UCUDPSocket_Bind(HSocket hSocket)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SUCSocket *pstSocket = NULL;
    struct sockaddr_in stUDPServerAddr;
    int nRet = 0;

	pstSocket = (SUCSocket *) hSocket;

   UC_memset(&stUDPServerAddr, 0, sizeof(stUDPServerAddr));
   stUDPServerAddr.sin_family = AF_INET;
   stUDPServerAddr.sin_addr.s_addr = INADDR_ANY;
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

uem_result UCUDPSocket_Sendto(HSocket hSocket, IN char *pszClientAddress, IN int nTimeout, IN unsigned char *pData, IN int nDataLen, OUT int *pnSentSize) {
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

	result = selectTimeout(pstSocket->nSocketfd, NULL, &stWriteSet, NULL, nTimeout);
	ERRIFGOTO(result, _EXIT);

	stUDPServerAddr.sin_family = AF_INET;
	stUDPServerAddr.sin_addr.s_addr = inet_addr(pszClientAddress);
	stUDPServerAddr.sin_port = htons(pstSocket->nPort);
	nDataSent = sendto(pstSocket->nSocketfd, pData, nDataLen, 0, (struct sockaddr *) &stUDPServerAddr, sizeof(stUDPServerAddr));
	if (nDataSent < 0) {
		ERRASSIGNGOTO(result, ERR_UEM_NET_SEND_ERROR, _EXIT);
	}

	if (pnSentSize != NULL) {
		*pnSentSize = nDataSent;
	}

	result = ERR_UEM_NOERROR;
	_EXIT: return result;
}

uem_result UCUDPSocket_RecvFrom(HSocket hSocket, IN char *pszClientAddress, IN int nTimeout, IN int nBufferLen, OUT char *pBuffer, OUT int *pnRecvSize) {
	uem_result result = ERR_UEM_UNKNOWN;
	SUCSocket *pstSocket = NULL;
	struct sockaddr_in stUDPClientAddr;
	fd_set stReadSet;
	unsigned int nAddrLen = 0;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(pBuffer, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	if (IS_VALID_HANDLE(hSocket, ID_UEM_SOCKET) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}

	if (nTimeout < 0 || nBufferLen <= 0) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
#endif
	pstSocket = (SUCSocket *) hSocket;
#ifdef ARGUMENT_CHECK
	if (pstSocket->bIsServer == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_SOCKET, _EXIT);
	}
#endif

	result = selectTimeout(pstSocket->nSocketfd, &stReadSet, NULL, NULL, nTimeout);
	ERRIFGOTO(result, _EXIT);

	stUDPClientAddr.sin_family = AF_INET;
	stUDPClientAddr.sin_addr.s_addr = inet_addr(pszClientAddress);
	stUDPClientAddr.sin_port = htons(pstSocket->nPort);
	nAddrLen = sizeof(stUDPClientAddr);
	*pnRecvSize = recvfrom(pstSocket->nSocketfd, pBuffer, nBufferLen, 0, (struct sockaddr *) &stUDPClientAddr, &nAddrLen);
	if (*pnRecvSize < 0) {
		ERRASSIGNGOTO(result, ERR_UEM_NET_RECEIVE_ERROR, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}
