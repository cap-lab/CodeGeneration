/*
 * UCTCPSocket.c
 *
 *  Created on: 2018. 10. 2.
 *      Author: chjej202
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

#include <UCBasic.h>

#include <UCDynamicSocket.h>

uem_result UCTCPSocket_Bind(HSocket hSocket)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SUCSocket *pstSocket = NULL;
    struct sockaddr_in stTCPServerAddr;
    int nRet = 0;
    struct linger stLinger;

	pstSocket = (SUCSocket *) hSocket;

   pstSocket->nSocketfd = socket(AF_INET, SOCK_STREAM, 0);
   if(pstSocket->nSocketfd < 0)
   {
	   ERRASSIGNGOTO(result, ERR_UEM_SOCKET_ERROR, _EXIT);
   }

   stLinger.l_onoff = TRUE;
   stLinger.l_linger = 0;
   //nRet = setsockopt(pstSocket->nSocketfd, SOL_SOCKET, TCP_NODELAY, &stLinger, sizeof(stLinger));
   nRet = setsockopt(pstSocket->nSocketfd, SOL_SOCKET, SO_LINGER, (void *)&stLinger, sizeof(stLinger));
   if(nRet != 0)
   {
	   ERRASSIGNGOTO(result, ERR_UEM_SOCKET_ERROR, _EXIT);
   }

   UC_memset(&stTCPServerAddr, 0, sizeof(stTCPServerAddr));
   stTCPServerAddr.sin_family = AF_INET;
   stTCPServerAddr.sin_addr.s_addr = htonl(INADDR_ANY);
   stTCPServerAddr.sin_port = htons(pstSocket->nPort);

   nRet = bind(pstSocket->nSocketfd, (struct sockaddr *)&stTCPServerAddr, sizeof(stTCPServerAddr));
   if(nRet != 0)
   {
	   UEM_DEBUG_PRINT("bind errno: %d, %s\n", errno, strerror(errno));
	   ERRASSIGNGOTO(result, ERR_UEM_BIND_ERROR, _EXIT);
   }

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UCTCPSocket_Accept(HSocket hServerSocket, HSocket hClientSocket)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SUCSocket *pstServerSocket = NULL;
	SUCSocket *pstClientSocket = NULL;
#ifndef WIN32
    struct sockaddr_in stClientAddr;
    socklen_t nLen = 0;
#else
    struct sockaddr stClientAddr;
    int nLen = 0;
#endif
	pstServerSocket = (SUCSocket *) hServerSocket;
	pstClientSocket = (SUCSocket *) hClientSocket;

	pstClientSocket->nSocketfd = accept(pstServerSocket->nSocketfd, (struct sockaddr *) &stClientAddr, &nLen);
    if(pstClientSocket->nSocketfd < 0)
    {
        ERRASSIGNGOTO(result, ERR_UEM_ACCEPT_ERROR, _EXIT);
    }

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

// TODO: nTimeout must be implemented
uem_result UCTCPSocket_Connect(HSocket hSocket, IN int nTimeout)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SUCSocket *pstSocket = NULL;
    struct sockaddr_in stTCPClientAddr;
    int nRet = 0;
    struct linger stLinger;

	pstSocket = (SUCSocket *) hSocket;

    pstSocket->nSocketfd = socket(AF_INET, SOCK_STREAM, 0);
    if(pstSocket->nSocketfd < 0)
    {
        ERRASSIGNGOTO(result, ERR_UEM_SOCKET_ERROR, _EXIT);
    }

    stLinger.l_onoff = TRUE;
    stLinger.l_linger = 0;
    nRet = setsockopt(pstSocket->nSocketfd, SOL_SOCKET, SO_LINGER, (void *)&stLinger, sizeof(stLinger));
    if(nRet != 0)
    {
        ERRASSIGNGOTO(result, ERR_UEM_SOCKET_ERROR, _EXIT);
    }

    UC_memset(&stTCPClientAddr, 0, sizeof(stTCPClientAddr));
    stTCPClientAddr.sin_family = AF_INET;
    stTCPClientAddr.sin_addr.s_addr = inet_addr(pstSocket->pszSocketPath);
    stTCPClientAddr.sin_port = htons(pstSocket->nPort);

    nRet = connect(pstSocket->nSocketfd, (struct sockaddr *)&stTCPClientAddr, sizeof(stTCPClientAddr));
    if(nRet != 0)
    {
        ERRASSIGNGOTO(result, ERR_UEM_CONNECT_ERROR, _EXIT);
    }

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


