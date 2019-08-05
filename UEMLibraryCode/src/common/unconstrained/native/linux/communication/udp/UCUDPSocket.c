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

uem_result UCUDPSocket_Create(HSocket hSocket, SSocketInfo *pstSocketInfo, uem_bool bIsServer)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SUCSocket *pstSocket = NULL;
	int nRet;
	int nOptVal;

	pstSocket = (SUCSocket *) hSocket;

	pstSocket->nSocketfd = socket(AF_INET, SOCK_DGRAM, 0);
	if(pstSocket->nSocketfd < 0)
	{
	    ERRASSIGNGOTO(result, ERR_UEM_SOCKET_ERROR, _EXIT);
	}

	nOptVal = TRUE;
	nRet = setsockopt(pstSocket->nSocketfd, SOL_SOCKET, SO_BROADCAST, &nOptVal, (socklen_t) sizeof(nOptVal));
	if(nRet != 0)
	{
	   ERRASSIGNGOTO(result, ERR_UEM_SOCKET_ERROR, _EXIT);
	}

	nRet = setsockopt(pstSocket->nSocketfd, SOL_SOCKET, SO_REUSEPORT, &nOptVal, (socklen_t) sizeof(nOptVal));
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
