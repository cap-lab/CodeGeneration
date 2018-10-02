/*
 * UCUnixDomainSocket.c
 *
 *  Created on: 2018. 10. 2.
 *      Author: chjej202
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <sys/un.h>
#include <sys/socket.h>

#include <uem_common.h>

#include <UCDynamicSocket.h>

#define MAX_SUN_PATH (107)

#define MAX_SOCKET_PATH (MAX_SUN_PATH+1)

uem_result UCUnixDomainSocket_Bind(HSocket hSocket)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SUCSocket *pstSocket = NULL;
	struct sockaddr_un stServerAddr;
	struct linger stLinger;
	int nLen = 0;
	 int nRet = 0;

	pstSocket = (SUCSocket *) hSocket;

    pstSocket->nSocketfd = socket(AF_UNIX, SOCK_STREAM, 0);
    if(pstSocket->nSocketfd < 0)
    {
        ERRASSIGNGOTO(result, ERR_UEM_SOCKET_ERROR, _EXIT);
    }

    stLinger.l_onoff = TRUE;
    stLinger.l_linger = 0;
    nRet = setsockopt(pstSocket->nSocketfd, SOL_SOCKET, SO_LINGER, &stLinger, sizeof(stLinger));
    if(nRet != 0)
    {
        ERRASSIGNGOTO(result, ERR_UEM_SOCKET_ERROR, _EXIT);
    }

    UC_memset(&stServerAddr, 0, sizeof(stServerAddr));
    stServerAddr.sun_family = AF_UNIX;
    nLen = MIN(MAX_SOCKET_PATH-1, UCString_Length(&(pstSocket->stSocketPath)));

    UC_memcpy(stServerAddr.sun_path, pstSocket->pszSocketPath, nLen);
    stServerAddr.sun_path[nLen] = '\0';

    nRet = bind(pstSocket->nSocketfd, (struct sockaddr *)&stServerAddr, sizeof(stServerAddr));
    if(nRet != 0)
    {
    	UEM_DEBUG_PRINT("bind errno: %d, %s\n", errno, strerror(errno));
    	ERRASSIGNGOTO(result, ERR_UEM_BIND_ERROR, _EXIT);
    }

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UCUnixDomainSocket_Connect(HSocket hSocket, IN int nTimeout)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SUCSocket *pstSocket = NULL;
    struct sockaddr_un stClientAddr;
    int nLen = 0;
    int nRet = 0;
    fd_set stReadSet;
    struct linger stLinger;

	pstSocket = (SUCSocket *) hSocket;

    pstSocket->nSocketfd = socket(AF_UNIX, SOCK_STREAM, 0);
    if(pstSocket->nSocketfd < 0)
     {
        ERRASSIGNGOTO(result, ERR_UEM_SOCKET_ERROR, _EXIT);
     }

    stLinger.l_onoff = TRUE;
    stLinger.l_linger = 0;
    nRet = setsockopt(pstSocket->nSocketfd, SOL_SOCKET, SO_LINGER, &stLinger, sizeof(stLinger));
    if(nRet != 0)
    {
        ERRASSIGNGOTO(result, ERR_UEM_SOCKET_ERROR, _EXIT);
    }

    UC_memset(&stClientAddr, 0, sizeof(stClientAddr));
    stClientAddr.sun_family = AF_UNIX;
    nLen = MIN(MAX_SUN_PATH-1, UCString_Length(&(pstSocket->stSocketPath)));

    UC_memcpy(stClientAddr.sun_path, pstSocket->pszSocketPath, nLen);
    stClientAddr.sun_path[nLen] = '\0';

    // TODO: please check this
    result = selectTimeout(pstSocket->nSocketfd, &stReadSet, NULL, NULL, nTimeout);
    ERRIFGOTO(result, _EXIT);

    nRet = connect(pstSocket->nSocketfd, (struct sockaddr *)&stClientAddr, sizeof(stClientAddr));
    if(nRet != 0)
    {
        ERRASSIGNGOTO(result, ERR_UEM_CONNECT_ERROR, _EXIT);
    }

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

// do nothing
uem_result UCUnixDomainSocket_Create(HSocket hSocket, SSocketInfo *pstSocketInfo, uem_bool bIsServer)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SUCSocket *pstSocket = NULL;

	pstSocket = (SUCSocket *) hSocket;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UCUnixDomainSocket_Destroy(HSocket hSocket)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SUCSocket *pstSocket = NULL;

	pstSocket = (SUCSocket *) hSocket;

	if(pstSocket->pszSocketPath != NULL && pstSocket->bIsServer == TRUE)
	{
		unlink(pstSocket->pszSocketPath);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

