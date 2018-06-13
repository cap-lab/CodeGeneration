/*
 * UCSocket.c
 *
 *  Created on: 2015. 8. 19.
 *      Author: chjej202
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <sys/types.h>
#include <sys/stat.h>

// this code is not run on Windows because I didn't call any WSAStartup or WSACleanup.
// ifdefs are used for removing compile errors on mingw32 build

#ifndef WIN32
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#else
#include <winsock.h>
#endif


#include <stdlib.h>
#include <errno.h>
#include <string.h>

#include <UCBasic.h>
#include <UCString.h>
#include <UCDynamicSocket.h>

typedef struct _SUCSocket
{
	EUemModuleId enID;
    int nSocketfd;
    ESocketType enSocketType;
    uem_string_struct stSocketPath;
    char *pszSocketPath;
    int nPort;
    uem_bool bIsServer;
} SUCSocket;

#define SOCKET_FD_NOT_SET (-1)
#define MAX_SUN_PATH (107)

#define MAX_SOCKET_PATH (MAX_SUN_PATH+1)

uem_result UCDynamicSocket_Create(IN SSocketInfo *pstSocketInfo, IN uem_bool bIsServer, OUT HSocket *phSocket)
{
    uem_result result = ERR_UEM_UNKNOWN;
    SUCSocket *pstSocket = NULL;
    uem_string_struct stInputPath;
#ifdef ARGUMENT_CHECK
    IFVARERRASSIGNGOTO(pstSocketInfo, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
    IFVARERRASSIGNGOTO(phSocket, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

    if(bIsServer != TRUE && bIsServer != FALSE)
    {
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
    }

    if(pstSocketInfo->enSocketType != SOCKET_TYPE_UDS && pstSocketInfo->enSocketType != SOCKET_TYPE_TCP)
    {
        ERRASSIGNGOTO(result, ERR_UEM_NOT_SUPPORTED, _EXIT);
    }

    if(pstSocketInfo->enSocketType == SOCKET_TYPE_TCP && pstSocketInfo->nPort <= 0)
    {
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
    }

    if((bIsServer == FALSE && pstSocket->pszSocketPath == NULL) ||
    	(bIsServer == TRUE && pstSocketInfo->enSocketType == SOCKET_TYPE_UDS && pstSocketInfo->pszSocketPath == NULL))
    {
    	ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
    }
#endif
    pstSocket = (SUCSocket *) UC_malloc(sizeof(SUCSocket));
    ERRMEMGOTO(pstSocket, result, _EXIT);

    pstSocket->enID = ID_UEM_SOCKET;
    pstSocket->bIsServer = bIsServer;
    pstSocket->nSocketfd = SOCKET_FD_NOT_SET;
    pstSocket->enSocketType = pstSocketInfo->enSocketType;
    pstSocket->nPort = pstSocketInfo->nPort;
    pstSocket->pszSocketPath = NULL;

    if(pstSocketInfo->pszSocketPath != NULL) // socket path is used
    {
        result = UCString_New(&stInputPath, pstSocketInfo->pszSocketPath, UEMSTRING_CONST);
        ERRIFGOTO(result, _EXIT);

        if(UCString_Length(&stInputPath) > 0)
        {
        	pstSocket->pszSocketPath = (char *) UC_malloc(sizeof((UCString_Length(&stInputPath)+1) * sizeof(char)));
            ERRMEMGOTO(pstSocket->pszSocketPath, result, _EXIT);

            result = UCString_New(&(pstSocket->stSocketPath), pstSocket->pszSocketPath, (UCString_Length(&stInputPath)+1) * sizeof(char));
            ERRIFGOTO(result, _EXIT);

            result = UCString_Set(&(pstSocket->stSocketPath), &stInputPath);
            ERRIFGOTO(result, _EXIT);
        }
    }

    *phSocket = (HSocket) pstSocket;

    result = ERR_UEM_NOERROR;
_EXIT:
    if(result != ERR_UEM_NOERROR && pstSocket != NULL)
    {
        UCDynamicSocket_Destroy((HSocket *)&pstSocket);
    }
    return result;
}

uem_result UCDynamicSocket_Destroy(IN OUT HSocket *phSocket)
{
    uem_result result = ERR_UEM_UNKNOWN;
    SUCSocket *pstSocket = NULL;
#ifdef ARGUMENT_CHECK
    IFVARERRASSIGNGOTO(phSocket, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

    if(IS_VALID_HANDLE(*phSocket, ID_UEM_SOCKET) == FALSE)
    {
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
    }
#endif
    pstSocket = (SUCSocket *) *phSocket;

    if(pstSocket->nSocketfd != SOCKET_FD_NOT_SET)
    {
        close(pstSocket->nSocketfd);
		if(pstSocket->bIsServer == TRUE && pstSocket->pszSocketPath != NULL &&
			pstSocket->enSocketType == SOCKET_TYPE_UDS)
		{
            unlink(pstSocket->pszSocketPath);
		}
    }

    SAFEMEMFREE(pstSocket->pszSocketPath);

    SAFEMEMFREE(pstSocket);

    *phSocket = NULL;

    result = ERR_UEM_NOERROR;
_EXIT:
    return result;
}


/*
static uem_result convertSocketErrorToCAPError(int nErrno)
{
    uem_result result = ERR_UEM_UNKNOWN;

    switch(nErrno)
    {
    case EADDRINUSE:
        result = ERR_UEM_BIND_ERROR;
        break;
    default:
        result = ERR_UEM_NET_ERROR;
        break;
    }

    return result;
}
*/


uem_result UCDynamicSocket_Bind(HSocket hServerSocket)
{
    uem_result result = ERR_UEM_UNKNOWN;
    SUCSocket *pstSocket = NULL;
#ifndef WIN32
    struct sockaddr_un stServerAddr;
    int nLen = 0;
#endif
    struct sockaddr_in stTCPServerAddr;
    int nRet = 0;
    struct linger stLinger;
#ifdef ARGUMENT_CHECK
    if(IS_VALID_HANDLE(hServerSocket, ID_UEM_SOCKET) == FALSE)
    {
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
    }
#endif
    pstSocket = (SUCSocket *) hServerSocket;
#ifdef ARGUMENT_CHECK
    if(pstSocket->bIsServer == FALSE)
    {
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_SOCKET, _EXIT);
    }
#endif
    switch(pstSocket->enSocketType)
    {
#ifndef WIN32
    case SOCKET_TYPE_UDS:
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
        nLen = MIN(MAX_SUN_PATH-1, UCString_Length(&(pstSocket->stSocketPath)));

        UC_memcpy(stServerAddr.sun_path, pstSocket->pszSocketPath, nLen);
        stServerAddr.sun_path[nLen] = '\0';

        nRet = bind(pstSocket->nSocketfd, (struct sockaddr *)&stServerAddr, sizeof(stServerAddr));
        if(nRet != 0)
        {
        	UEM_DEBUG_PRINT("bind errno: %d, %s\n", errno, strerror(errno));
            ERRASSIGNGOTO(result, ERR_UEM_BIND_ERROR, _EXIT);
        }

       break;
#endif
   case SOCKET_TYPE_TCP:
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
            ERRASSIGNGOTO(result, ERR_UEM_CONNECT_ERROR, _EXIT);
         }
        break;
    default:
        ERRASSIGNGOTO(result, ERR_UEM_INTERNAL_FAIL, _EXIT);
        break;
    }

    result = ERR_UEM_NOERROR;
_EXIT:
    return result;
}


uem_result UCDynamicSocket_Listen(HSocket hServerSocket)
{
    uem_result result = ERR_UEM_UNKNOWN;
    SUCSocket *pstSocket = NULL;
    int nRet = 0;
#ifdef ARGUMENT_CHECK
    if(IS_VALID_HANDLE(hServerSocket, ID_UEM_SOCKET) == FALSE)
    {
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
    }
#endif
    pstSocket = (SUCSocket *) hServerSocket;
#ifdef ARGUMENT_CHECK
    if(pstSocket->bIsServer == FALSE)
    {
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_SOCKET, _EXIT);
    }
#endif
    nRet = listen(pstSocket->nSocketfd, SOMAXCONN);
    if(nRet != 0)
    {
        ERRASSIGNGOTO(result, ERR_UEM_LISTEN_ERROR, _EXIT);
    }

    result = ERR_UEM_NOERROR;
_EXIT:
    return result;
}

static uem_result selectTimeout(int nSocketfd, fd_set *pstReadSet, fd_set *pstWriteSet, fd_set *pstExceptSet, int nTimeout)
{
    uem_result result = ERR_UEM_UNKNOWN;
    struct timeval stTimeVal;
    int nRet = 0;

    if(pstReadSet != NULL)
    {
        FD_ZERO(pstReadSet);
        FD_SET(nSocketfd, pstReadSet);
    }

    if(pstWriteSet != NULL)
    {
        FD_ZERO(pstWriteSet);
        FD_SET(nSocketfd, pstWriteSet);
    }

    if(pstExceptSet != NULL)
   {
       FD_ZERO(pstExceptSet);
       FD_SET(nSocketfd, pstExceptSet);
   }

    stTimeVal.tv_sec = nTimeout;
    stTimeVal.tv_usec = 0;

    nRet = select(nSocketfd+1, pstReadSet, pstWriteSet, pstExceptSet, &stTimeVal);
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

uem_result UCDynamicSocket_Accept(HSocket hServerSocket, IN int nTimeout, IN OUT HSocket hSocket)
{
    uem_result result = ERR_UEM_UNKNOWN;
    SUCSocket *pstSocket = NULL;
    fd_set stReadSet;
    SUCSocket *pstCliSocket = NULL;
#ifndef WIN32
    struct sockaddr_un stClientAddr;
    socklen_t nLen = 0;
#else
    struct sockaddr stClientAddr;
    int nLen = 0;
#endif
#ifdef ARGUMENT_CHECK
    if(IS_VALID_HANDLE(hServerSocket, ID_UEM_SOCKET) == FALSE)
    {
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
    }

    if(IS_VALID_HANDLE(hSocket, ID_UEM_SOCKET) == FALSE)
    {
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
    }

    if(nTimeout <= 0)
    {
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
    }
#endif
    pstSocket = (SUCSocket *) hServerSocket;
    pstCliSocket = (SUCSocket *) hSocket;
#ifdef ARGUMENT_CHECK
    if(pstSocket->bIsServer == FALSE)
    {
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_SOCKET, _EXIT);
    }
#endif
    if(pstCliSocket->bIsServer == TRUE || pstCliSocket->nSocketfd > 0)
    {
        UEM_DEBUG_PRINT("fd : %d\n", pstCliSocket->nSocketfd);
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_SOCKET, _EXIT);
    }

    result = selectTimeout(pstSocket->nSocketfd, &stReadSet, NULL, NULL, nTimeout);
    ERRIFGOTO(result, _EXIT);

    pstCliSocket->nSocketfd = accept(pstSocket->nSocketfd, (struct sockaddr *) &stClientAddr, &nLen);
    if(pstCliSocket->nSocketfd < 0)
    {
        ERRASSIGNGOTO(result, ERR_UEM_ACCEPT_ERROR, _EXIT);
    }

    result = ERR_UEM_NOERROR;
_EXIT:
    return result;
}


uem_result UCDynamicSocket_Connect(HSocket hClientSocket, IN int nTimeout)
{
    uem_result result = ERR_UEM_UNKNOWN;
    SUCSocket *pstSocket = NULL;
#ifndef WIN32
    struct sockaddr_un stClientAddr;
    int nLen = 0;
#endif
    struct sockaddr_in stTCPClientAddr;

    int nRet = 0;
    fd_set stReadSet;
    struct linger stLinger;
#ifdef ARGUMENT_CHECK
    if(IS_VALID_HANDLE(hClientSocket, ID_UEM_SOCKET) == FALSE)
    {
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
    }

    if(nTimeout <= 0)
    {
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
    }
#endif
    pstSocket = (SUCSocket *) hClientSocket;
#ifdef ARGUMENT_CHECK
    if(pstSocket->bIsServer == TRUE)
    {
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_SOCKET, _EXIT);
    }
#endif
    switch(pstSocket->enSocketType)
    {
#ifndef WIN32
    case SOCKET_TYPE_UDS:
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

        result = selectTimeout(pstSocket->nSocketfd, &stReadSet, NULL, NULL, nTimeout);
        ERRIFGOTO(result, _EXIT);

        nRet = connect(pstSocket->nSocketfd, (struct sockaddr *)&stClientAddr, sizeof(stClientAddr));
        if(nRet != 0)
        {
            ERRASSIGNGOTO(result, ERR_UEM_CONNECT_ERROR, _EXIT);
        }

       break;
#endif
   case SOCKET_TYPE_TCP:
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

        result = selectTimeout(pstSocket->nSocketfd, &stReadSet, NULL, NULL, nTimeout);
        ERRIFGOTO(result, _EXIT);

        nRet = connect(pstSocket->nSocketfd, (struct sockaddr *)&stTCPClientAddr, sizeof(stTCPClientAddr));
        if(nRet != 0)
        {
            ERRASSIGNGOTO(result, ERR_UEM_CONNECT_ERROR, _EXIT);
        }

       break;

    default:
        ERRASSIGNGOTO(result, ERR_UEM_INTERNAL_FAIL, _EXIT);
        break;
    }

    result = ERR_UEM_NOERROR;
_EXIT:
    return result;
}



uem_result UCDynamicSocket_Send(HSocket hSocket, IN int nTimeout, IN char *pData, IN int nDataLen, OUT int *pnSentSize)
{
    uem_result result = ERR_UEM_UNKNOWN;
    SUCSocket *pstSocket = NULL;
    fd_set stWriteSet;
    int nDataSent = 0;
#ifdef ARGUMENT_CHECK
    IFVARERRASSIGNGOTO(pData, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

    if(IS_VALID_HANDLE(hSocket, ID_UEM_SOCKET) == FALSE)
    {
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
    }

    if(nTimeout <= 0 || nDataLen <= 0)
    {
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
    }
#endif
    pstSocket = (SUCSocket *) hSocket;
#ifdef ARGUMENT_CHECK
    if(pstSocket->bIsServer == TRUE)
    {
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_SOCKET, _EXIT);
    }
#endif
    result = selectTimeout(pstSocket->nSocketfd, NULL, &stWriteSet, NULL, nTimeout);
    ERRIFGOTO(result, _EXIT);

    nDataSent = send(pstSocket->nSocketfd, pData, nDataLen, 0);
    if(nDataSent < 0)
    {
        ERRASSIGNGOTO(result, ERR_UEM_NET_SEND_ERROR, _EXIT);
    }

    if(pnSentSize != NULL)
    {
        *pnSentSize = nDataSent;
    }

    result = ERR_UEM_NOERROR;
_EXIT:
    return result;
}


uem_result UCDynamicSocket_Receive(HSocket hSocket, IN int nTimeout, IN OUT char *pBuffer, IN int nBufferLen, OUT int *pnReceivedSize)
{
    uem_result result = ERR_UEM_UNKNOWN;
    SUCSocket *pstSocket = NULL;
    fd_set stReadSet;
    int nDataReceived = 0;
#ifdef ARGUMENT_CHECK
    IFVARERRASSIGNGOTO(pBuffer, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

    if(IS_VALID_HANDLE(hSocket, ID_UEM_SOCKET) == FALSE)
    {
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
    }

    if(nTimeout <= 0 || nBufferLen <= 0)
    {
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
    }
#endif
    pstSocket = (SUCSocket *) hSocket;
#ifdef ARGUMENT_CHECK
    if(pstSocket->bIsServer == TRUE)
    {
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_SOCKET, _EXIT);
    }
#endif
    result = selectTimeout(pstSocket->nSocketfd, &stReadSet, NULL, NULL, nTimeout);
    ERRIFGOTO(result, _EXIT);

    // MSG_DONTWAIT
    nDataReceived = recv(pstSocket->nSocketfd, pBuffer, nBufferLen, 0);
    if(nDataReceived <= 0)
    {
        ERRASSIGNGOTO(result, ERR_UEM_NET_RECEIVE_ERROR, _EXIT);
    }

    if(pnReceivedSize != NULL)
    {
        *pnReceivedSize = nDataReceived;
    }

    result = ERR_UEM_NOERROR;
_EXIT:
    return result;
}

