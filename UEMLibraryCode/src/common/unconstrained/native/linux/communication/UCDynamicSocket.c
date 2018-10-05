/*
 * UCSocket.c
 *
 *  Created on: 2015. 8. 19.
 *      Author: chjej202
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif


// this code is not run on Windows because I didn't call any WSAStartup or WSACleanup.
// ifdefs are used for removing compile errors on mingw32 build

#ifndef WIN32
#include <unistd.h>
#include <sys/socket.h>
#else
#include <winsock.h>
#endif

#include <UCBasic.h>
#include <UCAlloc.h>
#include <UCString.h>
#include <UCDynamicSocket.h>

typedef struct _SSocketAPIList {
    SSocketAPI *pstTCPAPI;
    SSocketAPI *pstUnixDomainSocketAPI;
    SSocketAPI *pstBluetoothAPI;
} SSocketAPIList;

static SSocketAPIList s_stSocketAPIs = {
    NULL,
    NULL,
    NULL,
};

#define SOCKET_FD_NOT_SET (-1)

static uem_result  getSocketAPIByType(ESocketType enSocketType, SSocketAPI **ppstSocketAPI)
{
    uem_result result = ERR_UEM_UNKNOWN;
    SSocketAPI *pstSocketAPI = NULL;

    switch(enSocketType)
    {
    case SOCKET_TYPE_TCP:
        pstSocketAPI = s_stSocketAPIs.pstTCPAPI;
        break;
    case SOCKET_TYPE_UDS:
        pstSocketAPI = s_stSocketAPIs.pstUnixDomainSocketAPI;
        break;
    case SOCKET_TYPE_BLUETOOTH:
        pstSocketAPI = s_stSocketAPIs.pstBluetoothAPI;
        break;
    default:
        ERRASSIGNGOTO(result, ERR_UEM_NOT_SUPPORTED, _EXIT);
        break;
    }

    if(pstSocketAPI == NULL)
    {
        ERRASSIGNGOTO(result, ERR_UEM_NOT_FOUND, _EXIT);
    }

    *ppstSocketAPI = pstSocketAPI;

    result = ERR_UEM_NOERROR;
_EXIT:
    return result;
}

static uem_result setSocketPath(SUCSocket *pstSocket, char *pszSocketPath)
{
    uem_result result = ERR_UEM_UNKNOWN;
    uem_string_struct stInputPath;

    result = UCString_New(&stInputPath, pszSocketPath, UEMSTRING_CONST);
    ERRIFGOTO(result, _EXIT);

    if(UCString_Length(&stInputPath) > 0)
    {
        pstSocket->pszSocketPath = (char *) UCAlloc_malloc(sizeof((UCString_Length(&stInputPath)+1) * sizeof(char)));
        ERRMEMGOTO(pstSocket->pszSocketPath, result, _EXIT);

        result = UCString_New(&(pstSocket->stSocketPath), pstSocket->pszSocketPath, (UCString_Length(&stInputPath)+1) * sizeof(char));
        ERRIFGOTO(result, _EXIT);

        result = UCString_Set(&(pstSocket->stSocketPath), &stInputPath);
        ERRIFGOTO(result, _EXIT);
    }

    result = ERR_UEM_NOERROR;
_EXIT:
    return result;
}


uem_result UCDynamicSocket_Create(IN SSocketInfo *pstSocketInfo, IN uem_bool bIsServer, OUT HSocket *phSocket)
{
    uem_result result = ERR_UEM_UNKNOWN;
    SUCSocket *pstSocket = NULL;
    SSocketAPI *pstSocketAPI = NULL;
#ifdef ARGUMENT_CHECK
    IFVARERRASSIGNGOTO(pstSocketInfo, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
    IFVARERRASSIGNGOTO(phSocket, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

    if(bIsServer != TRUE && bIsServer != FALSE)
    {
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
    }

    if(pstSocketInfo->enSocketType == SOCKET_TYPE_TCP && pstSocketInfo->nPort <= 0)
    {
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
    }

    if(bIsServer == TRUE && pstSocketInfo->enSocketType == SOCKET_TYPE_UDS && pstSocketInfo->pszSocketPath == NULL)
    {
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
    }

    if(pstSocketInfo->enSocketType == SOCKET_TYPE_BLUETOOTH && pstSocketInfo->pszSocketPath == NULL)
    {
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
    }
#endif
    result = getSocketAPIByType(pstSocketInfo->enSocketType, &pstSocketAPI);
    ERRIFGOTO(result, _EXIT);

    pstSocket = (SUCSocket *) UCAlloc_malloc(sizeof(SUCSocket));
    ERRMEMGOTO(pstSocket, result, _EXIT);

    pstSocket->enID = ID_UEM_SOCKET;
    pstSocket->bIsServer = bIsServer;
    pstSocket->nSocketfd = SOCKET_FD_NOT_SET;
    pstSocket->enSocketType = pstSocketInfo->enSocketType;
    pstSocket->nPort = pstSocketInfo->nPort;
    pstSocket->pszSocketPath = NULL;

    if(pstSocketInfo->pszSocketPath != NULL) // socket path is used
    {
        result = setSocketPath(pstSocket, pstSocketInfo->pszSocketPath);
        ERRIFGOTO(result, _EXIT);
    }

    if(pstSocketAPI->fnCreate != NULL)
    {
        result = pstSocketAPI->fnCreate((HSocket)pstSocket, pstSocketInfo, bIsServer);
        ERRIFGOTO(result, _EXIT);
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
    SSocketAPI *pstSocketAPI = NULL;
#ifdef ARGUMENT_CHECK
    IFVARERRASSIGNGOTO(phSocket, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

    if(IS_VALID_HANDLE(*phSocket, ID_UEM_SOCKET) == FALSE)
    {
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
    }
#endif
    pstSocket = (SUCSocket *) *phSocket;

    result = getSocketAPIByType(pstSocket->enSocketType, &pstSocketAPI);
    ERRIFGOTO(result, _EXIT);

    if(pstSocket->nSocketfd != SOCKET_FD_NOT_SET)
    {
        close(pstSocket->nSocketfd);
        if(pstSocketAPI->fnDestroy != NULL)
        {
        	pstSocketAPI->fnDestroy((HSocket) pstSocket);
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
    SSocketAPI *pstSocketAPI = NULL;
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

    result = getSocketAPIByType(pstSocket->enSocketType, &pstSocketAPI);
    ERRIFGOTO(result, _EXIT);

    if(pstSocketAPI->fnBind != NULL)
    {
    	result = pstSocketAPI->fnBind(hServerSocket);
    	ERRIFGOTO(result, _EXIT);
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
    int nMaxAllowConnections = 0;
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

    if(pstSocket->enSocketType == SOCKET_TYPE_BLUETOOTH)
    {
        ERRASSIGNGOTO(result, ERR_UEM_NOT_SUPPORTED, _EXIT);
    }
#endif
    switch(pstSocket->enSocketType)
    {
    case SOCKET_TYPE_BLUETOOTH:
    	nMaxAllowConnections = 1;
    	break;
    default:
    	nMaxAllowConnections = SOMAXCONN;
    	break;
    }
    nRet = listen(pstSocket->nSocketfd, nMaxAllowConnections);
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
    SSocketAPI *pstSocketAPI = NULL;
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

    if(pstSocket->enSocketType == SOCKET_TYPE_BLUETOOTH)
    {
        ERRASSIGNGOTO(result, ERR_UEM_NOT_SUPPORTED, _EXIT);
    }
#endif
    if(pstCliSocket->bIsServer == TRUE || pstCliSocket->nSocketfd > 0)
    {
        UEM_DEBUG_PRINT("fd : %d\n", pstCliSocket->nSocketfd);
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_SOCKET, _EXIT);
    }

    result = getSocketAPIByType(pstSocket->enSocketType, &pstSocketAPI);
    ERRIFGOTO(result, _EXIT);

    if(pstSocketAPI->fnAccept != NULL)
    {
        result = selectTimeout(pstSocket->nSocketfd, &stReadSet, NULL, NULL, nTimeout);
        ERRIFGOTO(result, _EXIT);

    	result = pstSocketAPI->fnAccept(hServerSocket, hSocket);
    	ERRIFGOTO(result, _EXIT);
    }

    result = ERR_UEM_NOERROR;
_EXIT:
    return result;
}


uem_result UCDynamicSocket_Connect(HSocket hClientSocket, IN int nTimeout)
{
    uem_result result = ERR_UEM_UNKNOWN;
    SUCSocket *pstSocket = NULL;
    SSocketAPI *pstSocketAPI = NULL;
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

    result = getSocketAPIByType(pstSocket->enSocketType, &pstSocketAPI);
    ERRIFGOTO(result, _EXIT);

    if(pstSocketAPI->fnConnect != NULL)
    {
    	result = pstSocketAPI->fnConnect(hClientSocket, nTimeout);
    	ERRIFGOTO(result, _EXIT);
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


uem_result UCDynamicSocket_SetAPIList(ESocketType enSocketType, SSocketAPI *pstSocketAPIList)
{
    uem_result result = ERR_UEM_UNKNOWN;
#ifdef ARGUMENT_CHECK
    IFVARERRASSIGNGOTO(pstSocketAPIList, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
    switch(enSocketType)
    {
    case SOCKET_TYPE_TCP:
        s_stSocketAPIs.pstTCPAPI = pstSocketAPIList;
        break;
    case SOCKET_TYPE_UDS:
        s_stSocketAPIs.pstUnixDomainSocketAPI = pstSocketAPIList;
        break;
    case SOCKET_TYPE_BLUETOOTH:
        s_stSocketAPIs.pstBluetoothAPI = pstSocketAPIList;
        break;
    default:
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
        break;
    }

    result = ERR_UEM_NOERROR;
_EXIT:
    return result;
}



