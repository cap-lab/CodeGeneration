/*
 * UCBluetoothSocket.c
 *
 *  Created on: 2018. 10. 2.
 *      Author: chjej202
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <fcntl.h>
#include <errno.h>
#include <sys/socket.h>

#include <bluetooth/bluetooth.h>
#include <bluetooth/rfcomm.h>

#include <uem_common.h>

#include <UCBasic.h>

#include <UCDynamicSocket.h>

uem_result UCBluetoothSocket_Bind(HSocket hSocket)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SUCSocket *pstSocket = NULL;
    struct sockaddr_rc stBluetoothServerAddr;
    int nRet = 0;
    struct linger stLinger;

	pstSocket = (SUCSocket *) hSocket;

   pstSocket->nSocketfd = socket(AF_BLUETOOTH, SOCK_STREAM, BTPROTO_RFCOMM);
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

   UC_memset(&stBluetoothServerAddr, 0, sizeof(stBluetoothServerAddr));
   stBluetoothServerAddr.rc_family = AF_BLUETOOTH;
   stBluetoothServerAddr.rc_bdaddr = *BDADDR_ANY;

   // TODO: check channel need to be set or dynamically bind
   stBluetoothServerAddr.rc_channel = 1;

   nRet = bind(pstSocket->nSocketfd, (struct sockaddr *)&stBluetoothServerAddr, sizeof(stBluetoothServerAddr));
   if(nRet != 0)
   {
	   UEM_DEBUG_PRINT("bind errno: %d, %s\n", errno, strerror(errno));
	   ERRASSIGNGOTO(result, ERR_UEM_BIND_ERROR, _EXIT);
   }

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UCBluetoothSocket_Accept(HSocket hServerSocket, HSocket hClientSocket)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SUCSocket *pstServerSocket = NULL;
	SUCSocket *pstClientSocket = NULL;
    struct sockaddr_rc stClientAddr;
    socklen_t nLen = 0;

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
uem_result UCBluetoothSocket_Connect(HSocket hSocket, IN int nTimeout)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SUCSocket *pstSocket = NULL;
    struct sockaddr_rc stClientAddr;
    int nRet = 0;
    struct linger stLinger;

	pstSocket = (SUCSocket *) hSocket;

    pstSocket->nSocketfd = socket(AF_BLUETOOTH, SOCK_STREAM, BTPROTO_RFCOMM);
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
    stClientAddr.rc_family = AF_BLUETOOTH;
    stClientAddr.rc_channel = 1;

    nRet = str2ba( pstSocket->pszSocketPath, &stClientAddr.rc_bdaddr );
    if(nRet != 0)
    {
    	ERRASSIGNGOTO(result, ERR_UEM_CONVERSION_ERROR, _EXIT);
    }

    nRet = fcntl(pstSocket->nSocketfd, F_SETFL, O_NONBLOCK);
    if(nRet != 0)
    {
    	ERRASSIGNGOTO(result, ERR_UEM_SOCKET_ERROR, _EXIT);
    }

    nRet = connect(pstSocket->nSocketfd, (struct sockaddr *)&stClientAddr, sizeof(stClientAddr));
    if(nRet != 0)
    {
    	if(errno == EINPROGRESS)
    	{
    		UEMASSIGNGOTO(result, ERR_UEM_IN_PROGRESS, _EXIT);
    	}
    	ERRASSIGNGOTO(result, ERR_UEM_CONNECT_ERROR, _EXIT);
    }

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


