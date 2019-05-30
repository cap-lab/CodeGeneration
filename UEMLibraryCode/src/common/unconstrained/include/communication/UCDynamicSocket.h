/*
 * UCSocket.h
 *
 *  Created on: 2015. 8. 19.
 *      Author: chjej202
 */

#ifndef SRC_COMMON_INCLUDE_UCDYNAMICSOCKET_H_
#define SRC_COMMON_INCLUDE_UCDYNAMICSOCKET_H_

#include <uem_common.h>

#include <UCString.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef enum _ESocketType
{
    SOCKET_TYPE_UDS, //!< Unix domain socket.
    SOCKET_TYPE_TCP, //!< TCP/IP Socket
	SOCKET_TYPE_BLUETOOTH, //!< Bluetooth communication Socket
} ESocketType;

typedef struct _SSocketInfo
{
    ESocketType enSocketType; //!< Socket type.
    // Socket file path
    // (Ex. /tmp/unixsocket for @ref SOCKET_TYPE_UDS,
    //      127.0.0.1 for SOCKET_TYPE_TCP
    //      12:34:56:78:90:ab for SOCKET_TYPE_BLUETOOTH
    //      /dev/ttyUSB0 for SOCKET_TYPE_SERIAL).
    char *pszSocketPath;
    int nPort; //!< Port number used by SOCKET_TYPE_TCP.
}SSocketInfo;

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

typedef struct _SUCSocket *HSocket;

typedef uem_result (*FnSocketBind)(HSocket hSocket);
typedef uem_result (*FnSocketAccept)(HSocket hServerSocket, HSocket hClientSocket);
typedef uem_result (*FnSocketConnect)(HSocket hSocket, IN int nTimeout);
typedef uem_result (*FnSocketCreate)(HSocket hSocket, SSocketInfo *pstSocketInfo, uem_bool bIsServer);
typedef uem_result (*FnSocketDestroy)(HSocket hSocket);

typedef struct _SSocketAPI {
	FnSocketBind fnBind;
	FnSocketAccept fnAccept;
	FnSocketConnect fnConnect;
	FnSocketCreate fnCreate;
	FnSocketDestroy fnDestroy;
} SSocketAPI;


uem_result UCDynamicSocket_SetAPIList(ESocketType enSocketType, SSocketAPI *pstSocketAPIList);
uem_result UCDynamicSocket_Create(IN SSocketInfo *pstSocketInfo, IN uem_bool bIsServer, OUT HSocket *phSocket);
uem_result UCDynamicSocket_Destroy(IN OUT HSocket *phSocket);
uem_result UCDynamicSocket_Bind(HSocket hServerSocket);
uem_result UCDynamicSocket_Listen(HSocket hServerSocket);
uem_result UCDynamicSocket_Accept(HSocket hServerSocket, IN int nTimeout, IN OUT HSocket hSocket);
uem_result UCDynamicSocket_Connect(HSocket hClientSocket, IN int nTimeout);
uem_result UCDynamicSocket_Disconnect(HSocket hClientSocket);
uem_result UCDynamicSocket_Send(HSocket hSocket, IN int nTimeout, IN char *pData, IN int nDataLen, OUT int *pnSentSize);
uem_result UCDynamicSocket_Receive(HSocket hSocket, IN int nTimeout, IN OUT char *pBuffer, IN int nBufferLen, OUT int *pnReceivedSize);

#ifdef __cplusplus
}
#endif

#endif /* SRC_COMMON_INCLUDE_UCDYNAMICSOCKET_H_ */
