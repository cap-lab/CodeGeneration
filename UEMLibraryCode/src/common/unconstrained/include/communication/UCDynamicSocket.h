/*
 * UCDynamicSocket.h
 *
 *  Created on: 2015. 8. 19.
 *      Author: chjej202
 *      Changed :
 *  	    1. 2019. 06. 20. wecracy
 */

#ifndef SRC_COMMON_INCLUDE_COMMUNICATION_UCDYNAMICSOCKET_H_
#define SRC_COMMON_INCLUDE_COMMUNICATION_UCDYNAMICSOCKET_H_

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
	SOCKET_TYPE_UDP, //!< UDP/IP Socket
} ESocketType;

typedef struct _SSocketInfo
{
    ESocketType enSocketType; //!< Socket type.
    // Socket file path
    // (Ex. /tmp/unixsocket for @ref SOCKET_TYPE_UDS,
    //      127.0.0.1 for SOCKET_TYPE_TCP
    //      12:34:56:78:90:ab for SOCKET_TYPE_BLUETOOTH
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


/**
 * @brief Set an API list for socket-based communication.
 *
 * This function sets an API list for various socket-based communication. \n
 * Because socket is not always needed for all code generation, List of API to be set is \n
 * determined during code generation.
 *
 * @warning This function must be called first before using other UCDynamicSocket APIs.
 *
 * @param enSocketType Type of communication method to set an API List. See also @ref ESocketType.
 * @param pstSocketAPIList An API set to the corresponding socket type. See also @ref SSocketAPI.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM. \n
 *         @ref ERR_UEM_INVALID_PARAM can be occurred if the @a enSocketType is not valid or @a pstSocketAPIList is NULL.
 */
uem_result UCDynamicSocket_SetAPIList(ESocketType enSocketType, SSocketAPI *pstSocketAPIList);

/**
 * @brief Create a socket handle.
 *
 * This function creates a socket handle which allows to communicate with other processes/systems.
 * To use socket as a server, @a bIsServer needs to be TRUE. Otherwise, @a bIsServer needs to be FALSE.
 *
 * @param pstSocketInfo a socket setting information.
 * @param bIsServer a socket as a server or client.
 * @param[out] phSocket a socket handle to be created.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_OUT_OF_MEMORY, @ref ERR_UEM_NOT_SUPPORTED, \n
 *         @ref ERR_UEM_NOT_FOUND. \n
 *         @ref ERR_UEM_NOT_FOUND can be occurred if the socket API is not set yet.\n
 *         @ref ERR_UEM_INVALID_PARAM can be occurred if the parameters are invalid. \n
 *         @ref ERR_UEM_NOT_SUPPORTED if @a enSocketType of @a pstSocketInfo is invalid.
 */
uem_result UCDynamicSocket_Create(IN SSocketInfo *pstSocketInfo, IN uem_bool bIsServer, OUT HSocket *phSocket);

/**
 * @brief Destroy a socket handle.
 *
 * This function closes a socket and destroys a socket handle.
 *
 * @param[in,out] phSocket a socket handle to be destroyed.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_NOT_SUPPORTED, \n
 *         @ref ERR_UEM_NOT_FOUND. \n
 *         @ref ERR_UEM_INVALID_HANDLE can be occurred if the handle is not a socket handle.\n
 *         @ref ERR_UEM_NOT_FOUND can be occurred if the socket API is not set yet.\n
 *         @ref ERR_UEM_INVALID_PARAM can be occurred if @a phSocket is NULL \n
 *         @ref ERR_UEM_NOT_SUPPORTED if @a enSocketType of @a pstSocketInfo is invalid.
 */
uem_result UCDynamicSocket_Destroy(IN OUT HSocket *phSocket);

/**
 * @brief Bind a socket (server-only).
 *
 * This function binds a socket. This is a server-only function.
 *
 * @warning this function needs to be called before @ref UCDynamicSocket_Listen and @ref UCDynamicSocket_Accept.
 *
 * @param hServerSocket a socket handle to be binded.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_INVALID_SOCKET, \n
 *         and corresponding errors from @ref SSocketAPI fnBind() function. \n
 *         @ref ERR_UEM_INVALID_SOCKET can be occurred if the @a hServerSocket is a client socket. \n
 *         @ref ERR_UEM_INVALID_HANDLE can be occurred if the handle is not a socket handle.\n
 *         @ref ERR_UEM_INVALID_PARAM can be occurred if @a phSocket is NULL \n
 *
 * @sa UCBluetoothSocket_Bind, UCTCPSocket_Bind, UCUnixDomainSocket_Bind.
 */
uem_result UCDynamicSocket_Bind(HSocket hServerSocket);

/**
 * @brief Listen a socket (server-only).
 *
 * This function listens a socket. This is a server-only function. \n
 * After calling this function, a server can receive single or multiple requests with @ref UCDynamicSocket_Accept.
 *
 * @param hServerSocket a socket handle to listen.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INVALID_SOCKET, @ref ERR_UEM_LISTEN_ERROR. \n
 *         @ref ERR_UEM_INVALID_HANDLE can be occurred if the handle is not a socket handle.\n
 *         @ref ERR_UEM_INVALID_SOCKET can be occurred if the @a hServerSocket is a client socket. \n
 *         @ref ERR_UEM_LISTEN_ERROR can be occurred when listen operation is failed.
 */
uem_result UCDynamicSocket_Listen(HSocket hServerSocket);

/**
 * @brief Accept a client connection (server-only).
 *
 * This function accepts a client connection from different process/system. \n
 * To communicate with client, retrieved @a hSocket is used.  \n
 * To get new client connection, @a hSocket needs to be created as a client socket before. \n
 * SSocketInfo is not needed to be set for @a hSocket creation.
 *
 * @param hServerSocket a socket handle to accept client connection.
 * @param nTimeout a maximum time to wait for client connection.
 * @param[in,out] hSocket a retrieved client socket.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INVALID_SOCKET, @ref ERR_UEM_INVALID_PARAM, \n
 *         @ref ERR_UEM_SELECT_ERROR, @ref ERR_UEM_NET_TIMEOUT, and corresponding errors from @ref SSocketAPI fnAccept() function. \n
 *         @ref ERR_UEM_INVALID_HANDLE can be occurred if the handle is not a socket handle.\n
 *         @ref ERR_UEM_INVALID_SOCKET can be occurred if the @a hServerSocket is a client socket. \n
 *         @ref ERR_UEM_INVALID_PARAM can be occurred if the @a nTimeout is an invalid number. \n
 *         @ref ERR_UEM_SELECT_ERROR can be occurred when select operation is failed. \n
 *         @ref ERR_UEM_NET_TIMEOUT is occurred when the timeout is happened during select operation.
 *
 * @sa UCBluetoothSocket_Accept, UCTCPSocket_Accept, UCUnixDomainSocket_Accept.
 */
uem_result UCDynamicSocket_Accept(HSocket hServerSocket, IN int nTimeout, IN OUT HSocket hSocket);

/**
 * @brief Connect to a server (client-only).
 *
 * This function connects to a server.
 *
 * @param hClientSocket a socket handle.
 * @param nTimeout a maximum time to wait for connection.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned -  @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INVALID_SOCKET, @ref ERR_UEM_INVALID_PARAM, \n
 *         @ref ERR_UEM_CONNECT_ERROR, and corresponding errors from @ref SSocketAPI fnConnect() function. \n
 *         @ref ERR_UEM_INVALID_HANDLE can be occurred if the handle is not a socket handle.\n
 *         @ref ERR_UEM_INVALID_SOCKET can be occurred if the @a hClientSocket is a server socket. \n
 *         @ref ERR_UEM_INVALID_PARAM can be occurred if the @a nTimeout is an invalid number. \n
 *         @ref ERR_UEM_CONNECT_ERROR can be occurred when the server connection is failed.
 *
 * @sa UCBluetoothSocket_Connect, UCTCPSocket_Connect, UCUnixDomainSocket_Connect.
 *
 */
uem_result UCDynamicSocket_Connect(HSocket hClientSocket, IN int nTimeout);

/**
 * @brief Disconnect a socket from a server (client-only).
 *
 * This function disconnect a socket from a server.
 *
 * @param hClientSocket a socket handle
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_HANDLE. \n
 *         @ref ERR_UEM_INVALID_HANDLE can be occurred if the handle is not a socket handle.
 */
uem_result UCDynamicSocket_Disconnect(HSocket hClientSocket);

/**
 * @brief Send data (client-only).
 *
 * This function sends data.
 *
 * @param hSocket a socket handle.
 * @param nTimeout a maximum time to wait for sending data.
 * @param pData data to send.
 * @param nDataLen amount of data to send.
 * @param[out] pnSentSize amount of data sent.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INVALID_SOCKET, @ref ERR_UEM_INVALID_PARAM, \n
 *         @ref ERR_UEM_SELECT_ERROR, @ref ERR_UEM_NET_TIMEOUT, @ref ERR_UEM_NET_SEND_ERROR. \n
 *         @ref ERR_UEM_INVALID_HANDLE can be occurred if the handle is not a socket handle.\n
 *         @ref ERR_UEM_INVALID_SOCKET can be occurred if the @a hSocket is a server socket. \n
 *         @ref ERR_UEM_INVALID_PARAM can be occurred if the parameters are invalid. \n
 *         @ref ERR_UEM_SELECT_ERROR can be occurred when select operation is failed. \n
 *         @ref ERR_UEM_NET_TIMEOUT can be occurred the timeout is happened during select operation. \n
 *         @ref ERR_UEM_NET_SEND_ERROR can be occurred when send operation is failed.
 */
uem_result UCDynamicSocket_Send(HSocket hSocket, IN int nTimeout, IN char *pData, IN int nDataLen, OUT int *pnSentSize);

/**
 * @brief Receive data (client-only).
 *
 * This function receives data.
 *
 * @param hSocket a socket handle
 * @param nTimeout a maximum time to wait for receiving data.
 * @param pBuffer buffer to receive data.
 * @param nBufferLen size of buffer.
 * @param[out] pnReceivedSize amount of data received.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INVALID_SOCKET, @ref ERR_UEM_INVALID_PARAM, \n
 *         @ref ERR_UEM_SELECT_ERROR, @ref ERR_UEM_NET_TIMEOUT, @ref ERR_UEM_NET_RECEIVE_ERROR. \n
 *         @ref ERR_UEM_SELECT_ERROR, @ref ERR_UEM_NET_TIMEOUT, @ref ERR_UEM_NET_SEND_ERROR. \n
 *         @ref ERR_UEM_INVALID_HANDLE can be occurred if the handle is not a socket handle.\n
 *         @ref ERR_UEM_INVALID_SOCKET can be occurred if the @a hSocket is a server socket. \n
 *         @ref ERR_UEM_INVALID_PARAM can be occurred if the parameters are invalid. \n
 *         @ref ERR_UEM_SELECT_ERROR can be occurred when select operation is failed. \n
 *         @ref ERR_UEM_NET_TIMEOUT can be occurred the timeout is happened during select operation. \n
 *         @ref ERR_UEM_NET_RECEIVE_ERROR can be occurred when recv operation is failed.
 */
uem_result UCDynamicSocket_Receive(HSocket hSocket, IN int nTimeout, IN OUT char *pBuffer, IN int nBufferLen, OUT int *pnReceivedSize);

/**
 * @brief Send data to UDP.
 *
 * This function sends data to UDP.
 *
 * @param hSocket a socket handle.
 * @param unClientAddress address of receiver.
 * @param nTimeout a maximum time to wait for sending data.
 * @param pData data to send.
 * @param nDataLen amount of data to send.
 * @param[out] pnSentSize amount of data sent.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INVALID_SOCKET, @ref ERR_UEM_INVALID_PARAM, \n
 *         @ref ERR_UEM_SELECT_ERROR, @ref ERR_UEM_NET_TIMEOUT, @ref ERR_UEM_NET_SEND_ERROR. \n
 *         @ref ERR_UEM_INVALID_HANDLE can be occurred if the handle is not a socket handle.\n
 *         @ref ERR_UEM_INVALID_SOCKET can be occurred if the @a hSocket is a server socket. \n
 *         @ref ERR_UEM_INVALID_PARAM can be occurred if the parameters are invalid. \n
 *         @ref ERR_UEM_SELECT_ERROR can be occurred when select operation is failed. \n
 *         @ref ERR_UEM_NET_TIMEOUT can be occurred the timeout is happened during select operation. \n
 *         @ref ERR_UEM_NET_SEND_ERROR can be occurred when send operation is failed.
 */
uem_result UCDynamicSocket_Sendto(HSocket hSocket, IN const char *pszClientAddress, IN int nTimeout, IN unsigned char *pData, IN int nDataLen, OUT int *pnSentSize);

/**
 * @brief Receive data from UDP.
 *
 * This function receives data from UDP.
 *
 * @param hSocket a socket handle
 * @param pszClientAddress address of sender
 * @param nTimeout a maximum time to wait for receiving data.
 * @param pBuffer buffer to receive data.
 * @param nBufferLen size of buffer.
 * @param[out] pnReceivedSize amount of data received.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INVALID_SOCKET, @ref ERR_UEM_INVALID_PARAM, \n
 *         @ref ERR_UEM_SELECT_ERROR, @ref ERR_UEM_NET_TIMEOUT, @ref ERR_UEM_NET_RECEIVE_ERROR. \n
 *         @ref ERR_UEM_SELECT_ERROR, @ref ERR_UEM_NET_TIMEOUT, @ref ERR_UEM_NET_SEND_ERROR. \n
 *         @ref ERR_UEM_INVALID_HANDLE can be occurred if the handle is not a socket handle.\n
 *         @ref ERR_UEM_INVALID_SOCKET can be occurred if the @a hSocket is a server socket. \n
 *         @ref ERR_UEM_INVALID_PARAM can be occurred if the parameters are invalid. \n
 *         @ref ERR_UEM_SELECT_ERROR can be occurred when select operation is failed. \n
 *         @ref ERR_UEM_NET_TIMEOUT can be occurred the timeout is happened during select operation. \n
 *         @ref ERR_UEM_NET_RECEIVE_ERROR can be occurred when recv operation is failed.
 */
uem_result UCDynamicSocket_RecvFrom(HSocket hSocket, IN const char *pszClientAddress,  IN int nTimeout, IN int nBufferLen, OUT char *pBuffer, OUT int *pnRecvSize);

#ifdef __cplusplus
}
#endif

#endif /* SRC_COMMON_INCLUDE_COMMUNICATION_UCDYNAMICSOCKET_H_ */
