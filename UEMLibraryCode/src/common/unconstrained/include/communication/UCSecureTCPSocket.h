/*
 * UCSecureTCPSocket.h
 *
 *  Created on: 2020. 5. 21.
 *      Author: jrkim
 */

#ifndef SRC_COMMON_UNCONSTRAINED_INCLUDE_COMMUNICATION_UCSECURETCPSOCKET_H_
#define SRC_COMMON_UNCONSTRAINED_INCLUDE_COMMUNICATION_UCSECURETCPSOCKET_H_

#include <uem_common.h>

#include <UCTCPSocket.h>
#include <UCDynamicSocket.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct _SSecurityKeyInfo 
{
	char *pszCAPublicKey;
	char *pszPublicKey;
	char *pszPrivateKey;
} SSecurityKeyInfo;

typedef struct _SSecureSocketInfo 
{
	SSocketInfo stSocketInfo;
	SSecurityKeyInfo *pstKeyInfo;
} SSecureSocketInfo;

typedef struct _SUCSecureSocket *HSSLSocket;

/**
 * @brief Initialize a SSL socket handle.
 *
 * This function initialize the preprocess step to use SSL
 * To use a SSL, calling the SSL initialization functions is needed
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 */
uem_result UCSecureTCPSocket_Initialize();

/**
 * @brief Create a SSL socket handle.
 *
 * This function creates a SSL socket handle which allows to communicate with other processes/systems.
 * To use socket as a server, @a bIsServer needs to be TRUE. Otherwise, @a bIsServer needs to be FALSE.
 * To use a SSL, this function loads the key info, and initialize it.
 *
 * @param pstSocketInfo a SSL socket setting information.
 * @param bIsServer a socket as a server or client.
 * @param[out] phSocket a socket handle to be created.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_OUT_OF_MEMORY, @ref ERR_UEM_NOT_SUPPORTED, @ref ERR_UEM_SSL_ERROR, @ref ERR_UEM_SSL_KEY_NOT_FOUND, @ref ERR_UEM_SSL_KEY_INVALID \n
 *         @ref ERR_UEM_NOT_FOUND. \n
 *         @ref ERR_UEM_NOT_FOUND can be occurred if the socket API is not set yet.\n
 *         @ref ERR_UEM_INVALID_PARAM can be occurred if the parameters are invalid. \n
 *         @ref ERR_UEM_NOT_SUPPORTED if @a enSocketType of @a pstSocketInfo is invalid. \n
 *         @ref ERR_UEM_SSL_ERROR if SSL is not usable in the execution environment. \n
 *         @ref ERR_UEM_SSL_KEY_NOT_FOUND if SSL key does not exist. \n
 *         @ref ERR_UEM_SSL_KEY_INVALID if SSL key is not invalid.
 */
uem_result UCSecureTCPSocket_Create(IN SSecureSocketInfo *pstSocketInfo, IN uem_bool bIsServer, OUT HSSLSocket *phSocket);

/**
 * @brief Destroy a SSL socket handle.
 *
 * This function closes a SSL socket and destroys a SSL socket handle.
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
uem_result UCSecureTCPSocket_Destroy(IN OUT HSSLSocket *phSocket);

/**
 * @brief Bind a SSL socket (server-only).
 *
 * This function binds a SSL socket. This is a server-only function.
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
uem_result UCSecureTCPSocket_Bind(HSSLSocket hServerSocket);

/**
 * @brief Listen a SSL socket (server-only).
 *
 * This function listens a SSL socket. This is a server-only function. \n
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
uem_result UCSecureTCPSocket_Listen(HSSLSocket hServerSocket);

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
 *         Errors to be returned - @ref ERR_UEM_SELECT_ERROR, @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INVALID_SOCKET, @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_SELECT_ERROR, @ref ERR_UEM_NET_TIMEOUT, @ref ERR_UEM_SSL_ERROR\n
 *         @ref ERR_UEM_SELECT_ERROR, @ref ERR_UEM_NET_TIMEOUT, and corresponding errors from @ref SSocketAPI fnAccept() function. \n
 *         @ref ERR_UEM_INVALID_HANDLE can be occurred if the handle is not a socket handle.\n
 *         @ref ERR_UEM_INVALID_SOCKET can be occurred if the @a hServerSocket is a client socket. \n
 *         @ref ERR_UEM_INVALID_PARAM can be occurred if the @a nTimeout is an invalid number. \n
 *         @ref ERR_UEM_SELECT_ERROR can be occurred when select operation is failed. \n
 *         @ref ERR_UEM_NET_TIMEOUT is occurred when the timeout is happened during select operation. \n
 *         @ref ERR_UEM_SSL_ERROR occurs when the accept is failed due to the SSL.
 *
 * @sa UCBluetoothSocket_Accept, UCTCPSocket_Accept, UCUnixDomainSocket_Accept.
 */
uem_result UCSecureTCPSocket_Accept(HSSLSocket hServerSocket, IN int nTimeout, IN OUT HSSLSocket hSocket);

/**
 * @brief Connect to a server (client-only).
 *
 * This function connects to a server.
 *
 * @param hClientSocket a socket handle.
 * @param nTimeout a maximum time to wait for connection.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned -  @ref ERR_UEM_CONNECT_ERROR, @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INVALID_SOCKET, @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_CONNECT_ERROR, @ref ERR_UEM_SSL_ERROR\n
 *         @ref ERR_UEM_CONNECT_ERROR, and corresponding errors from @ref SSocketAPI fnConnect() function. \n
 *         @ref ERR_UEM_INVALID_HANDLE can be occurred if the handle is not a socket handle.\n
 *         @ref ERR_UEM_INVALID_SOCKET can be occurred if the @a hClientSocket is a server socket. \n
 *         @ref ERR_UEM_INVALID_PARAM can be occurred if the @a nTimeout is an invalid number. \n
 *         @ref ERR_UEM_CONNECT_ERROR can be occurred when the server connection is failed. \n
 *         @ref ERR_UEM_SSL_ERROR occurs when the accept is failed due to the SSL.
 *
 * @sa UCBluetoothSocket_Connect, UCTCPSocket_Connect, UCUnixDomainSocket_Connect.
 *
 */
uem_result UCSecureTCPSocket_Connect(HSSLSocket hClientSocket, IN int nTimeout);

/**
 * @brief Disconnect a socket from a server (client-only).
 *
 * This function disconnect a SSL socket from a server.
 *
 * @param hClientSocket a socket handle
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_HANDLE. \n
 *         @ref ERR_UEM_INVALID_HANDLE can be occurred if the handle is not a socket handle.
 */
uem_result UCSecureTCPSocket_Disconnect(HSSLSocket hClientSocket);

/**
 * @brief Send data (client-only).
 *
 * This function sends data.
 *
 * @param hSocket a SSL socket handle.
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
 *         @ref ERR_UEM_NET_SEND_ERROR can be occurred when send operation is failed. \n
 */
uem_result UCSecureTCPSocket_Send(HSSLSocket hSocket, IN int nTimeout, IN char *pData, IN int nDataLen, OUT int *pnSentSize);

/**
 * @brief Receive data (client-only).
 *
 * This function receives data.
 *
 * @param hSocket a SSL socket handle
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
uem_result UCSecureTCPSocket_Receive(HSSLSocket hSocket, IN int nTimeout, IN OUT char *pBuffer, IN int nBufferLen, OUT int *pnReceivedSize);

#ifdef __cplusplus
}
#endif

#endif /* SRC_COMMON_UNCONSTRAINED_INCLUDE_COMMUNICATION_UCSECURETCPSOCKET_H_ */
