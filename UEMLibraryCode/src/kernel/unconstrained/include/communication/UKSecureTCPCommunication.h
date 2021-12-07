/*
 * UKSecureTCPCommunication.h
 *
 *  Created on: 2020. 5. 21.
 *      Author: jrkim
 */

#ifndef SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKSECURETCPCOMMUNICATION_H_
#define SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKSECURETCPCOMMUNICATION_H_

#include <uem_common.h>

#include <UKVirtualCommunication.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief Create a SSL TCP communication socket.
 *
 * This function creates a SSL TCP communication socket. \n
 * This is an implementation function of SVirtualCommunicationAPI's fnCreate().
 *
 * @param[out] phSocket phSocket a socket handle to be created.
 * @param pSocketInfo SSL TCP socket options.
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
uem_result UKSecureTCPCommunication_Create(OUT HVirtualSocket *phSocket, void *pSocketInfo);

/**
 * @brief Destroy a SSL TCP communication socket.
 *
 * This function destroy a SSL socket.
 * This is an implementation function of SVirtualCommunicationAPI's fnDestroy().
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
uem_result UKSecureTCPCommunication_Destroy(HVirtualSocket *phSocket);

/**
 * @brief Connect to a server (client-only).
 *
 * This function connects to a server.
 * This is an implementation function of SVirtualCommunicationAPI's fnConnect().
 *
 * @param hSocket a socket handle.
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
uem_result UKSecureTCPCommunication_Connect(HVirtualSocket hSocket, int nTimeout);

/**
 * @brief Disconnect a socket from a server (client-only).
 *
 * This function disconnect a SSL socket from a server.
 * This is an implementation function of SVirtualCommunicationAPI's fnDisConnect().
 *
 * @param hSocket a socket handle
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_HANDLE. \n
 *         @ref ERR_UEM_INVALID_HANDLE can be occurred if the handle is not a socket handle.
 */
uem_result UKSecureTCPCommunication_Disconnect(HVirtualSocket hSocket);

/**
 * @brief Listen a SSL socket (server-only).
 *
 * This function listens a SSL socket. This is a server-only function. \n
 * This is an implementation function of SVirtualCommunicationAPI's fnListen().
 *
 * @param hSocket a socket handle to listen.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INVALID_SOCKET, @ref ERR_UEM_LISTEN_ERROR. \n
 *         @ref ERR_UEM_INVALID_HANDLE can be occurred if the handle is not a socket handle.\n
 *         @ref ERR_UEM_INVALID_SOCKET can be occurred if the @a hServerSocket is a client socket. \n
 *         @ref ERR_UEM_LISTEN_ERROR can be occurred when listen operation is failed.
 */
uem_result UKSecureTCPCommunication_Listen(HVirtualSocket hSocket);

/**
 * @brief Accept a client connection (server-only).
 *
 * This function accepts a client connection from different process/system. \n
 * This is an implementation function of SVirtualCommunicationAPI's fnAccept().
 *
 * @param[in,out] hSocket a retrieved client socket.
 * @param nTimeout a maximum time to wait for client connection.
 * @param hAcceptedSocket a socket handle to accept client connection.
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
uem_result UKSecureTCPCommunication_Accept(HVirtualSocket hSocket, int nTimeout, IN OUT HVirtualSocket hAcceptedSocket);

/**
 * @brief Send data.
 *
 * This function sends data.
  * This is an implementation function of SVirtualCommunicationAPI's fnSend().
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
uem_result UKSecureTCPCommunication_Send(HVirtualSocket hSocket, IN int nTimeout, IN char *pData, IN int nDataLen, OUT int *pnSentSize);

/**
 * @brief Receive data.
 *
 * This function receives data.
  * This is an implementation function of SVirtualCommunicationAPI's fnReceive().
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
uem_result UKSecureTCPCommunication_Receive(HVirtualSocket hSocket, IN int nTimeout, IN OUT char *pBuffer, IN int nBufferLen, OUT int *pnReceivedSize);

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKSECURETCPCOMMUNICATION_H_ */
