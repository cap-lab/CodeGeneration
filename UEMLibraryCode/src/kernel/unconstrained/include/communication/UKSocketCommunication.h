/*
 * UKSocketCommunication.h
 *
 *  Created on: 2019. 5. 23.
 *      Author: jej
 */

#ifndef SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKSOCKETCOMMUNICATION_H_
#define SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKSOCKETCOMMUNICATION_H_

#include <uem_common.h>

#include <UKVirtualCommunication.h>


#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief Destroy a socket.
 *
 * This function destroys a socket. This function is shared by both TCP and Bluetooth communication. \n
 * This is an implementation function of SVirtualCommunicationAPI's fnDestroy().
 *
 * @param[in,out] phSocket a socket handle to be destroyed.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_NOT_SUPPORTED, \n
 *         @ref ERR_UEM_NOT_FOUND. \n
 *         @ref ERR_UEM_INVALID_HANDLE can be occurred if the handle is not a socket handle.\n
 *         @ref ERR_UEM_NOT_FOUND can be occurred if the socket API is not set yet.\n
 *         @ref ERR_UEM_NOT_SUPPORTED if @a enSocketType of @a pstSocketInfo is invalid.
 */
uem_result UKSocketCommunication_Destroy(IN OUT HVirtualSocket *phSocket);

/**
 * @brief Connect through a socket.
 *
 * This function connects to a server. \n
 * This function is shared by both TCP and Bluetooth communication. \n
 * This is an implementation function of SVirtualCommunicationAPI's fnConnect().
 *
 * @param hSocket a socket handle.
 * @param nTimeout a maximum time to wait for connection (only used by Bluetooth connection).
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned -  @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INVALID_SOCKET, @ref ERR_UEM_INVALID_PARAM, \n
 *         @ref ERR_UEM_CONNECT_ERROR, and corresponding errors from @ref SSocketAPI fnConnect() function. \n
 *         @ref ERR_UEM_INVALID_HANDLE can be occurred if the handle is not a socket handle.\n
 *         @ref ERR_UEM_INVALID_SOCKET can be occurred if the @a hClientSocket is a server socket. \n
 *         @ref ERR_UEM_CONNECT_ERROR can be occurred when the server connection is failed.
 */
uem_result UKSocketCommunication_Connect(HVirtualSocket hSocket, int nTimeout);

/**
 * @brief Disconnect through a socket.
 *
 * This function disconnect a socket from a server. \n
 * This function is shared by both TCP and Bluetooth communication. \n
 * This is an implementation function of SVirtualCommunicationAPI's fnDisconnect().
 *
 * @param hSocket a socket handle.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_HANDLE. \n
 *         @ref ERR_UEM_INVALID_HANDLE can be occurred if the handle is not a socket handle.
 */
uem_result UKSocketCommunication_Disconnect(HVirtualSocket hSocket);

/**
 * @brief Listen a socket.
 *
 * This function starts to receive new connection from clients. \n
 * This function is shared by both TCP and Bluetooth communication. \n
 * This is an implementation function of SVirtualCommunicationAPI's fnListen().
 *
 * @param hSocket a socket handle.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INVALID_SOCKET, @ref ERR_UEM_LISTEN_ERROR, \n
 *         and corresponding errors from @ref SSocketAPI fnBind() function. \n
 *         @ref ERR_UEM_INVALID_HANDLE can be occurred if the handle is not a socket handle.\n
 *         @ref ERR_UEM_INVALID_SOCKET can be occurred if the @a hServerSocket is a client socket. \n
 *         @ref ERR_UEM_LISTEN_ERROR can be occurred when listen operation is failed.
 */
uem_result UKSocketCommunication_Listen(HVirtualSocket hSocket);

/**
 * @brief Accept a client connection.
 *
 * This function accepts a client connection from different process/system. \n
 * This function is shared by both TCP and Bluetooth communication. \n
 * This is an implementation function of SVirtualCommunicationAPI's fnAccept().
 *
 * @param hSocket a socket handle.
 * @param nTimeout a maximum time to wait for client connection.
 * @param[in,out] hAcceptedSocket
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INVALID_SOCKET, @ref ERR_UEM_INVALID_PARAM, \n
 *         @ref ERR_UEM_SELECT_ERROR, @ref ERR_UEM_NET_TIMEOUT, and corresponding errors from @ref SSocketAPI fnAccept() function. \n
 *         @ref ERR_UEM_INVALID_HANDLE can be occurred if the handle is not a socket handle.\n
 *         @ref ERR_UEM_INVALID_SOCKET can be occurred if the @a hServerSocket is a client socket. \n
 *         @ref ERR_UEM_INVALID_PARAM can be occurred if the @a nTimeout is an invalid number. \n
 *         @ref ERR_UEM_SELECT_ERROR can be occurred when select operation is failed. \n
 *         @ref ERR_UEM_NET_TIMEOUT is occurred when the timeout is happened during select operation.
 */
uem_result UKSocketCommunication_Accept(HVirtualSocket hSocket, int nTimeout, IN OUT HVirtualSocket hAcceptedSocket);

/**
 * @brief Send data.
 *
 * This function sends data through a socket. \n
 * This function is shared by both TCP and Bluetooth communication. \n
 * This is an implementation function of SVirtualCommunicationAPI's fnSend().
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
uem_result UKSocketCommunication_Send(HVirtualSocket hSocket, IN int nTimeout, IN char *pData, IN int nDataLen, OUT int *pnSentSize);

/**
 * @brief Receive data.
 *
 * This function receives data through a socket. \n
 * This function is shared by both TCP and Bluetooth communication. \n
 * This is an implementation function of SVirtualCommunicationAPI's fnReceive().
 *
 * @param hSocket a socket handle.
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
uem_result UKSocketCommunication_Receive(HVirtualSocket hSocket, IN int nTimeout, IN OUT char *pBuffer, IN int nBufferLen, OUT int *pnReceivedSize);

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKSOCKETCOMMUNICATION_H_ */
