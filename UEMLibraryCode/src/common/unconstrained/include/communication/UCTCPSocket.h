/*
 * UCTCPSocket.h
 *
 *  Created on: 2018. 10. 3.
 *      Author: chjej202
 */

#ifndef SRC_COMMON_UNCONSTRAINED_INCLUDE_COMMUNICATION_TCP_UCTCPSOCKET_H_
#define SRC_COMMON_UNCONSTRAINED_INCLUDE_COMMUNICATION_TCP_UCTCPSOCKET_H_

#include <uem_common.h>

#include <UCDynamicSocket.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief Bind a TCP socket (server-only).
 *
 * This function binds a TCP socket. This is a server-only function. \n
 * @ref UCDynamicSocket_Bind calls this function when the created socket is TCP.
 *
 * @param hServerSocket a socket handle to be binded.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_SOCKET_ERROR, @ref ERR_UEM_BIND_ERROR. \n
 *         @ref ERR_UEM_SOCKET_ERROR can be occurred when the socket/setsockopt operations are failed. \n
 *         @ref ERR_UEM_BIND_ERROR can be occurred when other process/thread is using a same port.
 */
uem_result UCTCPSocket_Bind(HSocket hServerSocket);

/**
 * @brief Accept a TCP client connection (server-only).
 *
 * This function accepts a client connection from different process/system. \n
 * To communicate with a client, retrieved @a hClientSocket is used. \n
 * To get new client connection, @a hClientSocket needs to be created before.
 *
 * @param hServerSocket a socket handle to accept client connection.
 * @param[in,out] hClientSocket a retrieved client connection socket.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_ACCEPT_ERROR. \n
 *         @ref ERR_UEM_ACCEPT_ERROR can be occurred when the accept operation is failed.
 */
uem_result UCTCPSocket_Accept(HSocket hServerSocket, IN OUT HSocket hClientSocket);

/**
 * @brief Connect to a TCP server (client-only).
 *
 * This function connects to a TCP server.
 *
 * @param hSocket a socket handle.
 * @param nTimeout (not used).
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_SOCKET_ERROR, @ref ERR_UEM_CONNECT_ERROR. \n
 *         @ref ERR_UEM_SOCKET_ERROR can be occurred when the socket/setsockopt operations are failed. \n
 *         @ref ERR_UEM_CONNECT_ERROR can be occurred when the connect operation is failed.
 */
uem_result UCTCPSocket_Connect(HSocket hSocket, IN int nTimeout);

#ifdef __cplusplus
}
#endif

#endif /* SRC_COMMON_UNCONSTRAINED_INCLUDE_COMMUNICATION_TCP_UCTCPSOCKET_H_ */
