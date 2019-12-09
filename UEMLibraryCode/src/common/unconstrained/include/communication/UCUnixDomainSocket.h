/*
 * UCUnixDomainSocket.h
 *
 *  Created on: 2018. 10. 3.
 *      Author: chjej202
 */

#ifndef SRC_COMMON_UNCONSTRAINED_INCLUDE_COMMUNICATION_UCUNIXDOMAINSOCKET_H_
#define SRC_COMMON_UNCONSTRAINED_INCLUDE_COMMUNICATION_UCUNIXDOMAINSOCKET_H_

#include <uem_common.h>

#include <UCDynamicSocket.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief Bind a Unix domain socket (server-only).
 *
 * This function binds a Unix domain socket. This is a server-only function. \n
 * @ref UCDynamicSocket_Bind calls this function when the created socket is Unix domain socket.
 *
 * @warning This function is not used in current UEM library code.
 *
 * @param hServerSocket a socket handle to be binded.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_SOCKET_ERROR, @ref ERR_UEM_BIND_ERROR. \n
 *         @ref ERR_UEM_SOCKET_ERROR can be occurred when the socket/setsockopt operations are failed. \n
 *         @ref ERR_UEM_BIND_ERROR can be occurred when other process/thread is using a same mac address.
 */
uem_result UCUnixDomainSocket_Bind(HSocket hServerSocket);

/**
 * @brief Accept a Unix domain socket client connection (server-only).
 *
 * This function accepts a client connection from different process/system. \n
 * To communicate with a client, retrieved @a hClientSocket is used. \n
 * To get new client connection, @a hClientSocket needs to be created before.
 *
 * @warning This function is not used in current UEM library code.
 *
 * @param hServerSocket a socket handle to accept client connection.
 * @param[in,out] hClientSocket a retrieved client connection socket.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_ACCEPT_ERROR. \n
 *         @ref ERR_UEM_ACCEPT_ERROR can be occurred when the accept operation is failed.
 */
uem_result UCUnixDomainSocket_Accept(HSocket hServerSocket, IN OUT HSocket hClientSocket);

/**
 * @brief Connect to a Unix domain socket server (client-only).
 *
 * This function connects to a Unix domain socket server.
 *
 * @warning This function is not used in current UEM library code.
 *
 * @param hSocket a socket handle.
 * @param nTimeout (not used).
 *
 * @return  @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_SOCKET_ERROR, @ref ERR_UEM_CONNECT_ERROR. \n
 *         @ref ERR_UEM_SOCKET_ERROR can be occurred when the socket/setsockopt operations are failed. \n
 *         @ref ERR_UEM_CONNECT_ERROR can be occurred when the connect operation is failed.
 */
uem_result UCUnixDomainSocket_Connect(HSocket hSocket, IN int nTimeout);

/**
 * @brief Extra destruction routine of Unix domain socket.
 *
 * This function removes the unix domain socket path which was used for communication.
 *
 * @warning This function is not used in current UEM library code.
 *
 * @param hSocket a socket handle.
 *
 * @return This function always returns @ref ERR_UEM_NOERROR. \n
 */
uem_result UCUnixDomainSocket_Destroy(HSocket hSocket);

#ifdef __cplusplus
}
#endif


#endif /* SRC_COMMON_UNCONSTRAINED_INCLUDE_COMMUNICATION_UCUNIXDOMAINSOCKET_H_ */
