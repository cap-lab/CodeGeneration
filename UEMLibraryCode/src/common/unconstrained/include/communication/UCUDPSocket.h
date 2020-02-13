/*
 * UCUDPSocket.h
 *
 *  Created on: 2019. 6. 20.
 *      Author: wecracy
 */

#ifndef SRC_COMMON_UNCONSTRAINED_INCLUDE_COMMUNICATION_UCUDPSOCKET_H_
#define SRC_COMMON_UNCONSTRAINED_INCLUDE_COMMUNICATION_UCUDPSOCKET_H_

#include <uem_common.h>

#include <UCDynamicSocket.h>

#ifdef __cplusplus
extern "C"
{
#endif

#define UDP_MAX 65507

/**
 * @brief Bind a UDP socket.
 *
 * This function binds a UDP socket. \n
 * @ref UCDynamicSocket_Bind calls this function when the created socket is UDP.
 *
 * @param hSocket a socket handle to be binded.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_SOCKET_ERROR, @ref ERR_UEM_BIND_ERROR. \n
 *         @ref ERR_UEM_SOCKET_ERROR can be occurred when the socket/setsockopt operations are failed. \n
 *         @ref ERR_UEM_BIND_ERROR can be occurred when other process/thread is using a same port.
 */
uem_result UCUDPSocket_Bind(HSocket hSocket);

/**
 * @brief Create a UDP socket handle.
 *
 * This function creates a UDP socket handle.
 * @ref UCDynamicSocket_Create calls this function when pstSocketInfo is UDP.
 *
 * @param hSocket a socket handle to be created
 * @param pstSocketInfo a socket setting information
 * @param bIsServer unused parameter (for API format)
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_SOCKET_ERROR. \n
 *         @ref ERR_UEM_SOCKET_ERROR can be occurred when creating socket is failed or setting option is failed.
 */
uem_result UCUDPSocket_Create(HSocket hSocket, SSocketInfo *pstSocketInfo, uem_bool bIsServer);

#ifdef __cplusplus
}
#endif

#endif /* SRC_COMMON_UNCONSTRAINED_INCLUDE_COMMUNICATION_UCUDPSOCKET_H_ */
