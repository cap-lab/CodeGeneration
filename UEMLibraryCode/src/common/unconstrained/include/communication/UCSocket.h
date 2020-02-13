/*
 * UCSocket.h
 *
 *  Created on: 2020. 1. 23.
 *      Author: JangryulKim
 */

#ifndef SRC_COMMON_UNCONSTRAINED_INCLUDE_COMMUNICATION_UCSOCKET_H_
#define SRC_COMMON_UNCONSTRAINED_INCLUDE_COMMUNICATION_UCSOCKET_H_

/**
 * @brief Initialize a socket operation.
 *
 * This function initialize a socket operation.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INTERNAL_FAIL. \n
 *         @ref ERR_UEM_INTERNAL_FAIL can be occurred when the WSAStartup failed.
 */
uem_result UCSocket_Initialize();

/**
 * @brief Finalize a socket operation.
 *
 * This function finalize a socket operation.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INTERNAL_FAIL. \n
 *         @ref ERR_UEM_INTERNAL_FAIL can be occurred when the WSACleanup failed.
 */
uem_result UCSocket_Finalize();

/**
 * @brief Close a socket.
 *
 * This function closes a socket. \n
 * In Linux, it close socket with close() function. \n
 * In Windows, it close socket with closesocket() function.
 *
 * @return Always returns @ref ERR_UEM_NOERROR.
 */
uem_result UCSocket_Close(int nSocketfd);

#endif /* SRC_COMMON_UNCONSTRAINED_INCLUDE_COMMUNICATION_UCSOCKET_H_ */
