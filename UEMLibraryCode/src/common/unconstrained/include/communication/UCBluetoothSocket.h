/*
 * UCBluetoothSocket.h
 *
 *  Created on: 2018. 10. 3.
 *      Author: chjej202
 */

#ifndef SRC_COMMON_UNCONSTRAINED_INCLUDE_COMMUNICATION_BLUETOOTH_UCBLUETOOTHSOCKET_H_
#define SRC_COMMON_UNCONSTRAINED_INCLUDE_COMMUNICATION_BLUETOOTH_UCBLUETOOTHSOCKET_H_

#include <uem_common.h>

#include <UCDynamicSocket.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief Bind a Bluetooth socket (slave-only).
 *
 * This function binds a Bluetooth socket. This is a slave-only function. \n
 * @ref UCDynamicSocket_Bind calls this function when the created socket is Bluetooth.
 *
 * @param hServerSocket a Socket handle to be binded.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_SOCKET_ERROR, @ref ERR_UEM_BIND_ERROR. \n
 *         @ref ERR_UEM_SOCKET_ERROR can be occurred when the socket/setsockopt operations are failed. \n
 *         @ref ERR_UEM_BIND_ERROR can be occurred when other process/thread is using a same mac address.
 */
uem_result UCBluetoothSocket_Bind(HSocket hServerSocket);

/**
 * @brief Accept a Bluetooth master connection (slave-only).
 *
 * This function accepts a master connection from different process/system. \n
 * To communicate with master, retrieved @a hClientSocket is used. \n
 * To get new master connection, @a hClientSocket needs to be created before.
 *
 * @param hServerSocket a socket handle to accept master connection.
 * @param[in,out] hClientSocket a retrieved master connection socket.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_ACCEPT_ERROR. \n
 *         @ref ERR_UEM_ACCEPT_ERROR can be occurred when the accept operation is failed.
 */
uem_result UCBluetoothSocket_Accept(HSocket hServerSocket, IN OUT HSocket hClientSocket);

/**
 * @brief Connect to a Bluetooth slave (master-only).
 *
 * This function connects to a Bluetooth slave.
 *
 * @param hSocket a socket handle.
 * @param nTimeout (not used).
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         @ref ERR_UEM_IN_PROGRESS is returned if the connect operation is nonblocking and \n
 *         needed to wait for establishing the connection. \n
 *         Errors to be returned - @ref ERR_UEM_SOCKET_ERROR, @ref ERR_UEM_CONVERSION_ERROR, @ref ERR_UEM_CONNECT_ERROR. \n
 *         @ref ERR_UEM_SOCKET_ERROR can be occurred when the socket/setsockopt operations are failed. \n
 *         @ref ERR_UEM_CONVERSION_ERROR can be occurred when the socket address is wrong. \n
 *         @ref ERR_UEM_CONNECT_ERROR can be occurred when the connect operation is failed.
 *
 */
uem_result UCBluetoothSocket_Connect(HSocket hSocket, IN int nTimeout);

#ifdef __cplusplus
}
#endif

#endif /* SRC_COMMON_UNCONSTRAINED_INCLUDE_COMMUNICATION_BLUETOOTH_UCBLUETOOTHSOCKET_H_ */
