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

uem_result UCBluetoothSocket_Bind(HSocket hSocket);
uem_result UCBluetoothSocket_Accept(HSocket hServerSocket, HSocket hClientSocket);
uem_result UCBluetoothSocket_Connect(HSocket hSocket, IN int nTimeout);

#ifdef __cplusplus
}
#endif

#endif /* SRC_COMMON_UNCONSTRAINED_INCLUDE_COMMUNICATION_BLUETOOTH_UCBLUETOOTHSOCKET_H_ */
