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

uem_result UCTCPSocket_Bind(HSocket hSocket);
uem_result UCTCPSocket_Accept(HSocket hServerSocket, HSocket hClientSocket);
uem_result UCTCPSocket_Connect(HSocket hSocket, IN int nTimeout);

#ifdef __cplusplus
}
#endif

#endif /* SRC_COMMON_UNCONSTRAINED_INCLUDE_COMMUNICATION_TCP_UCTCPSOCKET_H_ */
