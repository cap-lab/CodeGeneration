/*
 * UKUDPSocketMulticast.h
 *
 *  Created on: 2018. 6. 11.
 *      Author: wecracy
 */

#ifndef SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKUDPSOCKETMULTICAST_H_
#define SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKUDPSOCKETMULTICAST_H_

#include <uem_common.h>

#include <UCDynamicSocket.h>

#include <uem_data.h>
#include <uem_udp_data.h>

#ifdef __cplusplus
extern "C"
{
#endif

uem_result UKUDPSocketMulticast_Initialize(SMulticastGroup *pstMulticastGroup);
uem_result UKUDPSocket_Finalize(SMulticastGroup *pstMulticastGroup);

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKTCPSOCKETCHANNEL_H_ */
