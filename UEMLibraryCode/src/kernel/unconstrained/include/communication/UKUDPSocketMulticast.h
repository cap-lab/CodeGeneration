/*
 * UKUDPSocketMulticast.h
 *
 *  Created on: 2018. 6. 20.
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

#define CONNECT_TIMEOUT (3)
#define CONNECT_RETRY_COUNT (100)
#define SECOND_IN_MILLISECOND (1000)

#define MULTICAST_UDP_HEADER_GROUP_ID_SIZE (1)
#define MULTICAST_UDP_HEADER_SIZE (MULTICAST_UDP_HEADER_GROUP_ID_SIZE)

typedef struct _SUDPMulticast{
	SUDPSocket *pstSocket;
	HThread hManagementThread; // for Reader
	uem_bool bExit;
	void *pstMulticastManager;
}SUDPMulticast;

uem_result UKUDPSocketMulticast_Initialize(IN SMulticastGroup *pstMulticastGroup);
uem_result UKUDPSocketMulticast_WriteToBuffer(IN SMulticastPort *pstMulticastPort, IN unsigned char *pData, IN int nDataToWrite, OUT int *pnDataWritten);
uem_result UKUDPSocketMulticast_Finalize(IN SMulticastGroup *pstMulticastGroup);

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKUDPSOCKETMULTICAST_H_ */
