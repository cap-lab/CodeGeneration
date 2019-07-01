/*
 * uem_udp_data.h
 *
 *  Created on: 2019. 6. 20.
 *      Author: wecracy
 */

#ifndef SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UEM_UDP_DATA_H_
#define SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UEM_UDP_DATA_H_

#include <uem_common.h>

#include <UCThreadMutex.h>
#include <UCThread.h>

#include <UCDynamicSocket.h>

#include <uem_enum.h>

#include <uem_multicast_data.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct _SUDPInfo{
	int nPort;
} SUDPInfo;

typedef struct _SUDPSocket {
	HThread hManagementThread; // for Reader
	char *pBuffer; // temporary buffer for writing data to shared memory channel (for Reader)
	int nBufLen; // temporary buffer length (for Reader)
	HThreadMutex hMutex;
	uem_bool bExit;
	void *pstMulticastManager;
	SGenericMemoryAccess *pstReaderAccess; // for READER channel
	HSocket *hSocket;
} SUDPSocket;

#ifdef __cplusplus
}
#endif


#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UEM_UDP_DATA_H_ */
