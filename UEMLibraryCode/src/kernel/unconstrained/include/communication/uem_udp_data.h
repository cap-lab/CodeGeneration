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

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct _SUDPInfo{
	const char *pszIP;
	int nPort;
} SUDPInfo;

typedef struct _SUDPSocket {
	char *pHeader;
	char *pData;
	int nHeaderLen;
	int nDataLen;
	HThreadMutex hMutex;
	HSocket hSocket;
} SUDPSocket;

#ifdef __cplusplus
}
#endif


#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UEM_UDP_DATA_H_ */
