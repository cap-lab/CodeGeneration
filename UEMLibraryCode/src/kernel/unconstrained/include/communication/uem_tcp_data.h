/*
 * uem_tcp_data.h
 *
 *  Created on: 2018. 6. 11.
 *      Author: jej
 */

#ifndef SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UEM_TCP_DATA_H_
#define SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UEM_TCP_DATA_H_

#include <uem_common.h>

#include <UCThreadMutex.h>
#include <UCThread.h>

#include <uem_enum.h>

#include <UKVirtualCommunication.h>

#include <uem_remote_data.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct _STCPInfo {
	int nPort;
	char *pszIPAddress;
	EServerClientPairType enType;
} STCPInfo;

typedef struct _STCPServerInfo {
	STCPInfo stTCPInfo;
	SIndividualServiceInfo stServiceInfo;
} STCPServerInfo;



extern STCPServerInfo g_astTCPServerInfo[];
extern int g_nTCPServerInfoNum;


#ifdef __cplusplus
}
#endif


#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UEM_TCP_DATA_H_ */
