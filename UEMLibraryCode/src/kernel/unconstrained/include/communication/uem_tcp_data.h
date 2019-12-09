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

#ifndef AGGREGATE_TCP_CONNECTION
typedef struct _STCPServerInfo {
	STCPInfo stTCPInfo;
	SIndividualServiceInfo stServiceInfo;
} STCPServerInfo;
#else
typedef struct _STCPAggregatedServiceInfo {
	STCPInfo stTCPInfo;
	SAggregateServiceInfo stServiceInfo;
} STCPAggregatedServiceInfo;
#endif

#ifndef AGGREGATE_TCP_CONNECTION
extern STCPServerInfo g_astTCPServerInfo[];
extern int g_nTCPServerInfoNum;
#else
extern STCPAggregatedServiceInfo g_astTCPAggregateServerInfo[];
extern int g_nTCPAggregateServerInfoNum;

extern STCPAggregatedServiceInfo g_astTCPAggregateClientInfo[];
extern int g_nTCPAggregateClientInfoNum;
#endif

#ifdef __cplusplus
}
#endif


#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UEM_TCP_DATA_H_ */
