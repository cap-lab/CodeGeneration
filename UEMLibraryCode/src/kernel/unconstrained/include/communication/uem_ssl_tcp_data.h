/*
 * uem_tcp_data.h
 *
 *  Created on: 2018. 6. 11.
 *      Author: jej
 */

#ifndef SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UEM_SSL_TCP_DATA_H_
#define SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UEM_SSL_TCP_DATA_H_

#include <uem_common.h>

#include <UCThreadMutex.h>
#include <UCThread.h>

#include <uem_enum.h>

#include <UKVirtualCommunication.h>

#include <uem_remote_data.h>
#include <uem_tcp_data.h>

#include <UCSSLTCPSocket.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct _SSSLTCPInfo {
	STCPInfo stTCPInfo;
	SKeyInfo *pstKeyInfo;
} SSSLTCPInfo;

#ifndef AGGREGATE_SSL_TCP_CONNECTION
typedef struct _SSSLTCPServerInfo {
	SSSLTCPInfo stSSLTCPInfo;
	SIndividualServiceInfo stServiceInfo;
} SSSLTCPServerInfo;
#else
typedef struct _SSSLTCPAggregatedServiceInfo {
	SSSLTCPInfo stSSLTCPInfo;
	SAggregateServiceInfo stServiceInfo;
} SSSLTCPAggregatedServiceInfo;
#endif

#ifndef AGGREGATE_SSL_TCP_CONNECTION
extern SSSLTCPServerInfo g_astSSLTCPServerInfo[];
extern int g_nSSLTCPServerInfoNum;
#else
extern SSSLTCPAggregatedServiceInfo g_astSSLTCPAggregateServerInfo[];
extern int g_nSSLTCPAggregateServerInfoNum;

extern SSSLTCPAggregatedServiceInfo g_astSSLTCPAggregateClientInfo[];
extern int g_nSSLTCPAggregateClientInfoNum;
#endif

#ifdef __cplusplus
}
#endif


#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UEM_SSL_TCP_DATA_H_ */
