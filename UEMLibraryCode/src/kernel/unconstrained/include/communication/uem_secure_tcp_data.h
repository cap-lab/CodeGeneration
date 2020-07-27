/*
 * uem_secure_tcp_data.h
 *
 *  Created on: 2020. 5. 21.
 *      Author: jrkim
 */

#ifndef SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UEM_SECURE_TCP_DATA_H_
#define SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UEM_SECURE_TCP_DATA_H_

#include <uem_common.h>

#include <UCThreadMutex.h>
#include <UCThread.h>

#include <uem_enum.h>

#include <UKVirtualCommunication.h>
#include <uem_remote_data.h>
#include <uem_tcp_data.h>
#include <UCSecureTCPSocket.h>


#ifdef __cplusplus
extern "C"
{
#endif

typedef struct _SSecureTCPInfo {
	STCPInfo stTCPInfo;
	SSecurityKeyInfo *pstKeyInfo;
} SSecureTCPInfo;

#ifndef AGGREGATE_SECURETCP_CONNECTION
typedef struct _SSecureTCPServerInfo {
	SSecureTCPInfo stSSLTCPInfo;
	SIndividualServiceInfo stServiceInfo;
} SSecureTCPServerInfo;
#else
typedef struct _SSecureTCPAggregatedServiceInfo {
	SSecureTCPInfo stSSLTCPInfo;
	SAggregateServiceInfo stServiceInfo;
} SSecureTCPAggregatedServiceInfo;
#endif

#ifndef AGGREGATE_SECURETCP_CONNECTION
extern SSecureTCPServerInfo g_astSecureTCPServerInfo[];
extern int g_nSecureTCPServerInfoNum;
#else
extern SSecureTCPAggregatedServiceInfo g_astSSLTCPAggregateServerInfo[];
extern int g_nSSLTCPAggregateServerInfoNum;

extern SSecureTCPAggregatedServiceInfo g_astSSLTCPAggregateClientInfo[];
extern int g_nSSLTCPAggregateClientInfoNum;
#endif

#ifdef __cplusplus
}
#endif


#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UEM_SECURE_TCP_DATA_H_ */
