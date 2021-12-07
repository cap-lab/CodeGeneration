/*
 * uem_remote_data.h
 *
 *  Created on: 2019. 5. 21.
 *      Author: jej
 */

#ifndef SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UEM_REMOTE_DATA_H_
#define SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UEM_REMOTE_DATA_H_

#include <uem_common.h>

#include <UCThreadMutex.h>
#include <UCFixedSizeQueue.h>

#include <UKSerialCommunicationManager.h>

#include <UKUEMProtocol.h>

#include <UKVirtualCommunication.h>

#ifdef __cplusplus
extern "C"
{
#endif


typedef enum _ServerClientPairType {
	PAIR_TYPE_CLIENT,
	PAIR_TYPE_SERVER,
} EServerClientPairType;

typedef enum _EMasterSlavePairType {
	PAIR_TYPE_MASTER,
	PAIR_TYPE_SLAVE,
} EMasterSlavePairType;


typedef enum _ECommunicationMethod {
	COMMUNICATION_METHOD_TCP,
	COMMUNICATION_METHOD_BLUETOOTH,
	COMMUNICATION_METHOD_SERIAL,
	COMMUNICATION_METHOD_SECURETCP,
} ECommunicationMethod;


typedef enum _EConnectionMethod {
	CONNECTION_METHOD_INDIVIDUAL,
	CONNECTION_METHOD_AGGREGATE,
} EConnectionMethod;


typedef struct _SAggregateServiceInfo {
	HThread hServiceThread;
	int nMaxChannelAccessNum;
	HVirtualSocket hSocket;
	SVirtualCommunicationAPI *pstAPI;
	HSerialCommunicationManager hManager;
	uem_bool bInitialized;
	SEncryptionKeyInfo *pstEncKeyInfo;
} SAggregateServiceInfo;


typedef struct _SIndividualServiceInfo {
	HVirtualSocket hSocket;
	HThread hServiceThread;
	SVirtualCommunicationAPI *pstAPI;
	SEncryptionKeyInfo *pstEncKeyInfo;
} SIndividualServiceInfo;


typedef union _UCommunicationQueue {
	HFixedSizeQueue hRequestQueue;
	HFixedSizeQueue hResponseQueue;
} UCommunicationQueue;


typedef struct _SAggregateConnectionInfo {
	int nChannelId;
	UCommunicationQueue uQueue;
	SAggregateServiceInfo *pstServiceInfo;
} SAggregateConnectionInfo;

typedef struct _SIndividualConnectionInfo {
	int nChannelId;
	ECommunicationMethod enCommunicationMethod;
	void *pCommunicationInfo; // STCPInfo (for TCP client with individual connection), other communication, not used
	EServerClientPairType enType;
	SVirtualCommunicationAPI *pstCommunicationAPI;
	HVirtualSocket hSocket;
	HUEMProtocol hProtocol;
	SEncryptionKeyInfo *pstEncKeyInfo;
} SIndividualConnectionInfo;

typedef struct _SRemoteChannel {
	EConnectionMethod enConnectionMethod;
	void *pConnectionInfo; // SIndividualConnectionInfo or SAggregateConnectionInfo
	HThreadMutex hMutex; // not used (may be used sometime?)
	uem_bool bChannelExit;
} SRemoteChannel;

typedef struct _SRemoteWriterChannel {
	SRemoteChannel stCommonInfo;
	HThread hReceivingThread;
	char *pBuffer;
	int nBufLen;
	SSharedMemoryChannel *pstInternalChannel;
} SRemoteWriterChannel;

typedef struct _SRemoteReaderChannel {
	SRemoteChannel stCommonInfo;
	SGenericMemoryAccess *pstReaderAccess;
} SRemoteReaderChannel;


extern SIndividualConnectionInfo g_astIndividualConnectionInfo[];
extern int g_nIndividualConnectionInfoNum;

extern SAggregateConnectionInfo g_astAggregateConnectionInfo[];
extern int g_nAggregateConnectionInfoNum;


#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UEM_REMOTE_DATA_H_ */
