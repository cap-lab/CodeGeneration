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

#include <UCDynamicSocket.h>

#include <uem_enum.h>

#include <UKUEMProtocol.h>
#include <uem_channel_data.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct _STCPServerInfo {
	int nPort;
	HSocket hServerSocket;
	HThread hServerThread;
} STCPServerInfo;

typedef struct _STCPClientInfo {
	const char *pszIPAddress;
	int nPort;
} STCPClientInfo;

// TCP_XXXX_WRITER
// CLIENT_WRITER => connect and create thread
// SERVER_WRITER => create thread
typedef struct _SExternalCommunicationInfo {
	int nChannelId;
	ECommunicationType enType;
	HSocket hSocket;
	HUEMProtocol hProtocol;
} SExternalCommunicationInfo;


typedef struct _STCPSocketChannel {
	STCPClientInfo *pstClientInfo; // if TCP is server, this value is NULL (for CLIENT)
	SExternalCommunicationInfo *pstCommunicationInfo; // get and set
	HThread hReceivingThread; // for WRITER channel
	char *pBuffer; // temporary buffer for getting data from shared memory channel (for WRITER)
	int nBufLen; // temporary buffer length (for WRITER)
	HThreadMutex hMutex;
	uem_bool bChannelExit;
	SSharedMemoryChannel *pstInternalChannel; // for WRITER
	SGenericMemoryAccess *pstReaderAccess; // for READER channel
} STCPSocketChannel;


extern STCPServerInfo g_astTCPServerInfo[];
extern SExternalCommunicationInfo g_astExternalCommunicationInfo[];
extern int g_nTCPServerInfoNum;
extern int g_nExternalCommunicationInfoNum;

#ifdef __cplusplus
}
#endif


#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UEM_TCP_DATA_H_ */
