/*
 * uem_channel_data.h
 *
 *  Created on: 2018. 3. 30.
 *      Author: jej
 */

#ifndef SRC_KERNEL_INCLUDE_UEM_CHANNEL_DATA_H_
#define SRC_KERNEL_INCLUDE_UEM_CHANNEL_DATA_H_

#include <uem_common.h>


#include <UCThreadMutex.h>
#include <UCThreadEvent.h>
#include <UCThread.h>

//#include <UCDynamicSocket.h>

#include <uem_enum.h>

//#include <UKUEMProtocol.h>

#ifdef __cplusplus
extern "C"
{
#endif


typedef struct _SChunk {
	void *pChunkStart; // fixed
	void *pDataStart; // vary
	void *pDataEnd; // vary
	int nChunkDataLen; // written data length
	int nAvailableDataNum; // for broadcast loop
} SChunk;

typedef struct _SChunkInfo {
	// These values can be changed during execution depending on Mode transition
	SChunk *astChunk;
	int nChunkNum; // nTotalSampleRate / nSampleRate
	int nChunkLen; // nSampleRate * nSampleSize => maximum size of each chunk item
} SChunkInfo;

typedef struct _SAvailableChunk SAvailableChunk;

typedef struct _SAvailableChunk {
	int nChunkIndex;
	int nSampleNum;
	SAvailableChunk *pstPrev;
	SAvailableChunk *pstNext;
} SAvailableChunk;

typedef uem_result (*FnCreateMemory)(int nSize, OUT void **ppMemory);
typedef uem_result (*FnCopyToMemory)(IN void *pMemory, IN void *pSource, int nCopySize);
typedef uem_result (*FnCopyFromMemory)(IN void *pDestination, IN void *pMemory, int nCopySize);
typedef uem_result (*FnDestroyMemory)(IN OUT void **ppMemory);

typedef struct _SGenericMemoryAccess {
	FnCreateMemory fnCreateMemory;
	FnCopyToMemory fnCopyToMemory;
	FnCopyFromMemory fnCopyFromMemory;
	FnDestroyMemory fnDestroyMemory;
} SGenericMemoryAccess;

typedef struct _SSharedMemoryChannel {
	void *pBuffer;
	void *pDataStart;
	void *pDataEnd;
	int nDataLen;
	int nReadReferenceCount;
	int nWriteReferenceCount;
	uem_bool bReadExit;
	uem_bool bWriteExit;
	HThreadMutex hMutex; // Channel global mutex
	HThreadEvent hReadEvent; // Channel read available notice conditional variable
	HThreadEvent hWriteEvent; // Channel write available notice conditional variable
	SChunkInfo stInputPortChunk;
	SChunkInfo stOutputPortChunk;
	int nWrittenOutputChunkNum;

	SAvailableChunk *astAvailableInputChunkList; // size
	int nMaxChunkNum; // maximum chunk size for all port sample rate cases
	SAvailableChunk *pstAvailableInputChunkHead;
	SAvailableChunk *pstAvailableInputChunkTail;
	SGenericMemoryAccess *pstMemoryAccessAPI;
	uem_bool bStaticAllocation;
} SSharedMemoryChannel;


typedef struct _STCPClientInfo {
	const char *pszIPAddress;
	int nPort;
} STCPClientInfo;

/*
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
	STCPClientInfo *pstClientInfo; // if TCP is server, this value is NULL
	SExternalCommunicationInfo *pstCommunicationInfo; // get and set
	HThread hReceivingThread; // for WRITER channel
	char *pBuffer; // temporary buffer for getting data from shared memory channel
	int nBufLen; // temporary buffer length
	HThreadMutex hMutex;
	SSharedMemoryChannel *pstInternalChannel;
} STCPSocketChannel;
*/
#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_INCLUDE_UEM_CHANNEL_DATA_H_ */
