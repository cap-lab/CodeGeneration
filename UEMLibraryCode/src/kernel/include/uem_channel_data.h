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

#include <uem_enum.h>


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


typedef struct _SPortSampleRate {
	const char *pszModeName; // Except MTM, all mode name becomes "Default"
	int nSampleRate; // sample rate (for general task, nSampleRate and nTotalSampleRate are same)
	int nMaxAvailableDataNum; // for broadcast loop
} SPortSampleRate;

typedef struct _SPort SPort;

// nBufSize /  (nTotalSampleRate *nSampleSize) => number of loop queue?

typedef struct _SPort {
	int nTaskId;
	const char *pszPortName;
	EPortSampleRateType enSampleRateType;
	SPortSampleRate *astSampleRates; // If the task is MTM, multiple sample rates can be existed.
	int nNumOfSampleRates;
	int nCurrentSampleRateIndex;
	int nSampleSize;
	EPortType enPortType;
	SPort *pstSubGraphPort;
} SPort;


/*
typedef struct _SPortMap {
	int nTaskId;
	char *pszPortName;
	int nChildTaskId;
	char *pszChildTaskPortName;
	EPortDirection enPortDirection;
	EPortMapType enPortMapType;
} SPortMap;
*/

typedef struct _SChannel {
	int nChannelIndex;
	int nNextChannelIndex;
	ECommunicationType enType;
	EChannelType enChannelType;
	int nBufSize;
	SPort stInputPort;
	SPort stOutputPort;
	int nInitialDataLen;
	void *pChannelStruct;
} SChannel;


typedef uem_result (*FnChannelAPIInitialize)();
typedef uem_result (*FnChannelAPIFinalize)();
typedef uem_result (*FnChannelInitialize)(SChannel *pstChannel);
typedef uem_result (*FnChannelReadFromQueue)(SChannel *pstChannel, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead);
typedef uem_result (*FnChannelReadFromBuffer)(SChannel *pstChannel, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead);
typedef uem_result (*FnChannelWriteToBuffer)(SChannel *pstChannel, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten);
typedef uem_result (*FnChannelWriteToQueue)(SChannel *pstChannel, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten);
typedef uem_result (*FnChannelGetAvailableChunk)(SChannel *pstChannel, OUT int *pnChunkIndex);
typedef uem_result (*FnChannelGetNumOfAvailableData)(SChannel *pstChannel, IN int nChunkIndex, OUT int *pnDataNum);
typedef uem_result (*FnChannelClear)(SChannel *pstChannel);
typedef uem_result (*FnChannelSetExit)(SChannel *pstChannel, int nExitFlag);
typedef uem_result (*FnChannelClearExit)(SChannel *pstChannel, int nExitFlag);
typedef uem_result (*FnChannelFinalize)(SChannel *pstChannel);

typedef struct _SChannelAPI {
	FnChannelInitialize fnInitialize;
	FnChannelReadFromQueue fnReadFromQueue;
	FnChannelReadFromBuffer fnReadFromBuffer;
	FnChannelWriteToQueue fnWriteToQueue;
	FnChannelWriteToBuffer fnWriteToBuffer;
	FnChannelGetAvailableChunk fnGetAvailableChunk;
	FnChannelGetNumOfAvailableData fnGetNumOfAvailableData;
	FnChannelClear fnClear;
	FnChannelSetExit fnSetExit;
	FnChannelClearExit fnClearExit;
	FnChannelFinalize fnFinalize;
	FnChannelAPIInitialize fnAPIInitialize;
	FnChannelAPIFinalize fnAPIFinalize;
} SChannelAPI;


#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_INCLUDE_UEM_CHANNEL_DATA_H_ */
