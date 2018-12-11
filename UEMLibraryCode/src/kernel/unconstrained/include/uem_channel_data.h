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


typedef enum _ESharedMemoryAccessType {
	ACCESS_TYPE_CPU_ONLY,
	ACCESS_TYPE_CPU_GPU,
	ACCESS_TYPE_GPU_CPU,
	ACCESS_TYPE_GPU_GPU,
	ACCESS_TYPE_GPU_GPU_DIFFERENT,
} ESharedMemoryAccessType;

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

typedef uem_result (*FnCreateMemory)(int nSize, int nProcessorId, OUT void **ppMemory);
typedef uem_result (*FnCopyMemory)(IN void *pDest, IN void *pSource, int nCopySize);
typedef uem_result (*FnDestroyMemory)(IN OUT void **ppMemory);

typedef struct _SGenericMemoryAccess {
	FnCreateMemory fnCreateMemory;
	FnCopyMemory fnCopyToMemory;
	FnCopyMemory fnCopyInMemory;
	FnCopyMemory fnCopyFromMemory;
	FnDestroyMemory fnDestroyMemory;
} SGenericMemoryAccess;

typedef struct _SSharedMemoryChannel {
	ESharedMemoryAccessType enAccessType;
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
	int nOutputMaxChunkNum; // maximum chunk size for all port sample rate cases (output port)
	int nMaxChunkNum; // maximum chunk size for all port sample rate cases (input port)
	SAvailableChunk *pstAvailableInputChunkHead;
	SAvailableChunk *pstAvailableInputChunkTail;
	SGenericMemoryAccess *pstMemoryAccessAPI;
	uem_bool bStaticAllocation;
	uem_bool bInitialDataUpdated;
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
	SPort *pstInputPort;
	SPort *pstOutputPort;
	int nInitialDataLen;
	int nProcessorId;
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
typedef uem_result (*FnChannelFillInitialData)(SChannel *pstChannel);
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
	FnChannelFillInitialData fnFillInitialData;
	FnChannelFinalize fnFinalize;
	FnChannelAPIInitialize fnAPIInitialize;
	FnChannelAPIFinalize fnAPIFinalize;
} SChannelAPI;

extern SChannel g_astChannels[];
extern int g_nChannelNum;

extern SChannelAPI *g_astChannelAPIList[];
extern int g_nChannelAPINum;

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_INCLUDE_UEM_CHANNEL_DATA_H_ */
