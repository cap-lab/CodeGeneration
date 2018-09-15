/*
 * uem_channel_data.h
 *
 *  Created on: 2018. 8. 30.
 *      Author: chjej202
 */

#ifndef SRC_KERNEL_CONSTRAINED_INCLUDE_UEM_CHANNEL_DATA_H_
#define SRC_KERNEL_CONSTRAINED_INCLUDE_UEM_CHANNEL_DATA_H_

#include <uem_common.h>

#include <uem_enum.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct _SPort SPort;

typedef struct _SPort {
	int nTaskId;
	const char *pszPortName;
	//EPortSampleRateType enSampleRateType;
	//int nSampleRate;
	//int nSampleSize;
	SPort *pstSubGraphPort;
} SPort;

typedef struct _SSharedMemoryChannel {
	void *pBuffer;
	void *pDataStart;
	void *pDataEnd;
	int nDataLen;
	uem_bool bInitialDataUpdated;
} SSharedMemoryChannel;

typedef struct _SChannel {
	int nChannelIndex;
	int nNextChannelIndex;
	ECommunicationType enType;
	int nBufSize;
	SPort *pstInputPort;
	SPort *pstOutputPort;
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

extern SChannelAPI *g_astChannelAPIList[];
extern int g_nChannelAPINum;

extern SChannel g_astChannels[];
extern int g_nChannelNum;


#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_CONSTRAINED_INCLUDE_UEM_CHANNEL_DATA_H_ */
