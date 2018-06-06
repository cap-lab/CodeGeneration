/*
 * UKChannelMemory.h
 *
 *  Created on: 2018. 5. 26.
 *      Author: jej
 */

#ifndef SRC_KERNEL_INCLUDE_UKCHANNELMEMORY_H_
#define SRC_KERNEL_INCLUDE_UKCHANNELMEMORY_H_

#include <uem_common.h>

#include <uem_data.h>

#ifdef __cplusplus
extern "C"
{
#endif

uem_result UKChannelMemory_Initialize(SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel);
uem_result UKChannelMemory_ReadFromQueue(SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead);
uem_result UKChannelMemory_ReadFromBuffer(SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead);
uem_result UKChannelMemory_WriteToBuffer (SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten);
uem_result UKChannelMemory_WriteToQueue (SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten);
uem_result UKChannelMemory_GetAvailableChunk (SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel, OUT int *pnChunkIndex);
uem_result UKChannelMemory_GetNumOfAvailableData (SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel, IN int nChunkIndex, OUT int *pnDataNum);
uem_result UKChannelMemory_Clear(SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel);
uem_result UKChannelMemory_SetExit(SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel, int nExitFlag);
uem_result UKChannelMemory_ClearExit(SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel, int nExitFlag);
uem_result UKChannelMemory_Finalize(SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel);

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_INCLUDE_UKCHANNELMEMORY_H_ */
