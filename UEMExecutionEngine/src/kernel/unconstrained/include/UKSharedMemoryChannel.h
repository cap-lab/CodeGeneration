/*
 * UKSharedMemoryChannel.h
 *
 *  Created on: 2017. 11. 9.
 *      Author: jej
 */

#ifndef SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKSHAREDMEMORYCHANNEL_H_
#define SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKSHAREDMEMORYCHANNEL_H_

#include <uem_common.h>

#include <uem_data.h>

#ifdef __cplusplus
extern "C"
{
#endif

uem_result UKSharedMemoryChannel_Initialize(SChannel *pstChannel);
uem_result UKSharedMemoryChannel_ReadFromQueue(SChannel *pstChannel, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead);
uem_result UKSharedMemoryChannel_ReadFromBuffer(SChannel *pstChannel, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead);
uem_result UKSharedMemoryChannel_WriteToBuffer (SChannel *pstChannel, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten);
uem_result UKSharedMemoryChannel_WriteToQueue (SChannel *pstChannel, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten);
uem_result UKSharedMemoryChannel_GetAvailableChunk (SChannel *pstChannel, OUT int *pnChunkIndex);
uem_result UKSharedMemoryChannel_GetNumOfAvailableData (SChannel *pstChannel, IN int nChunkIndex, OUT int *pnDataNum);
uem_result UKSharedMemoryChannel_Clear(SChannel *pstChannel);

#ifdef __cplusplus
}
#endif




#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKSHAREDMEMORYCHANNEL_H_ */
