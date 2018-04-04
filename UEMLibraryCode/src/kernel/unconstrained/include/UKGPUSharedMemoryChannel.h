/*
 * UKGPUSharedMemoryChannel.h
 *
 *  Created on: 2018. 4. 2.
 *      Author: DG-SHIN
 */

#ifndef SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKGPUSHAREDMEMORYCHANNEL_H_
#define SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKGPUSHAREDMEMORYCHANNEL_H_

#include <uem_common.h>

#include <uem_data.h>

#ifdef __cplusplus
extern "C"
{
#endif

uem_result UKGPUSharedMemoryChannel_Initialize(SChannel *pstChannel);
uem_result UKGPUSharedMemoryChannel_ReadFromQueue(SChannel *pstChannel, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead);
uem_result UKGPUSharedMemoryChannel_WriteToQueue (SChannel *pstChannel, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten);
uem_result UKGPUSharedMemoryChannel_GetNumOfAvailableData (SChannel *pstChannel, IN int nChunkIndex, OUT int *pnDataNum);
uem_result UKGPUSharedMemoryChannel_Clear(SChannel *pstChannel);
uem_result UKGPUSharedMemoryChannel_SetExit(SChannel *pstChannel, int nExitFlag);
uem_result UKGPUSharedMemoryChannel_ClearExit(SChannel *pstChannel, int nExitFlag);
uem_result UKGPUSharedMemoryChannel_Finalize(SChannel *pstChannel);

#ifdef __cplusplus
}
#endif


#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKGPUSHAREDMEMORYCHANNEL_H_ */
