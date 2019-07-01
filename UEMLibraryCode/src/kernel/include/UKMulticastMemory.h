/*
 * UKMulticastMemory.h
 *
 *  Created on: 2019. 5. 26.
 *      Author: wecracy
 */

#ifndef SRC_KERNEL_INCLUDE_UKMULTICASTMEMORY_H_
#define SRC_KERNEL_INCLUDE_UKMULTICASTMEMORY_H_

#include <uem_common.h>

#include <uem_data.h>

#ifdef __cplusplus
extern "C"
{
#endif

uem_result UKMulticastMemory_Initialize(SMulticastGroup *pstMulticastGroup, SSharedMemoryMulticast *pstSharedMemoryMulticast);
uem_result UKMulticastMemory_ReadFromBuffer(SMulticastGroup *pstMulticastGroup, SSharedMemoryMulticast *pstSharedMemoryMulticast, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead);
uem_result UKMulticastMemory_WriteToBuffer (SMulticastGroup *pstMulticastGroup, SSharedMemoryMulticast *pstSharedMemoryMulticast, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten);
uem_result UKMulticastMemory_Clear(SMulticastGroup *pstMulticastGroup, SSharedMemoryMulticast *pstSharedMemoryMulticast);
uem_result UKMulticastMemory_SetExit(SMulticastGroup *pstMulticastGroup, SSharedMemoryMulticast *pstSharedMemoryMulticast, int nExitFlag);
uem_result UKMulticastMemory_ClearExit(SMulticastGroup *pstMulticastGroup, SSharedMemoryMulticast *pstSharedMemoryMulticast, int nExitFlag);
uem_result UKMulticastMemory_Finalize(SMulticastGroup *pstMulticastGroup, SSharedMemoryMulticast *pstSharedMemoryMulticast);

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_INCLUDE_UKMULTICASTMEMORY_H_ */
