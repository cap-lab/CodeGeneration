/*
 * UKMulticastMemory.h
 *
 *  Created on: 2019. 6. 20.
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
uem_result UKMulticastMemory_ReadFromBuffer(SMulticastPort *pstMulticastPort, SSharedMemoryMulticast *pstSharedMemoryMulticast, IN OUT unsigned char *pBuffer, IN int nDataToRead, OUT int *pnDataRead);
uem_result UKMulticastMemory_WriteToBuffer(SMulticastPort *pstMulticastPort, SSharedMemoryMulticast *pstSharedMemoryMulticast, IN unsigned char *pBuffer, IN int nDataToWrite, OUT int *pnDataWritten);
uem_result UKMulticastMemory_Clear(SMulticastGroup *pstMulticastGroup, SSharedMemoryMulticast *pstSharedMemoryMulticast);
uem_result UKMulticastMemory_Finalize(SMulticastGroup *pstMulticastGroup, SSharedMemoryMulticast *pstSharedMemoryMulticast);

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_INCLUDE_UKMULTICASTMEMORY_H_ */
