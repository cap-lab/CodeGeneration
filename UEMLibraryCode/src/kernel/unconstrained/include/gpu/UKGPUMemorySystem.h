/*
 * UKGPUMemorySystem.h
 *
 *  Created on: 2018. 4. 5.
 *      Author: jej
 */

#ifndef SRC_KERNEL_UNCONSTRAINED_UKGPUMEMORYSYSTEM_H_
#define SRC_KERNEL_UNCONSTRAINED_UKGPUMEMORYSYSTEM_H_

#include <uem_common.h>

#ifdef __cplusplus
extern "C"
{
#endif

uem_result UKGPUMemorySystem_CreateMemory(int nSize, int nProcessorId, OUT void **ppMemory);
uem_result UKGPUMemorySystem_CreateHostAllocMemory(int nSize, int nProcessorId, OUT void **ppMemory);
uem_result UKGPUMemorySystem_CopyHostToDeviceMemory(IN void *pDest, IN void *pSrc, int nCopySize);
uem_result UKGPUMemorySystem_CopyDeviceToHostMemory(IN void *pDest, IN void *pSrc, int nCopySize);
uem_result UKGPUMemorySystem_CopyDeviceToDeviceMemory(IN void *pDest, IN void *pSrc, int nCopySize);
uem_result UKGPUMemorySystem_DestroyHostAllocMemory(IN OUT void **ppMemory);
uem_result UKGPUMemorySystem_DestroyMemory(IN OUT void **ppMemory);

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_UNCONSTRAINED_UKGPUMEMORYSYSTEM_H_ */

