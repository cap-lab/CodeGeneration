/*
 * UKGPUSystem.h
 *
 *  Created on: 2018. 4. 5.
 *      Author: jej
 */

#ifndef SRC_KERNEL_UNCONSTRAINED_UKGPUSYSTEM_H_
#define SRC_KERNEL_UNCONSTRAINED_UKGPUSYSTEM_H_

#include <uem_common.h>

#include <UCThread.h>

#ifdef __cplusplus
extern "C"
{
#endif

uem_result UKGPUSystem_CreateMemory(int nSize, int nProcessorId, OUT void **ppMemory);
uem_result UKGPUSystem_CreateHostAllocMemory(int nSize, int nProcessorId, OUT void **ppMemory);
uem_result UKGPUSystem_CopyHostToDeviceMemory(IN void *pDest, IN void *pSrc, int nCopySize);
uem_result UKGPUSystem_CopyDeviceToHostMemory(IN void *pDest, IN void *pSrc, int nCopySize);
uem_result UKGPUSystem_CopyDeviceToDeviceMemory(IN void *pDest, IN void *pSrc, int nCopySize);
uem_result UKGPUSystem_DestroyHostAllocMemory(IN OUT void **ppMemory);
uem_result UKGPUSystem_DestroyMemory(IN OUT void **ppMemory);
uem_result UKGPUSystem_MapGPU(HThread hThread, int nProcessorId, int nLocalId);

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_UNCONSTRAINED_UKGPUSYSTEM_H_ */

