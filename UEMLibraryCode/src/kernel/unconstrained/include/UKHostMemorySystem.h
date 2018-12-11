/*
 * UKHostMemorySystem.h
 *
 *  Created on: 2018. 4. 5.
 *      Author: jej
 */

#ifndef SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKHOSTMEMORYSYSTEM_H_
#define SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKHOSTMEMORYSYSTEM_H_

#include <uem_common.h>

#ifdef __cplusplus
extern "C"
{
#endif

uem_result UKHostMemorySystem_CreateMemory(int nSize, int nProcessorId, OUT void **ppMemory);
uem_result UKHostMemorySystem_CopyToMemory(IN void *pMemory, IN void *pSource, int nCopySize);
uem_result UKHostMemorySystem_CopyFromMemory(IN void *pDestination, IN void *pMemory, int nCopySize);
uem_result UKHostMemorySystem_CopyInMemory(IN void *pInMemoryDst, IN void *pInMemorySrc, int nCopySize);
uem_result UKHostMemorySystem_DestroyMemory(IN OUT void **ppMemory);

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKHOSTMEMORYSYSTEM_H_ */
