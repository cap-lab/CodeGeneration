/*
 * uem_memory_data.h
 *
 *  Created on: 2019. 6. 20.
 *      Author: wecracy
 */

#ifndef SRC_KERNEL_UNCONSTRAINED_INCLUDE_UEM_MEMORY_DATA_H_
#define SRC_KERNEL_UNCONSTRAINED_INCLUDE_UEM_MEMORY_DATA_H_

#include <uem_common.h>

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

#ifdef __cplusplus
}
#endif


#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_UEM_MEMORY_DATA_H_ */
