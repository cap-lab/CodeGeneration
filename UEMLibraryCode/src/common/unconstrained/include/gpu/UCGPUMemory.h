/*
 * UCCUDABasic.h
 *
 *  Created on: 2018. 4. 2.
 *      Author: DG-SHIN
 */

#ifndef SRC_COMMON_INCLUDE_GPU_UCGPUMEMORY_H_
#define SRC_COMMON_INCLUDE_GPU_UCGPUMEMORY_H_

#include "uem_common.h"

#ifdef __cplusplus
extern "C"
{
#endif


typedef enum _EMemcpyKind {
	MEMCPY_KIND_HOST_TO_HOST,
	MEMCPY_KIND_HOST_TO_DEVICE,
	MEMCPY_KIND_DEVICE_TO_HOST,
	MEMCPY_KIND_DEVICE_TO_DEVICE,
	MEMCPY_KIND_DEFAULT,
} EMemcpyKind;


typedef enum _EMemoryProperty {
	MEMORY_PROPERTY_DEFAULT,		//cudaHostAllocDefault
	MEMORY_PROPERTY_PORTABLE,		//cudaHostAllocPortable
	MEMORY_PROPERTY_MAPPED,			//cudaHostAllocMapped
	MEMORY_PROPERTY_WRITE_COMBINED,	//cudaHostAllocWriteCombined
} EMemoryProperty;

/**
 * @brief
 *
 * This function
 *
 * @param ppMemory
 * @param nSize
 *
 * @return
 */
uem_result UCGPUMemory_Malloc(void **ppMemory, int nSize);

/**
 * @brief
 *
 * This function
 *
 * @param ppMemory
 * @param nSize
 * @param flags
 *
 * @return
 */
uem_result UCGPUMemory_HostAlloc(void **ppMemory, int nSize, EMemoryProperty flags);

/**
 * @brief
 *
 * This function
 *
 * @param pMemory
 *
 * @return
 */
uem_result UCGPUMemory_Free(void *pMemory);

/**
 * @brief
 *
 * This function
 *
 * @param pMemory
 *
 * @return
 */
uem_result UCGPUMemory_FreeHost(void *pMemory);

/**
 * @brief
 *
 * This function
 *
 * @param pDest
 * @param pSrc
 * @param nSize
 * @param flags
 *
 * @return
 */
uem_result UCGPUMemory_Memcpy(void *pDest, const void *pSrc, int nSize, EMemcpyKind flags);

/**
 * @brief
 *
 * This function
 *
 * @param nDevice
 *
 * @return
 */
uem_result UCGPUMemory_SetDevice(int nDevice);

/**
 * @brief
 *
 * This function
 *
 * @param[out] pnDevice
 *
 * @return
 */
uem_result UCGPUMemory_GetDevice(OUT int *pnDevice);

#ifdef __cplusplus
}
#endif



#endif /* SRC_COMMON_INCLUDE_GPU_UCGPUMEMORY_H_ */
