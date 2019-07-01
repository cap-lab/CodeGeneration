/*
 * UCCUDABasic.h
 *
 *  Created on: 2018. 4. 2.
 *      Author: DG-SHIN
 */

#ifndef SRC_COMMON_UNCONSTRAINED_INCLUDE_GPU_UCGPUMEMORY_H_
#define SRC_COMMON_UNCONSTRAINED_INCLUDE_GPU_UCGPUMEMORY_H_

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
 * @brief Allocate new memory for GPU.
 *
 * This function allocates new GPU memory. This function can be used if CUDA is available.
 *
 * @param[out] ppMemory an allocated GPU memory's pointer.
 * @param nSize memory size to be allocated
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - converted errors retrieved from cudamalloc().
 */
uem_result UCGPUMemory_Malloc(OUT void **ppMemory, int nSize);

/**
 * @brief Allocate host-side memory.
 *
 * This function allocates host memory which is must faster than accessing normal host memory. \n
 * This function can be used if CUDA is available.
 *
 * @param[out] ppMemory an allocated host memory's pointer.
 * @param nSize memory size to be allocated
 * @param flags memory flags of @ref EMemoryProperty which are corresponding from cudaHostAllocXXX Flags
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - converted errors retrieved from cudaHostAlloc().
 */
uem_result UCGPUMemory_HostAlloc(void **ppMemory, int nSize, EMemoryProperty flags);

/**
 * @brief Deallocate the memory previously allocated from @ref UCGPUMemory_Malloc.
 *
 * This function deallocates the memory previously allocated from @ref UCGPUMemory_Malloc.
 *
 * @param pMemory the allocated GPU memory pointer.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - converted errors retrieved from cudaFree().
 */
uem_result UCGPUMemory_Free(void *pMemory);

/**
 * @brief Deallocate the memory previously allocated from @ref UCGPUMemory_HostAlloc.
 *
 * This function deallocates the memory previously allocated from @ref UCGPUMemory_HostAlloc.
 *
 * @param pMemory the allocated host memory pointer.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - converted errors retrieved from cudaFreeHost().
 */
uem_result UCGPUMemory_FreeHost(void *pMemory);

/**
 * @brief Copy memory from/to host/GPU depending on @a EMemcpyKind.
 *
 * This function copies the host/GPU memory depending on @a EMemcpyKind.
 *
 * @param pDest pointer to the destination of data to be copied.
 * @param pSrc pointer to the source of data to be copied.
 * @param nSize size of data to be copied.
 * @param flags a flag indicates the location of source and destination pointer.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - converted errors retrieved from cudaMemcpy().
 *
 * @sa EMemcpyKind
 */
uem_result UCGPUMemory_Memcpy(void *pDest, const void *pSrc, int nSize, EMemcpyKind flags);

/**
 * @brief Set GPU memory access location.
 *
 * This function sets the GPU memory access location. \n
 * By setting this value, the running thread can map the GPU memory to the specific GPU if the multiple GPUs are used.
 *
 * @param nDevice A GPU device number to be mapped.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - converted errors retrieved from cudaSetDevice().
 */
uem_result UCGPUMemory_SetDevice(int nDevice);

/**
 * @brief Get a current thread's GPU memory access location.
 *
 * This function retrieves the GPU device number which is mapped on the current running thread.
 *
 * @param[out] pnDevice A GPU device number which is mapped on the current thread.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - converted errors retrieved from cudaGetDevice().
 */
uem_result UCGPUMemory_GetDevice(OUT int *pnDevice);

#ifdef __cplusplus
}
#endif



#endif /* SRC_COMMON_UNCONSTRAINED_INCLUDE_GPU_UCGPUMEMORY_H_ */
