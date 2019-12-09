/*
 * UKGPUSystem.h
 *
 *  Created on: 2018. 4. 5.
 *      Author: jej
 */

#ifndef SRC_KERNEL_UNCONSTRAINED_INCLUDE_GPU_UKGPUSYSTEM_H_
#define SRC_KERNEL_UNCONSTRAINED_INCLUDE_GPU_UKGPUSYSTEM_H_

#include <uem_common.h>

#include <UCThread.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief Create a GPU memory block.
 *
 * This function creates a memory block used by GPU. \n
 * This function is fnCreateMemory() function of @ref SGenericMemoryAccess.
 *
 * @param nSize memory block size to be allocated.
 * @param nProcessorId a processor id to specify GPU.
 * @param[out] ppMemory an allocated GPU memory block's pointer.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, converted errors retrieved from cudamalloc(), cudaGetDevice(), cudaSetDevice().
 */
uem_result UKGPUSystem_CreateMemory(int nSize, int nProcessorId, OUT void **ppMemory);

/**
 * @brief Create a host-side memory block.
 *
 * This function creates host memory block. This memory block is used for transferring data between different GPUs. \n
 * This function is fnCreateMemory() function of @ref SGenericMemoryAccess.
 *
 * @param nSize memory block size to be allocated.
 * @param nProcessorId (not used).
 * @param[out] ppMemory an allocated host memory block's pointer.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, converted errors retrieved from cudaHostAlloc().
 */
uem_result UKGPUSystem_CreateHostAllocMemory(int nSize, int nProcessorId, OUT void **ppMemory);

/**
 * @brief Copy CPU memory block to GPU memory block.
 *
 * This function copies memory block located in host memory to GPU memory. \n
 * This function can be one of fnCopyToMemory()/fnCopyFromMemory/fnCopyInMemory() function in @ref SGenericMemoryAccess.
 *
 * @param[in,out] pDest pointer to the destination of data to be copied.
 * @param pSrc pointer to the source of data to be copied.
 * @param nCopySize size of data to be copied.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, converted errors retrieved from cudaMemcpy().
 *
 */
uem_result UKGPUSystem_CopyHostToDeviceMemory(IN OUT void *pDest, IN void *pSrc, int nCopySize);

/**
 * @brief Copy from CPU memory block to GPU memory block.
 *
 * This function copies memory block located in GPU memory to host memory. \n
 * This function can be one of fnCopyToMemory()/fnCopyFromMemory/fnCopyInMemory() function in @ref SGenericMemoryAccess.
 *
 * @param[in,out] pDest pointer to the destination of data to be copied.
 * @param pSrc pointer to the source of data to be copied.
 * @param nCopySize size of data to be copied.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, converted errors retrieved from cudaMemcpy().
 */
uem_result UKGPUSystem_CopyDeviceToHostMemory(IN OUT void *pDest, IN void *pSrc, int nCopySize);

/**
 * @brief Copy GPU memory block to GPU memory block.
 *
 * This function copies memory inside GPU. \n
 * This function can be one of fnCopyToMemory()/fnCopyFromMemory/fnCopyInMemory() function in @ref SGenericMemoryAccess.
 *
 * @param[in,out] pDest pointer to the destination of data to be copied.
 * @param pSrc pointer to the source of data to be copied.
 * @param nCopySize size of data to be copied.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, converted errors retrieved from cudaMemcpy().
 */
uem_result UKGPUSystem_CopyDeviceToDeviceMemory(IN OUT void *pDest, IN void *pSrc, int nCopySize);

/**
 * @brief Destroy memory block allocated from @ref UKGPUSystem_CreateHostAllocMemory.
 *
 * This function destroys memory block allocated by @ref UKGPUSystem_CreateHostAllocMemory. \n
 * This function is fnDestroyMemory() function of @ref SGenericMemoryAccess.
 *
 * @param[in,out] ppMemory host memory block to be destroyed.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, converted errors retrieved from cudaFreeHost().
 */
uem_result UKGPUSystem_DestroyHostAllocMemory(IN OUT void **ppMemory);

/**
 * @brief Destroy allocated memory block from @ref UKGPUSystem_CreateMemory.
 *
 * This function destroys memory block allocated by @ref UKGPUSystem_CreateMemory. \n
 * This function is fnDestroyMemory() function of @ref SGenericMemoryAccess.
 *
 * @param[in,out] ppMemory GPU memory block to be destroyed.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, converted errors retrieved from cudaFree().
 */
uem_result UKGPUSystem_DestroyMemory(IN OUT void **ppMemory);

/**
 * @brief Map current thread to specific GPU.
 *
 * This function maps current thread to specific GPU. This function changes the GPU memory allocation location. \n
 * This function is fnMapProcessor() function of @ref SGenericMapProcessor.
 *
 * @param hThread a target thread to change the mapped GPU.
 * @param nProcessorId a processor id to distinguish different GPUs.
 * @param nLocalId (not used).
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, converted errors retrieved from cudaSetDevice().
 */
uem_result UKGPUSystem_MapGPU(HThread hThread, int nProcessorId, int nLocalId);

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_GPU_UKGPUSYSTEM_H_ */

