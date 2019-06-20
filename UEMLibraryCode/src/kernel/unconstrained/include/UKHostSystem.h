/*
 * UKHostMemorySystem.h
 *
 *  Created on: 2018. 4. 5.
 *      Author: jej
 */

#ifndef SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKHOSTSYSTEM_H_
#define SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKHOSTSYSTEM_H_

#include <uem_common.h>

#include <UCThread.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief Create a CPU memory block.
 *
 * This function creates a memory block used by CPU. \n
 * This function is fnCreateMemory() function of @ref SGenericMemoryAccess.
 *
 * @param nSize memory block size to be allocated.
 * @param nProcessorId a processor id to specify CPU.
 * @param[out] ppMemory an allocated GPU memory block's pointer.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 */
uem_result UKHostSystem_CreateMemory(int nSize, int nProcessorId, OUT void **ppMemory);

/**
 * @brief Copy data to memory.
 *
 * This function copies data to a memory block. \n
 * This function's internal behavior is same to @ref UKHostSystem_CopyInMemory, @ref UKHostSystem_CopyFromMemory.
 *
 * @param[in,out] pMemory pointer of destination buffer where the content is to be copied.
 * @param pSource pointer of source of data to be copied.
 * @param nCopySize number of bytes to copy.
 *
 * @return It always returns @ref ERR_UEM_NOERROR.
 */
uem_result UKHostSystem_CopyToMemory(IN OUT void *pMemory, IN void *pSource, int nCopySize);

/**
 * @brief Copy memory to destination buffer.
 *
 * This function copies a memory block to destination buffer. \n
 * This function's internal behavior is same to @ref UKHostSystem_CopyInMemory, @ref UKHostSystem_CopyToMemory.
 *
 * @param[in,out] pDestination pointer of destination buffer where the content is to be copied.
 * @param pMemory pointer of source of data to be copied.
 * @param nCopySize number of bytes to copy.
 *
 * @return It always returns @ref ERR_UEM_NOERROR.
 */
uem_result UKHostSystem_CopyFromMemory(IN OUT void *pDestination, IN void *pMemory, int nCopySize);

/**
 * @brief Copy a memory block to another memory block.
 *
 * This function copies a memory block to another memory block. \n
 * This function's internal behavior is same to @ref UKHostSystem_CopyFromMemory, @ref UKHostSystem_CopyToMemory.
 *
 * @param[in,out] pInMemoryDst pointer of destination buffer where the content is to be copied.
 * @param pInMemorySrc pointer of source of data to be copied.
 * @param nCopySize number of bytes to copy.
 *
 * @return It always returns @ref ERR_UEM_NOERROR.
 */
uem_result UKHostSystem_CopyInMemory(IN OUT void *pInMemoryDst, IN void *pInMemorySrc, int nCopySize);

/**
 * @brief Destroy a memory block.
 *
 * This function destroys a memory block.
 *
 * @param[in,out] ppMemory a memory block to be destroyed.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKHostSystem_DestroyMemory(IN OUT void **ppMemory);

/**
 * @brief Map a thread to the specific core.
 *
 * This function maps a thread to the specific core.
 *
 * @param hThread a thread handle.
 * @param nProcessorId (not used).
 * @param nLocalId a core id starting from 0.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INTERNAL_FAIL. \n
 *         @ref ERR_UEM_INTERNAL_FAIL can be occurred when mapping-related operations are failed.
 */
uem_result UKHostSystem_MapCPU(HThread hThread, int nProcessorId, int nLocalId);

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKHOSTSYSTEM_H_ */
