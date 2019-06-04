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
 * @brief
 *
 * This function
 *
 * @param nSize
 * @param nProcessorId
 * @param[out] ppMemory
 *
 * @return
 */
uem_result UKHostSystem_CreateMemory(int nSize, int nProcessorId, OUT void **ppMemory);

/**
 * @brief
 *
 * This function
 *
 * @param pMemory
 * @param pSource
 * @param nCopySize
 *
 * @return
 */
uem_result UKHostSystem_CopyToMemory(IN void *pMemory, IN void *pSource, int nCopySize);

/**
 * @brief
 *
 * This function
 *
 * @param pDestination
 * @param pMemory
 * @param nCopySize
 *
 * @return
 */
uem_result UKHostSystem_CopyFromMemory(IN void *pDestination, IN void *pMemory, int nCopySize);

/**
 * @brief
 *
 * This function
 *
 * @param pInMemoryDst
 * @param pInMemorySrc
 * @param nCopySize
 *
 * @return
 */
uem_result UKHostSystem_CopyInMemory(IN void *pInMemoryDst, IN void *pInMemorySrc, int nCopySize);

/**
 * @brief
 *
 * This function
 *
 * @param[in,out] ppMemory
 *
 * @return
 */
uem_result UKHostSystem_DestroyMemory(IN OUT void **ppMemory);

/**
 * @brief
 *
 * This function
 *
 * @param hThread
 * @param nProcessorId
 * @param nLocalId
 *
 * @return
 */
uem_result UKHostSystem_MapCPU(HThread hThread, int nProcessorId, int nLocalId);

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKHOSTSYSTEM_H_ */
