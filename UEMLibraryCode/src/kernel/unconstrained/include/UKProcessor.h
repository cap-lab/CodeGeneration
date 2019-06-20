/*
 * UKProcessor.h
 *
 *  Created on: 2018. 1. 1.
 *      Author: jej
 */

#ifndef SRC_KERNEL_INCLUDE_UKPROCESSOR_H_
#define SRC_KERNEL_INCLUDE_UKPROCESSOR_H_

#include <uem_common.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief Check the processor ID is CPU.
 *
 * This function checks the processor ID is CPU.
 * If the processor is CPU, it is TRUE. Otherwise it is FALSE.
 *
 * @param nProcessorId processor ID
 * @param[out] pbIsCPU processor id is CPU or not.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKProcessor_IsCPUByProcessorId(int nProcessorId, OUT uem_bool *pbIsCPU);

/**
 * @brief Get GPU processor ID from generic processor ID.
 *
 * This function gets GPU processor ID from generic processor ID number.
 * GPU processor ID is used for mapping GPU.
 *
 * @param nProcessorId processor ID
 * @param[out] pnGPUProcessorId  processor ID in GPU.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKProcessor_GetGPUProcessorId(IN int nProcessorId, OUT int *pnGPUProcessorId);

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_INCLUDE_UKPROCESSOR_H_ */
