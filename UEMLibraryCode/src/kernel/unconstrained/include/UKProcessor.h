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

uem_result UKProcessor_IsCPUByProcessorId(int nProcessorId, uem_bool *pbIsCPU);
uem_result UKProcessor_GetGPUProcessorId(IN int nProcessorId, OUT int *pnGPUProcessorId);

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_INCLUDE_UKPROCESSOR_H_ */
