/*
 * UKProgram.h
 *
 *  Created on: 2020. 7. 23.
 *      Author: jrkim
 */

#ifndef SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKPROGRAM_H_
#define SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKPROGRAM_H_

#include <uem_common.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief Get program execution time.
 *
 * This function retrieves program execution time.
 *
 * @param[out] pnValue time value.
 * @param[out] penMetric time unit.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKProgram_GetExecutionTime(OUT int *pnValue, OUT ETimeMetric *penMetric);

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKPROGRAM_H_ */
