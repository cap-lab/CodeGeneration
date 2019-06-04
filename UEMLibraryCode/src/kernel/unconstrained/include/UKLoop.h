/*
 * UKLoop.h
 *
 *  Created on: 2018. 8. 30.
 *      Author: DG-SHIN
 */

#ifndef SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKLOOP_H_
#define SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKLOOP_H_

#include <uem_common.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief
 *
 * This function
 *
 * @param nCallerTaskId
 * @param nTaskThreadId
 * @param[out] pnTaskIteration
 *
 * @return
 */
uem_result UKLoop_GetLoopTaskIteration(IN int nCallerTaskId, IN int nTaskThreadId, OUT int *pnTaskIteration);

/**
 * @brief
 *
 * This function
 *
 * @param nCallerTaskId
 *
 * @return
 */
uem_result UKLoop_StopNextIteration(IN int nCallerTaskId);

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKLOOP_H_ */
