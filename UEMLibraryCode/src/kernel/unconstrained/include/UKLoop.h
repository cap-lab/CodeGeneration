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
 * @brief Get current loop iteration count of parent loop task.
 *
 * This function retrieves the current loop iteration count. \n
 * If the task itself is a loop task, this function retrieves the task's loop iteration count. \n
 * Otherwise, it returns the iteration count of parent loop task.
 *
 * @param nCallerTaskId id of caller task.
 * @param nTaskThreadId id of call task thread.
 * @param[out] pnTaskIteration retrieved iteration number.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_NO_DATA.
 *         @ref ERR_UEM_NO_DATA is occurred when the task is not belong to loop task.
 */
uem_result UKLoop_GetLoopTaskIteration(IN int nCallerTaskId, IN int nTaskThreadId, OUT int *pnTaskIteration);

/**
 * @brief Stop next iteration of loop task.
 *
 * This function terminates the loop of loop task. This function can be called by designated task only.
 *
 * @param nCallerTaskId id of caller task.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_ILLEGAL_DATA. \n
 *         @ref ERR_UEM_NO_DATA is occurred when the task is not found or there is no loop task for the caller task.
 */
uem_result UKLoop_StopNextIteration(IN int nCallerTaskId);

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKLOOP_H_ */
