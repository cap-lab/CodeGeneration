/*
 * UFLoop.h
 *
 *  Created on: 2018. 8. 30.
 *      Author: DG-SHIN
 */

#ifndef SRC_API_INCLUDE_UFLOOP_H_
#define SRC_API_INCLUDE_UFLOOP_H_

#include <uem_common.h>

#ifdef __cplusplus
extern "C"
{
#endif

#ifndef API_LITE
/**
 * @brief Return current loop Iteration count of task.
 *
 * Used when the task itself is a loop task or belongs to another loop task. \n
 * Returns the iteration count of the nearest loop task. \n
 *
 * @param nCallerTaskId id of caller task.
 * @param nTaskThreadId id of caller task thread.
 * @param[out] pnTaskIteration returned iteration value.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_INVALID_PARAM for invalid caller task id or task thread id. \n
 * @ref ERR_UEM_NO_DATA if task is not a loop task or not belongs to a loop task. \n
 */
uem_result UFLoop_GetIteration(IN int nCallerTaskId, IN int nTaskThreadId, OUT int *pnTaskIteration);

/**
 * @brief Stop next iteration of loop task.
 *
 * Can be used only for Deignated task inside CType Loop task.
 *
 * @param nCallerTaskId id of caller task thread.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_ILLEGAL_DATA if parent Task does not exists or is not a loop task. \n
 * @ref ERR_UEM_INVALID_HANDLE if parent Task is not Convergent loop task or caller task is not Designated task. \n
 *
 */
uem_result UFLoop_StopNextIteration(IN int nCallerTaskId);
#endif

#ifdef __cplusplus
}
#endif


#endif /* SRC_API_INCLUDE_UFLOOP_H_ */
