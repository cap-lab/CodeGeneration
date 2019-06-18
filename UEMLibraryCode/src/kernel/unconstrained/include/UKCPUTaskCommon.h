/*
 * UKCPUTaskCommon.h
 *
 *  Created on: 2018. 2. 5.
 *      Author: jej
 */

#ifndef SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKCPUTASKCOMMON_H_
#define SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKCPUTASKCOMMON_H_

#include <uem_common.h>

#include <uem_data.h>

#include <UCThreadMutex.h>

#include <UKTask_internal.h>

#ifdef __cplusplus
extern "C"
{
#endif


typedef enum _ECPUTaskState {
	TASK_STATE_STOP,
	TASK_STATE_RUNNING,
	TASK_STATE_SUSPEND,
	TASK_STATE_STOPPING,
} ECPUTaskState;

/**
 * @brief Check task state transition is allowable.
 *
 * This function checks the task state transition is allowable.
 *
 * @param enOldState old task state.
 * @param enNewState new task state.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_ALREADY_DONE, @ref ERR_UEM_SKIP_THIS, @ref ERR_UEM_ILLEGAL_CONTROL. \n
 *         @ref ERR_UEM_ALREADY_DONE is occurred when old task state and new task state are same. \n
 *         @ref ERR_UEM_SKIP_THIS is occurred when the task state transition is ignorable (ex. stop => suspend). \n
 *         @ref ERR_UEM_ILLEGAL_CONTROL is occurred when the task state transition is not allowable.
 */
uem_result UKCPUTaskCommon_CheckTaskState(ECPUTaskState enOldState, ECPUTaskState enNewState);

/**
 * @brief Execute time-driven task with time period.
 *
 * This function calculates the next execution time period of time-driven task and executes time-driven task at every period. \n
 * If the time period is too small, it uses a run count to execute multiple times in a minimum time period (1ms).
 *
 * @param pstCurrentTask task structure information.
 * @param fnGo task go() function.
 * @param[in,out] pllNextTime next time period to execute a task.
 * @param[in,out] pnRunCount task number of execution in current time period.
 * @param[in,out] pnNextMaxRunCount maximum number of execution to run on current time period.
 * @param[out] pbFunctionCalled boolean value of go() function is called or not.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_ILLEGAL_DATA, @ref ERR_UEM_INTERNAL_FAIL. \n
 *         @ref ERR_UEM_ILLEGAL_DATA is occurred when the time unit is wrong. \n
 *         @ref ERR_UEM_INTERNAL_FAIL is occurred when the time measurement function is failed.
 */
uem_result UKCPUTaskCommon_HandleTimeDrivenTask(STask *pstCurrentTask, FnUemTaskGo fnGo, IN OUT long long *pllNextTime,
										IN OUT int *pnRunCount, IN OUT int *pnNextMaxRunCount, OUT uem_bool *pbFunctionCalled);

/**
 * @brief Traverse all the task structure in the subgraphs.
 *
 * This function traverses subgraph tasks of the specific parent task. \n
 * If @a pstParentTask is NULL, it traverse all tasks.
 *
 * @param pstParentTask parent task structure.
 * @param fnCallback callback function which are going to be called during traverse.
 * @param pUserData user data used in callback function.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_OUT_OF_MEMORY, @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_NO_DATA, \n
 *         and corresponding results from the callback function. \n
 *         @ref ERR_UEM_NO_DATA can be occurred if there is no task in application.
 */
uem_result UKCPUTaskCommon_TraverseSubGraphTasks(STask *pstParentTask, FnTaskTraverse fnCallback, void *pUserData);

#ifdef __cplusplus
}
#endif


#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKCPUTASKCOMMON_H_ */
