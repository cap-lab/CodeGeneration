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
 * @brief
 *
 * This function
 *
 * @param enOldState
 * @param enNewState
 *
 * @return
 */
uem_result UKCPUTaskCommon_CheckTaskState(ECPUTaskState enOldState, ECPUTaskState enNewState);

/**
 * @brief
 *
 * This function
 *
 * @param pstCurrentTask
 * @param fnGo
 * @param[in,out] pllNextTime
 * @param[in,out] pnRunCount
 * @param[in,out] pnNextMaxRunCount
 * @param[out] pbFunctionCalled
 *
 * @return
 */
uem_result UKCPUTaskCommon_HandleTimeDrivenTask(STask *pstCurrentTask, FnUemTaskGo fnGo, IN OUT long long *pllNextTime,
										IN OUT int *pnRunCount, IN OUT int *pnNextMaxRunCount, OUT uem_bool *pbFunctionCalled);

/**
 * @brief
 *
 * This function
 *
 * @param pstParentTask
 * @param fnCallback
 * @param pUserData
 *
 * @return
 */
uem_result UKCPUTaskCommon_TraverseSubGraphTasks(STask *pstParentTask, FnTaskTraverse fnCallback, void *pUserData);

#ifdef __cplusplus
}
#endif


#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKCPUTASKCOMMON_H_ */
