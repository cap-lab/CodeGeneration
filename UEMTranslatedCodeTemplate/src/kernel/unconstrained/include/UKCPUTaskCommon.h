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

#include <UKTask.h>

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

uem_result UKCPUTaskCommon_CheckTaskState(ECPUTaskState enOldState, ECPUTaskState enNewState);
uem_result UKCPUTaskCommon_HandleTimeDrivenTask(STask *pstCurrentTask, FnUemTaskGo fnGo, IN OUT long long *pllNextTime,
										IN OUT int *pnRunCount, IN OUT int *pnNextMaxRunCount);

#ifdef __cplusplus
}
#endif


#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKCPUTASKCOMMON_H_ */
