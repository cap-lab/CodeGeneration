/*
 * UKTask.h
 *
 *  Created on: 2017. 11. 11.
 *      Author: jej
 */

#ifndef SRC_KERNEL_INCLUDE_UKTASK_H_
#define SRC_KERNEL_INCLUDE_UKTASK_H_

#include <uem_common.h>

#include <uem_data.h>

#ifdef __cplusplus
extern "C"
{
#endif


typedef enum _EInternalTaskState {
	INTERNAL_STATE_RUN,
	INTERNAL_STATE_STOP,
	INTERNAL_STATE_WAIT,
	INTERNAL_STATE_END,
} EInternalTaskState;

uem_result UKTask_Initialize();
void UKTask_Finalize();

uem_result UKTask_RunTask (IN int nCallerTaskId, IN char *pszTaskName);
uem_result UKTask_StopTask (IN int nCallerTaskId, IN char *pszTaskName, IN uem_bool bDelayedStop);
uem_result UKTask_CallTask (IN int nCallerTaskId, IN char *pszTaskName);
uem_result UKTask_GetTaskState(IN int nCallerTaskId, char *pszTaskName, EInternalTaskState *penTaskState);

#ifndef API_LITE
uem_result UKTask_SuspendTask (IN int nCallerTaskId, IN char *pszTaskName);
uem_result UKTask_ResumeTask (IN int nCallerTaskId, IN char *pszTaskName);
uem_result UKTask_SetThroughputConstraint (IN int nCallerTaskId, IN char *pszTaskName, IN char *pszValue, IN char *pszUnit);
#endif

// Common task API used by both constrained and unconstrained devices
uem_result UKTask_GetTaskFromTaskName(char *pszTaskName, OUT STask **ppstTask);
uem_result UKTask_GetTaskFromTaskId(int nTaskId, OUT STask **ppstTask);
uem_result UKTask_GetTaskByTaskNameAndCallerTask(STask *pstCallerTask, char *pszTaskName, OUT STask **ppstTask);
uem_bool UKTask_isParentTask(int nTaskId, int nParentTaskId);

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_INCLUDE_UKTASK_H_ */
