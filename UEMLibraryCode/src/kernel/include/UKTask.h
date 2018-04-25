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

typedef uem_result (*FnTaskTraverse)(STask *pstTask, void *pUserData);

uem_result UKTask_Initialize();
void UKTask_Finalize();

uem_result UKTask_RunTask (IN int nCallerTaskId, IN char *pszTaskName);
uem_result UKTask_StopTask (IN int nCallerTaskId, IN char *pszTaskName, IN uem_bool bDelayedStop);
uem_result UKTask_SuspendTask (IN int nCallerTaskId, IN char *pszTaskName);
uem_result UKTask_ResumeTask (IN int nCallerTaskId, IN char *pszTaskName);
uem_result UKTask_CallTask (IN int nCallerTaskId, IN char *pszTaskName);
uem_result UKTask_GetTaskState(IN int nCallerTaskId, char *pszTaskName, EInternalTaskState *penTaskState);
uem_result UKTask_SetThroughputConstraint (IN int nCallerTaskId, IN char *pszTaskName, IN char *pszValue, IN char *pszUnit);

uem_result UKTask_GetTaskByTaskNameAndCallerTask(STask *pstCallerTask, char *pszTaskName, OUT STask **ppstTask);
uem_bool UKTask_isParentTask(int nTaskId, int nParentTaskId);
uem_result UKTask_GetTaskFromTaskName(char *szTaskName, STask **ppstTask);
uem_result UKTask_GetTaskFromTaskId(int nTaskId, STask **ppstTask);
uem_result UKTask_TraverseAllTasks(FnTaskTraverse fnCallback, void *pUserData);
uem_result UKTask_ClearRunCount(STask *pstTask);
uem_result UKTask_IncreaseRunCount(STask *pstTask, uem_bool *pbTargetIterationReached);
uem_result UKTask_CheckIterationRunCount(STask *pstTask, OUT uem_bool *pbTargetIterationReached);
uem_result UKTask_SetTargetIteration(STask *pstTask, int nTargetIteration, int nTargetTaskId);
uem_result UKTask_SetAllTargetIteration(int nTargetIteration);

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_INCLUDE_UKTASK_H_ */
