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

uem_result UKTask_RunTask (IN char *pszTaskName);
uem_result UKTask_StopTask (IN char *pszTaskName, IN uem_bool bDelayedStop);
uem_result UKTask_SuspendTask (IN char *pszTaskName);
uem_result UKTask_ResumeTask (IN char *pszTaskName);
uem_result UKTask_CallTask (IN char *pszTaskName);
uem_result UKTask_GetTaskState(char *pszTaskName, EInternalTaskState *penTaskState);

uem_result UKTask_GetTaskFromTaskName(char *szTaskName, STask **ppstTask);
uem_result UKTask_GetTaskFromTaskId(int nTaskId, STask **ppstTask);
uem_result UKTask_TraverseAllTasks(FnTaskTraverse fnCallback, void *pUserData);

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_INCLUDE_UKTASK_H_ */
