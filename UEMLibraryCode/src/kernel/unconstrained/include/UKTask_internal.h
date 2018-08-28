/*
 * UKTask_internal.h
 *
 *  Created on: 2018. 8. 27.
 *      Author: jej
 */

#ifndef SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKTASK_INTERNAL_H_
#define SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKTASK_INTERNAL_H_

#include <uem_common.h>

#include <uem_data.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef uem_result (*FnTaskTraverse)(STask *pstTask, void *pUserData);

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

#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKTASK_INTERNAL_H_ */
