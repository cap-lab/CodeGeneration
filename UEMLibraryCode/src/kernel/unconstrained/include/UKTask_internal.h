/*
 * UKTask_internal.h
 *
 *  Created on: 2018. 8. 27.
 *      Author: jej
 */

#ifndef SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKTASK_INTERNAL_H_
#define SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKTASK_INTERNAL_H_

#include <uem_common.h>
#include <uem_enum.h>

#include <uem_data.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef uem_result (*FnTaskTraverse)(STask *pstTask, void *pUserData);

/**
 * @brief
 *
 * This function
 *
 * @param pstCallerTask
 * @param pszTaskName
 * @param[out] ppstTask
 *
 * @return
 */
uem_result UKTask_GetTaskByTaskNameAndCallerTask(STask *pstCallerTask, char *pszTaskName, OUT STask **ppstTask);

/**
 * @brief
 *
 * This function
 *
 * @param szTaskName
 * @param[out] ppstTask
 *
 * @return
 */
uem_result UKTask_GetTaskFromTaskName(char *szTaskName, STask **ppstTask);

/**
 * @brief
 *
 * This function
 *
 * @param nTaskId
 * @param[out] ppstTask
 *
 * @return
 */
uem_result UKTask_GetTaskFromTaskId(int nTaskId, STask **ppstTask);

/**
 * @brief
 *
 * This function
 *
 * @param fnCallback
 * @param pUserData
 *
 * @return
 */
uem_result UKTask_TraverseAllTasks(FnTaskTraverse fnCallback, void *pUserData);

/**
 * @brief
 *
 * This function
 *
 * @param pstTask
 *
 * @return
 */
uem_result UKTask_ClearRunCount(STask *pstTask);

/**
 * @brief
 *
 * This function
 *
 * @param pstTask
 * @param nThreadId
 * @param[out] pbTargetIterationReached
 *
 * @return
 */
uem_result UKTask_IncreaseRunCount(STask *pstTask, int nThreadId, OUT uem_bool *pbTargetIterationReached);

/**
 * @brief
 *
 * This function
 *
 * @param pstTask
 * @param nThreadId
 * @param[out] pbTargetIterationReached
 *
 * @return
 */
uem_result UKTask_CheckIterationRunCount(STask *pstTask, int nThreadId, OUT uem_bool *pbTargetIterationReached);

/**
 * @brief
 *
 * This function
 *
 * @param pstTask
 * @param nTargetIteration
 * @param nTargetTaskId
 *
 * @return
 */
uem_result UKTask_SetTargetIteration(STask *pstTask, int nTargetIteration, int nTargetTaskId);

/**
 * @brief
 *
 * This function
 *
 * @param nTargetIteration
 *
 * @return
 */
uem_result UKTask_SetAllTargetIteration(int nTargetIteration);

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKTASK_INTERNAL_H_ */
