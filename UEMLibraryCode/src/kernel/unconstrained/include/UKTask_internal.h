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
 * @brief Get task structure by task name and caller task structure.
 *
 * This function retrieves a task structure by task name and caller task structure. \n
 * @a pszTaskName is pure task name not combined with parent task name.
 *
 * @param pstCallerTask the caller task's (task which calls this function) task structure.
 * @param pszTaskName task name to get task structure (pure task name not combined with parent task name).
 * @param[out] ppstTask retrieved task structure.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_NO_DATA. \n
 *         @ref ERR_UEM_NO_DATA can be occurred if there is no task in the task graph.
 */
uem_result UKTask_GetTaskByTaskNameAndCallerTask(STask *pstCallerTask, char *pszTaskName, OUT STask **ppstTask);

/**
 * @brief Get task structure by task name.
 *
 * This function retrieves a task structure by full task name. \n
 * @a pszTaskName is full task name combined with parent task name.
 *
 * @param pszTaskName full task name to search.
 * @param[out] ppstTask retrieved task structure.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_NO_DATA. \n
 *         @ref ERR_UEM_NO_DATA can be occurred if there is no task in application.
 */
uem_result UKTask_GetTaskFromTaskName(char *pszTaskName, OUT STask **ppstTask);

/**
 * @brief Get task structure by task ID.
 *
 * This function retrieves a task structure by task ID.
 *
 * @param nTaskId task ID to search.
 * @param[out] ppstTask retrieved task structure.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_NO_DATA. \n
 *         @ref ERR_UEM_NO_DATA can be occurred if there is no task in application.
 */
uem_result UKTask_GetTaskFromTaskId(int nTaskId, STask **ppstTask);

/**
 * @brief Traverse all tasks.
 *
 * This function traverses all tasks.
 *
 * @param fnCallback callback function during traversing all tasks.
 * @param pUserData user data used in the callback function.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_NO_DATA, and corresponding results from the callback function. \n
 *         @ref ERR_UEM_NO_DATA can be occurred if there is no task in application.
 */
uem_result UKTask_TraverseAllTasks(FnTaskTraverse fnCallback, void *pUserData);

/**
 * @brief Clear iteration-related information in the task structure.
 *
 * This function clears iteration-related information in the task structure.
 *
 * @param pstTask target task structure to clear iteration information.
 *
 * @return This function returns @ref ERR_UEM_NOERROR.
 */
uem_result UKTask_ClearRunCount(STask *pstTask);

/**
 * @brief Increase iteration in the task structure.
 *
 * This function increases the iteration number in the task structure. \n
 * It also checks the new iteration number reaches the target iteration number. \n
 * It also increases not only task iteration number but also thread iteration number.
 *
 * @param pstTask target task structure to increase iteration.
 * @param nThreadId thread ID of target task.
 * @param[out] pbTargetIterationReached increased iteration reaches a target iteration or not.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_ILLEGAL_DATA, @ref ERR_UEM_NOT_FOUND. \n
 *         @ref ERR_UEM_ILLEGAL_DATA can be occurred if the MTM task has no corresponding modes. \n
 *         @ref ERR_UEM_NOT_FOUND can be occurred if when the corresponding mode index for the MTM task cannot be found.
 */
uem_result UKTask_IncreaseRunCount(STask *pstTask, int nThreadId, OUT uem_bool *pbTargetIterationReached);

/**
 * @brief Check the task reaches the target iteration.
 *
 * This function checks iteration run count.
 *
 * @param pstTask target task structure to check iteration.
 * @param nThreadId thread ID of target task.
 * @param[out] pbTargetIterationReached current iteration reaches a target iteration or not.
 *
 * @return This function returns @ref ERR_UEM_NOERROR.
 */
uem_result UKTask_CheckIterationRunCount(STask *pstTask, int nThreadId, OUT uem_bool *pbTargetIterationReached);

/**
 * @brief Set target iteration in the task structure.
 *
 * This function sets target iteration in the task structure. \n
 * @a nTargetIteration is based from @a nTargetTaskId. \n
 * If the @a nTargetTaskId is a parent task, the iteration number actually set to the task is different.
 *
 * @param pstTask target task structure to set target iteration.
 * @param nTargetIteration iteration number to set.
 * @param nTargetTaskId ID of task itself or parent task.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_ILLEGAL_DATA, @ref ERR_UEM_NOT_FOUND. \n
 *         @ref ERR_UEM_ILLEGAL_DATA can be occurred if the MTM task has no corresponding modes. \n
 *         @ref ERR_UEM_NOT_FOUND can be occurred if when the corresponding mode index for the MTM task cannot be found. \n
 *         @ref ERR_UEM_NOT_FOUND can also be happened if @a nTargetTaskId is not found in the @a pstTask or its parent tasks.
 */
uem_result UKTask_SetTargetIteration(STask *pstTask, int nTargetIteration, int nTargetTaskId);

/**
 * @brief Set target iteration to all tasks.
 *
 * This function sets all target iteration. \n
 * @a nTargetIteration is based on the top of the task graph.
 *
 * @param nTargetIteration target iteration number to set.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned -  @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_ILLEGAL_DATA, @ref ERR_UEM_NOT_FOUND. \n
 *         @ref ERR_UEM_ILLEGAL_DATA can be occurred if the MTM task has no corresponding modes. \n
 *         @ref ERR_UEM_NOT_FOUND can be occurred if when the corresponding mode index for the MTM task cannot be found. \n
 *         @ref ERR_UEM_NOT_FOUND can also be happened if @a nTargetTaskId is not found in the @a pstTask or its parent tasks.
 */
uem_result UKTask_SetAllTargetIteration(int nTargetIteration);

uem_result UKTask_GetTaskIteration(STask *pstTask, OUT int *pnTaskIteration);
uem_result UKTask_GetIterationNumberBasedOnTargetParentTaskId(STask *pstTask, int nIterationNumber, int nTargetTaskId, OUT int *pnConvertedIterationNumber);
uem_result UKTask_ConvertIterationToUpperTaskGraphBase(STask *pstTask, STaskGraph *pstTaskGraph, OUT int *pnConvertedIteration);
uem_result UKTask_UpdateAllSubGraphCurrentIteration(STaskGraph *pstTaskGraph, STask *pstLeafTask, int nNewIterationNumber);

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKTASK_INTERNAL_H_ */
