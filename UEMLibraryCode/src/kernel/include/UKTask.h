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

/**
 * @brief
 *
 * This function
 *
 * @return
 */
uem_result UKTask_Initialize();

/*
 * @brief
 *
 * This function
 *
 */
void UKTask_Finalize();

/**
 * @brief
 *
 * This function
 *
 * @param nCallerTaskId
 * @param pszTaskName
 *
 * @return
 */
uem_result UKTask_RunTask (IN int nCallerTaskId, IN char *pszTaskName);

/**
 * @brief
 *
 * This function
 *
 * @param nCallerTaskId
 * @param pszTaskName
 * @param bDelayedStop
 *
 * @return
 */
uem_result UKTask_StopTask (IN int nCallerTaskId, IN char *pszTaskName, IN uem_bool bDelayedStop);

/**
 * @brief
 *
 * This function
 *
 * @param nCallerTaskId
 * @param pszTaskName
 *
 * @return
 */
uem_result UKTask_CallTask (IN int nCallerTaskId, IN char *pszTaskName);

/**
 * @brief
 *
 * This function
 *
 * @param nCallerTaskId
 * @param pszTaskName
 * @param penTaskState [out]
 *
 * @return
 */
uem_result UKTask_GetTaskState(IN int nCallerTaskId, char *pszTaskName, OUT EInternalTaskState *penTaskState);

#ifndef API_LITE
/**
 * @brief
 *
 * This function
 *
 * @param nCallerTaskId
 * @param pszTaskName
 *
 * @return
 */
uem_result UKTask_SuspendTask (IN int nCallerTaskId, IN char *pszTaskName);

/**
 * @brief
 *
 * This function
 *
 * @param nCallerTaskId
 * @param pszTaskName
 *
 * @return
 */
uem_result UKTask_ResumeTask (IN int nCallerTaskId, IN char *pszTaskName);

/**
 * @brief
 *
 * This function
 *
 * @param nCallerTaskId
 * @param pszTaskName
 * @param pszValue
 * @param pszUnit
 *
 * @return
 */
uem_result UKTask_SetThroughputConstraint (IN int nCallerTaskId, IN char *pszTaskName, IN char *pszValue, IN char *pszUnit);
#endif

// Common task API used by both constrained and unconstrained devices
/**
 * @brief
 *
 * This function
 *
 * @param pszTaskName
 * @param ppstTask
 *
 * @return
 */
uem_result UKTask_GetTaskFromTaskName(char *pszTaskName, OUT STask **ppstTask);

/**
 * @brief
 *
 * This function
 *
 * @param nTaskId
 * @param[out] ppstTask
 * @return
 */
uem_result UKTask_GetTaskFromTaskId(int nTaskId, OUT STask **ppstTask);

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
 * @param nTaskId
 * @param nParentTaskId
 *
 * @return
 */
uem_bool UKTask_isParentTask(int nTaskId, int nParentTaskId);

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_INCLUDE_UKTASK_H_ */
