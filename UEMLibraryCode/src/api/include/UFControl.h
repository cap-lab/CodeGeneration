/*
 * UFControl.h
 *
 *  Created on: 2017. 8. 11.
 *      Author: jej
 */

#ifndef SRC_API_INCLUDE_UFCONTROL_H_
#define SRC_API_INCLUDE_UFCONTROL_H_

#include <uem_common.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief Execute task.
 *
 * The task keeps running all the time.
 *
 * @param nCallerTaskId id of caller task.
 * @param pszTaskName target task name to be run.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_INVALID_PARAM for invalid Caller Task or target task name. \n
 * @ref ERR_UEM_ILLEGAL_CONTROL if caller task is not a Control task. \n
 * @ref ERR_UEM_ILLEGAL_DATA if subgraph containing target task is Process Network type. \n

 *\n
 * Functions that may propagate error results : \n
 * (constrained kernel code) UKTaskControl_RunTask \n
 * (unconstrained kernel code) UKCPUTaskManager_RunTask \n
 * \n
 *
 */
uem_result UFControl_RunTask (IN int nCallerTaskId, IN char *pszTaskName);

/**
  * @brief Stop task execution.
 *
 * The task could be continued by @ref UFControl_RunTask.
 *
 * @param nCallerTaskId id of caller task.
 * @param pszTaskName target task name to be stopped.
 * @param bDelayedStop task is terminated after The job currently being run is done.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_INVALID_PARAM for invalid caller task id or target task name.
 * @ref ERR_UEM_ILLEGAL_CONTROL, \n
 * (unconstrained kernel code) when bDelayedStop is True, task type is Computational,
 * parents task is null and taskId is not matched to the id found by name. \n
 * (constrained kernel code) when Caller task is not Control Task. \n
 *
 *\n
 * Functions that may propagate error results : \n
 * @ref UKCPUTaskManager_StopTask \n
 * @ref UKCPUTaskManager_StoppingTask \n
 *  \n
 */
uem_result UFControl_StopTask (IN int nCallerTaskId, IN char *pszTaskName, IN uem_bool bDelayedStop);

/**
 * @brief Execute a task only once.
 *
 * TASK_INIT and TASK_WRAPUP are also called when executing.
 *
 * @param nCallerTaskId id of caller task.
 * @param pszTaskName target task name to be stopped.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_INVALID_PARAM for invalid caller task id or target task name.\n
 * @ref ERR_UEM_ILLEGAL_CONTROL, \n
 * (constrained kernel code) when Caller task is not Control Task. \n
 */
uem_result UFControl_CallTask (IN int nCallerTaskId, IN char *pszTaskName);

#ifndef API_LITE
/**
 * @brief Suspend a task.
 *
 * The task could be continued by @ref UFControl_ResumeTask.
 *
 * @param nCallerTaskId id of caller task.
 * @param pszTaskName target task to be suspended.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_INVALID_PARAM for invalid caller task id or target task name. \n
 * @ref ERR_UEM_ILLEGAL_CONTROL when Caller task is not Control Task. \n
 */

uem_result UFControl_SuspendTask (IN int nCallerTaskId, IN char *pszTaskName);

/**
 * @brief Resume suspended task.
 *
 * @param nCallerTaskId id of caller task.
 * @param pszTaskName target task to be resumed.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_INVALID_PARAM for invalid caller task id or target task name. \n
 * @ref ERR_UEM_ILLEGAL_CONTROL when Caller task is not Control Task. \n

 * \n
 *  Functions that may propagate error results \n
 * @ref UKCPUTaskManager_ResumeTask \n
 * \n
 *
 */
uem_result UFControl_ResumeTask (IN int nCallerTaskId, IN char *pszTaskName);
#endif

#ifdef __cplusplus
}
#endif

#endif /* SRC_API_INCLUDE_UFCONTROL_H_ */
