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
 * @brief Perform task initialize operation.
 *
 * This function performs task initialize operation.
 * Do nothing for constrained device.
 * (unconstrained device) This function creates task mutex and task event handler.
 *
 * @return
 *  @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *  Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_OUT_OF_MEMORY, @ref ERR_UEM_MUTEX_ERROR when creating mutex and event handler.
 */
uem_result UKTask_Initialize();

/*
 * @brief Perform task finalize operation.
 *
 * This function performs task finalize operation.
 * Do nothing for constrained device.
 * (unconstrained device) This function destroys task mutex and task event handler.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * Errors to be returned - @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_MUTEX_ERROR when destroying mutex and event handler.
 */
void UKTask_Finalize();

/**
 * @brief Execute task.
 *
 * This function makes target task to be run. \n
 * The task keeps running all the time. \n
 * this function either calls UKTaskControl_RunTask function if used in constrained device, \n
 * or calls UKCPUTaskManager_RunTask function if used in unconstrained device.
 *
 * @param nCallerTaskId id of caller task.
 * @param pszTaskName target task name to be run.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_INVALID_PARAM for invalid @a nCallerTaskId or @a pszTaskName. \n
 * @ref ERR_UEM_ILLEGAL_CONTROL if caller task is not a Control task. \n
 * @ref ERR_UEM_ILLEGAL_DATA if subgraph containing target task is Process Network type.
 *
 * \n
 * Functions that propagates error results : \n
 * UKTaskControl_RunTask in constrained device. \n
 * UKCPUTaskManager_RunTask in unconstrained device. \n
 * \n
 */
uem_result UKTask_RunTask (IN int nCallerTaskId, IN char *pszTaskName);

/**
  * @brief Stop task execution.
 *
 * This function stops task execution. \n
 * The task could be continued by @ref UKTask_RunTask.
 *
 * @param nCallerTaskId id of caller task.
 * @param pszTaskName target task name to be stopped.
 * @param bDelayedStop task is terminated after The job currently being run is done.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_INVALID_PARAM for invalid @a nCallerTaskId or @a pszTaskName. \n
 * (unconstrained device) \n
 * @ref ERR_UEM_ILLEGAL_CONTROL if bDelayedStop is TRUE and task type is Computational,
 * parents task is null, and taskId is not matched to the id found by name.
 * or bDelayedStop is FALSE and task is not control task. \n
 * parents task is null and taskId is not matched to the id found by name. \n
 * (constrained device) \n
 * @ref ERR_UEM_ILLEGAL_CONTROL if Caller task is not Control Task.
 *
 *\n
 * Functions that may propagate error results : \n
 * UKCPUTaskManager_StopTask \n
 * UKCPUTaskManager_StoppingTask \n
 *  \n
 */
uem_result UKTask_StopTask (IN int nCallerTaskId, IN char *pszTaskName, IN uem_bool bDelayedStop);

/**
 * @brief Execute a task only once.
 *
 * This function executes a task only once. \n
 * TASK_INIT and TASK_WRAPUP are also called when executing.
 *
 * @param nCallerTaskId id of caller task.
 * @param pszTaskName target task name to be stopped.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_INVALID_PARAM for invalid @a nCallerTaskId or @a pszTaskName.\n
 * (constrained device) \n
 * @ref ERR_UEM_ILLEGAL_CONTROL if Caller task is not control Task.
 */
uem_result UKTask_CallTask (IN int nCallerTaskId, IN char *pszTaskName);

/**
 * @brief Return task status information.
 *
 * This function returns task status information. \n
 * if task contains subgraph, traverse and check sub task's states. \n
 * if one of them has TASK_STATE_RUNNING then returns running. \n
 * else if one of them has TASK_STATE_SUSPEND then returns wait. \n
 * else if one of them has TASK_STATE_STOPPING then returns end. \n
 * else, when all sub sub task's state is TASK_STATE_STOP, returns stop. \n
 *
 * penTaskState value is \n
 * 0 for INTERNAL_STATE_RUN (running state) \n
 * 1 for INTERNAL_STATE_STOP (stopped state) \n
 * 2 for INTERNAL_STATE_WAIT (waiting state) \n
 * 3 for INTERNAL_STATE_END (termination requesting state) \n
 *
 * @param nCallerTaskId id of caller task.
 * @param pszTaskName task name to check state of.
 * @param[out] penTaskState task state value.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_INVALID_PARAM for invalid @a nCallerTaskId, @a pszTaskName, or for NULL @a penTaskState. \n
 * @ref ERR_UEM_NO_DATA if task corresponding to caller task id does not exists. \n
 * @ref ERR_UEM_ILLEGAL_DATA if task is not static scheduled and includes subgraph of process Network. \n
 * @ref ERR_UEM_INVALID_HANDLE for invalid CPUTaskManager handler(generated from translator). \n
 * errors could be propagated while checking task state of the sub-tasks if current task includes subtasks.
 * (unconstrained device) \n
 * @ref ERR_UEM_ILLEGAL_CONTROL if Caller task is not control Task and target task it not caller task. \n
 * (constrained device) \n
 * @ref ERR_UEM_ILLEGAL_CONTROL if Caller task is not control Task.
 */
uem_result UKTask_GetTaskState(IN int nCallerTaskId, char *pszTaskName, OUT EInternalTaskState *penTaskState);

#ifndef API_LITE
/**
 * @brief Suspend a task.
 *
 * This function makes a task to be suspended. \n
 * The suspended task could be continued by @ref UKTask_ResumeTask.
 *
 * @param nCallerTaskId id of caller task.
 * @param pszTaskName target task to be suspended.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_INVALID_PARAM for invalid @a nCallerTaskId or @a pszTaskName. \n
 * @ref ERR_UEM_ILLEGAL_CONTROL when Caller task is not Control Task.
 */
uem_result UKTask_SuspendTask (IN int nCallerTaskId, IN char *pszTaskName);

/**
 * @brief Resume suspended task.
 *
 * This function resumes suspended task.
 *
 * @param nCallerTaskId id of caller task.
 * @param pszTaskName target task to be resumed.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_INVALID_PARAM for invalid @a nCallerTaskId or @a pszTaskName. \n
 * @ref ERR_UEM_ILLEGAL_CONTROL when Caller task is not Control Task.

 * \n
 *  Functions that may propagate error results \n
 * @ref UKCPUTaskManager_ResumeTask \n
 * \n
 *
 */
uem_result UKTask_ResumeTask (IN int nCallerTaskId, IN char *pszTaskName);

/**
 * @brief Set a schedule that meets the entered throughput.
 *
 * This function sets a schedule that meets the entered throughput.
 *
 * @param nCallerTaskId id of caller task.
 * @param pszTaskName task name to set throughput.
 * @param pszValue character array containing value.
 * @param pszUnit throughput unit(currently not used).
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_INVALID_PARAM for invalid @a nCallerTaskId, @a pszTaskName, @a pszValue, @a pszUnit. \n
 * @ref ERR_UEM_ILLEGAL_CONTROL if task is not Control task. \n
 * @ref ERR_UEM_NO_DATA if caller task id does not match to any task or target task name does not match to any.
 *
 */
uem_result UKTask_SetThroughputConstraint (IN int nCallerTaskId, IN char *pszTaskName, IN char *pszValue, IN char *pszUnit);
#endif

// Common task API used by both constrained and unconstrained devices
/**
 * @brief Get task pointer by task name.
 *
 * This function returns task pointer by task name.
 *
 * @param pszTaskName target task name to get task pointer of.
 * @param ppstTask pointer to traget task pointer.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_INVALID_PARAM for invalid @a pszTaskName or @a ppstTask. \n
 * @ref ERR_UEM_NO_DATA if no task exists corresponding to task name.
 */
uem_result UKTask_GetTaskFromTaskName(char *pszTaskName, OUT STask **ppstTask);

/**
 * @brief Get task pointer by task name.
 *
 * This function returns task pointer by task id.
 *
 * @param nTaskId target task id to get task pointer from.
 * @param[out] ppstTask pointer to traget task pointer.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_INVALID_PARAM for invalid @a nTaskId or @a ppstTask. \n
 * @ref ERR_UEM_NO_DATA if no task exists corresponding to task id.
 */
uem_result UKTask_GetTaskFromTaskId(int nTaskId, OUT STask **ppstTask);

/**
 * @brief Get task pointer by target task name and caller task.
 *
 * This function returns task pointer by target task name and calller task name.
 *
 * @param pstCallerTask pointer to caller task.
 * @param pszTaskName target task name to get task pointer of.
 * @param[out] ppstTask pointer to target task pointer.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_INVALID_PARAM for invalid @a nCallerTaskId, @a pszTaskName or @a ppstTask. \n
 * @ref ERR_UEM_NO_DATA if target task does not exist in the task graph to which the caller task belongs.
 */
uem_result UKTask_GetTaskByTaskNameAndCallerTask(STask *pstCallerTask, char *pszTaskName, OUT STask **ppstTask);

/**
 * @brief Check whether a task is a parent task of another task.
 *
 * This function checks whether a task is a parent task of another task by given task id.
 *
 * @param nTaskId task id to check if it is a subtask of other one.
 * @param nParentTaskId task id to check if it is a parent task of other one.
 *
 * @return TRUE if one is a parent task of another task, FALSE otherwise.
 */
uem_bool UKTask_isParentTask(int nTaskId, int nParentTaskId);

/**
 * @brief Set a period.
 *
 * This function sets a period of the task.
 *
 * @param nCallerTaskId id of caller task.
 * @param pszTaskName task name to set period.
 * @param nValue integer containing value.
 * @param pszTimeUnit time unit for the period.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_INVALID_PARAM for invalid @a nCallerTaskId, @a pszTaskName, @a pszValue. \n
 * @ref ERR_UEM_ILLEGAL_CONTROL if task is not Control task. \n
 * @ref ERR_UEM_NO_DATA if caller task id does not match to any task or target task name does not match to any.
 *
 */
uem_result UKTask_SetPeriod (IN int nCallerTaskId, IN char *pszTaskName, IN int nValue, IN char *pszTimeUnit);

/**
 * @brief Update a mapping info of the task.
 *
 * This function updates a mapping info of the task.
 *
 * @param nCallerTaskId id of caller task.
 * @param pszTaskName task name to update the mapping info.
 * @param nNewLocalId local core id to assign newly.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_INVALID_PARAM for invalid @a nCallerTaskId, @a pszTaskName, @a pszValue. \n
 * @ref ERR_UEM_ILLEGAL_CONTROL if task is not Control task. \n
 * @ref ERR_UEM_NO_DATA if caller task id does not match to any task or target task name does not match to any.
 *
 */
uem_result UKTask_ChangeMappedCore (IN int nCallerTaskId, IN char *pszTaskName, IN int nNewLocalId);

/**
 * @brief Update a mapping set of the task and and its subtasks.
 *
 * This function updates a mapping set of the task and and its subtasks.
 *
 * @param nCallerTaskId id of caller task.
 * @param pszTaskName task name to update the mapping set.
 * @param pszMappingSet mapping set name to update.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_INVALID_PARAM for invalid @a nCallerTaskId, @a pszTaskName, @a pszValue. \n
 * @ref ERR_UEM_ILLEGAL_CONTROL if task is not Control task. \n
 * @ref ERR_UEM_NO_DATA if caller task id does not match to any task or target task name does not match to any.
 *
 */
uem_result UKTask_ChangeMappingSet(IN int nCallerTaskId, IN char *pszTaskName, IN const char *pszMappingSet);

/**
 * @brief Get the current mapping set of the task.
 *
 * This function gets a mapping set of the task.
 *
 * @param nCallerTaskId id of caller task.
 * @param pszTaskName task name to update the mapping set.
 * @param nBufferLen Buffer length.
 * @param ppszMappingSet mapping set name.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_INVALID_PARAM for invalid @a nCallerTaskId, @a pszTaskName, @a pszValue. \n
 * @ref ERR_UEM_ILLEGAL_CONTROL if task is not Control task. \n
 * @ref ERR_UEM_NO_DATA if caller task id does not match to any task or target task name does not match to any.
 * @ref ERR_UEM_REALLOCATE_BUFFER is occurred when the buffer length is not enough. \n
 *
 */
uem_result UKTask_GetCurrentMappingSet(IN int nCallerTaskId, IN char *pszTaskName, IN int nBufferLen, OUT char **ppszMappingSet);

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_INCLUDE_UKTASK_H_ */
