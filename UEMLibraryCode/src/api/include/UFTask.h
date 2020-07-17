/*
 * UFTask.h
 *
 *  Created on: 2017. 8. 11.
 *      Author: jej
 */

#ifndef SRC_API_INCLUDE_UFTASK_H_
#define SRC_API_INCLUDE_UFTASK_H_

#include <uem_common.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef enum _ETaskState {
	STATE_RUN,
	STATE_STOP,
	STATE_WAIT,
	STATE_END,
} ETaskState;

/**
 * @brief Return the corresponding integer parameter of the corresponding task.
 *
 * @param nCallerTaskId id of caller task.
 * @param pszTaskName task name with the corresponding parameter.
 * @param pszParamName parameter name.
 * @param[out] pnParamVal corresponding integer parameter value of corresponding task.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_INVALID_PARAM for invalid caller task id, task name, parameter name. \n
 * @ref ERR_UEM_ILLEGAL_CONTROL if task is not Control task and task tries to get parameter at other task. \n
 * @ref ERR_UEM_NO_DATA if no task exists corresponding to task id or task name. \n
 * @ref ERR_UEM_NOT_FOUND if parameter does not exists at target task.
 */
uem_result UFTask_GetIntegerParameter (IN int nCallerTaskId, IN char *pszTaskName, IN char *pszParamName, OUT int *pnParamVal);

/**
 * @brief Set the corresponding integer parameter of the corresponding task.
 *
 * @param nCallerTaskId id of caller task.
 * @param pszTaskName task name with the corresponding parameter.
 * @param pszParamName parameter name.
 * @param nParamVal the value to be assigned to the parameter.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_INVALID_PARAM for invalid caller task id, task name, parameter name. \n
 * @ref ERR_UEM_ILLEGAL_CONTROL if task is not Control task. \n
 * @ref ERR_UEM_NO_DATA if no task exists corresponding to task id or task name. \n
 * @ref ERR_UEM_NOT_FOUND if parameter does not exists at target task.
 */

uem_result UFTask_SetIntegerParameter (IN int nCallerTaskId, IN char *pszTaskName, IN char *pszParamName, IN int nParamVal);

/**
 * @brief Return the corresponding floating point parameter of the corresponding task.
 *
 * @param nCallerTaskId id of caller task.
 * @param pszTaskName task name with the corresponding parameter.
 * @param pszParamName parameter name.
 * @param[out] pdbParamVal corresponding floating point parameter value of corresponding task.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_INVALID_PARAM for invalid caller task id, task name, parameter name. \n
 * @ref ERR_UEM_ILLEGAL_CONTROL if task is not Control task and task tries to get parameter at other task. \n
 * @ref ERR_UEM_NO_DATA if no task exists corresponding to task id or task name. \n
 * @ref ERR_UEM_NOT_FOUND if parameter does not exists at target task.
 */
uem_result UFTask_GetFloatParameter (IN int nCallerTaskId, IN char *pszTaskName, IN char *pszParamName, OUT double *pdbParamVal);

/**
 * @brief Set the corresponding floating point parameter of the corresponding task.
 *
 * @param nCallerTaskId id of caller task.
 * @param pszTaskName task name with the corresponding parameter.
 * @param pszParamName parameter name.
 * @param dbParamVal the value to be assigned as the parameter value.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_INVALID_PARAM for invalid caller task id, task name, parameter name. \n
 * @ref ERR_UEM_ILLEGAL_CONTROL if task is not Control task. \n
 * @ref ERR_UEM_NO_DATA if no task exists corresponding to task id or task name. \n
 * @ref ERR_UEM_NOT_FOUND if parameter does not exists at target task.
 */
uem_result UFTask_SetFloatParameter (IN int nCallerTaskId, IN char *pszTaskName, IN char *pszParamName, IN double dbParamVal);

/**
 * @brief Return the task status information.
 *
 * if task contains subgraph, traverse and check sub task's states. \n
 * if one of them has TASK_STATE_RUNNING then returns running. \n
 * else if one of them has TASK_STATE_SUSPEND then returns wait. \n
 * else if one of them has TASK_STATE_STOPPING then returns end. \n
 * else, when all sub sub task's state is TASK_STATE_STOP, returns stop. \n
 *
 * penTaskState value is \n
 * 0 for STATE_RUN (running state) \n
 * 1 for STATE_STOP (stopped state) \n
 * 2 for STATE_WAIT (waiting state) \n
 * 3 for STATE_END (termination requesting state) \n
 *
 * @sa UKTask_GetTaskState.
 *
 * @param nCallerTaskId id of caller task.
 * @param pszTaskName task name to check state of.
 * @param[out] penTaskState task state value.
 *
 * @return
 *  @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *  @ref ERR_UEM_INVALID_PARAM for invalid @a nCallerTaskId or @a pszTaskName, or for NULL @a penTaskState. \n
 *  @ref ERR_UEM_NO_DATA if task corresponding to caller task id does not exists. \n
 *  @ref ERR_UEM_ILLEGAL_DATA if task is not static scheduled and includes subgraph of process Network. \n
 *  @ref ERR_UEM_INVALID_HANDLE for invalid CPUTaskManager handler(generated from translator). \n
 *  errors could be propagated while checking task state of the sub-tasks if current task includes subtasks.
 *  (unconstrained device) \n
 *  @ref ERR_UEM_ILLEGAL_CONTROL if Caller task is not control Task and target task it not caller task. \n
 *  (constrained device) \n
 *  @ref ERR_UEM_ILLEGAL_CONTROL if Caller task is not control Task.
 */
uem_result UFTask_GetState (IN int nCallerTaskId, IN char *pszTaskName, OUT ETaskState *penTaskState);

#ifndef API_LITE
/**
 * @brief Set a schedule that meets the entered throughput.
 *
 * @sa UKTask_SetThroughputConstraint.
 *
 * @param nCallerTaskId id of caller task.
 * @param pszTaskName task name to set throughput.
 * @param pszValue character array containing value.
 * @param pszUnit throughput unit(currently not used).
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_ILLEGAL_CONTROL if task is not Control task. \n
 * @ref ERR_UEM_INVALID_PARAM for invalid caller task id, task name, parameter name. \n
 * @ref ERR_UEM_NO_DATA if caller task id does not match to any task or target task name does not match to any. \n
 *
 */
uem_result UFTask_SetThroughput (IN int nCallerTaskId, IN char *pszTaskName, IN char *pszValue, IN char *pszUnit);

/**
 * @brief Get the mode of the task.
 *
 * could be used for MTM task or subtask of MTM task.
 *
 * @param nCallerTaskId id of caller task(currently not used).
 * @param pszTaskName task to get mode from.
 * @param[out] ppszModeName mode name.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_INVALID_PARAM for invalid task name. \n
 * @ref ERR_UEM_NO_DATA if task name does not match to any task. \n
 * @ref ERR_UEM_ILLEGAL_DATA if the task is not MTM task and parent task is not. \n
 * @ref ERR_UEM_NOT_FOUND if task is not static scheduled and nCurIteration value is not zero and invalid mode array index?.
 */
uem_result UFTask_GetCurrentModeName (IN int nCallerTaskId, IN char *pszTaskName, OUT char **ppszModeName);

/**
 * @brief Change parameter integer value used in MTM.
 *
  * could be used for MTM task or subtask of MTM task.
 *
 * @param nCallerTaskId id of caller task(currently not used).
 * @param pszTaskName  task name containing that parameter.
 * @param pszParamName parameter name.
 * @param nParamVal the value to be assigned to the parameter.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_INVALID_PARAM for invalid task name. \n
 * @ref ERR_UEM_NO_DATA if task name does not match to any task. \n
 * @ref ERR_UEM_ILLEGAL_DATA if the task is not MTM task and parent task is not. \n
 */
uem_result UFTask_SetModeIntegerParameter (IN int nCallerTaskId, IN char *pszTaskName, IN char *pszParamName, IN int nParamVal);

/**
 * @brief Perform mode transition in MTM.
 *
 * @param nCallerTaskId id of caller task(currently not used).
 * @param pszTaskName task name to perform mode transition.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_INVALID_PARAM for invalid task name. \n
 * @ref ERR_UEM_NO_DATA if task name does not match to any task. \n
 * @ref ERR_UEM_ILLEGAL_DATA if the task is not MTM task and parent task is not. \n
 *
 */
uem_result UFTask_UpdateMode (IN int nCallerTaskId, IN char *pszTaskName);

/**
 * @brief Set a period.
 *
 * @sa UKTask_SetPeriod.
 *
 * @param nCallerTaskId id of caller task.
 * @param pszTaskName task name to set period.
 * @param nValue integer containing value.
 * @param pszTimeUnit period unit.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_ILLEGAL_CONTROL if task is not Control task. \n
 * @ref ERR_UEM_INVALID_PARAM for invalid caller task id, task name, time unit. \n
 * @ref ERR_UEM_NO_DATA if caller task id does not match to any task or target task name does not match to any. \n
 *
 */
uem_result UFTask_SetPeriod (IN int nCallerTaskId, IN char *pszTaskName, IN int nValue, IN char *pszTimeUnit);

/**
 * @brief Update a mapping info.
 *
 * @sa UFTask_ChangeMappedCore.
 *
 * @param nCallerTaskId id of caller task.
 * @param pszTaskName task name to set period.
 * @param nNewLocalId core id to assign.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_ILLEGAL_CONTROL if task is not Control task. \n
 * @ref ERR_UEM_INVALID_PARAM for invalid caller task id, task name, time unit. \n
 * @ref ERR_UEM_NO_DATA if caller task id does not match to any task or target task name does not match to any. \n
 *
 */
uem_result UFTask_ChangeMappedCore (IN int nCallerTaskId, IN char *pszTaskName, IN int nNewLocalId);

#endif

#ifdef __cplusplus
}
#endif

#endif /* SRC_API_INCLUDE_UFTASK_H_ */
