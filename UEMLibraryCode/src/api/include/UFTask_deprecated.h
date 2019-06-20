/*
 * UFTask_deprecated.h
 *
 *  Created on: 2017. 8. 13.
 *      Author: jej
 */

#ifndef SRC_API_INCLUDE_UFTASK_DEPRECATED_H_
#define SRC_API_INCLUDE_UFTASK_DEPRECATED_H_

#include <uem_common.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief (Deprecated) Return the corresponding integer parameter of the corresponding task.
 *
 * @param nCallerTaskId id of caller task.
 * @param pszTaskName task name with the corresponding parameter.
 * @param pszParamName parameter name.
 *
 * @return corresponding integer parameter value of corresponding task.
 */

long SYS_REQ_GET_PARAM_INT(int nCallerTaskId, char *pszTaskName, char *pszParamName);
/**
 * @brief (Deprecated) Set the corresponding integer parameter of the corresponding task.
 *
 * @param nCallerTaskId id of caller task.
 * @param pszTaskName task name with the corresponding parameter.
 * @param pszParamName parameter name.
 * @param lParamVal the value to be assigned to the parameter.
 *
 * @return
 */
void SYS_REQ_SET_PARAM_INT(int nCallerTaskId, char *pszTaskName, char *pszParamName, long lParamVal);

/**
 * @brief (Deprecated) Return the corresponding floating point parameter of the corresponding task.
 *
 * @param nCallerTaskId id of caller task.
 * @param pszTaskName task name with the corresponding parameter.
 * @param pszParamName parameter name.
 *
  * @return corresponding floating point parameter value of corresponding task.
 */
double SYS_REQ_GET_PARAM_FLOAT(int nCallerTaskId, char *pszTaskName, char *pszParamName);

/**
 * @brief (Deprecated) Set the corresponding floating point parameter of the corresponding task.
 *
 * @param nCallerTaskId id of caller task.
 * @param pszTaskName task name with the corresponding parameter.
 * @param pszParamName parameter name.
 * @param dbParamVal the value to be assigned as the parameter value.
 *
 * @return
 */
void SYS_REQ_SET_PARAM_FLOAT(int nCallerTaskId, char *pszTaskName, char *pszParamName, double dbParamVal);

/**
 * @brief (Deprecated) Return the task status information.
 *
 * if task contains subgraph, traverse and check sub task's states. \n
 * if one of them has TASK_STATE_RUNNING then returns running. \n
 * else if one of them has TASK_STATE_SUSPEND then returns wait. \n
 * else if one of them has TASK_STATE_STOPPING then returns end. \n
 * else, when all sub sub task's state is TASK_STATE_STOP, returns stop. \n
 *
 * @param nCallerTaskId id of caller task.
 * @param pszTaskName task name to check state of.
 *
 * @return
 * 0 for STATE_RUN (running state) \n
 * 1 for STATE_STOP (stopped state) \n
 * 2 for STATE_WAIT (waiting state) \n
 * 3 for STATE_END (termination requesting state) \n
 */
int SYS_REQ_CHECK_TASK_STATE(int nCallerTaskId, char *pszTaskName);

#ifndef API_LITE
/**
 * @brief (Deprecated) Set a schedule that meets the entered throughput.
 *
 * @param nCallerTaskId id of caller task.
 * @param pszTaskName task name to set throughput.
 * @param pszValue character array containing value.
 * @param pszUnit throughput unit(currently not used).
 *
 * @return
 */
void SYS_REQ_SET_THROUGHPUT(int nCallerTaskId, char *pszTaskName, char *pszValue, char *pszUnit);

/**
 * @brief (Deprecated) Get the mode of the task.
 *
 * @param nCallerTaskId id of caller task.
 * @param pszTaskName task to get mode from.
 *
 * @return mode name.
 */
char *SYS_REQ_GET_MODE(int nCallerTaskId, char *pszTaskName);

/**
 * @brief (Deprecated) Change parameter integer value used in MTM.
 *
 * @param nCallerTaskId id of caller task.
 * @param pszTaskName  task name containing that parameter.
 * @param pszParamName parameter name.
 * @param lParamVal the value to be assigned to the parameter.
 */

void SYS_REQ_SET_MTM_PARAM_INT(int nCallerTaskId, char *pszTaskName, char *pszParamName, long lParamVal);
/**
 * @brief (Deprecated) Perform mode transition in MTM.
 *
 * @param nCallerTaskId id of caller task.
 * @param pszTaskName task name to perform mode transition.
 *
 * @return
 */
void SYS_REQ_EXECUTE_TRANSITION(int nCallerTaskId, char *pszTaskName);

#define SYS_REQ_GET_CURRENT_MODE_NAME(a) SYS_REQ_GET_MODE(THIS_TASK_ID, TASK_NAME)
#endif

#ifdef __cplusplus
}
#endif

#endif /* SRC_API_INCLUDE_UFTASK_DEPRECATED_H_ */
