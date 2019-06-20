/*
 * UKTaskParameter.h
 *
 *  Created on: 2018. 2. 12.
 *      Author: jej
 */

#ifndef SRC_KERNEL_INCLUDE_UKTASKPARAMETER_H_
#define SRC_KERNEL_INCLUDE_UKTASKPARAMETER_H_

#include <uem_common.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief Return the corresponding integer parameter of the corresponding task.
 *
 * This function returns integer parameter value from target task.
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
uem_result UKTaskParameter_GetInteger (IN int nCallerTaskId, IN char *pszTaskName, IN char *pszParamName, OUT int *pnParamVal);

/**
 * @brief Set the corresponding integer parameter of the corresponding task.
 *
 * This function set integer parameter value at target task.
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
uem_result UKTaskParameter_SetInteger (IN int nCallerTaskId, IN char *pszTaskName, IN char *pszParamName, IN int nParamVal);

/**
  @brief Return the corresponding floating point parameter of the corresponding task.
 *
 * This function returns floating point parameter value from target task.
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
uem_result UKTaskParameter_GetFloat (IN int nCallerTaskId, IN char *pszTaskName, IN char *pszParamName, OUT double *pdbParamVal);

/**
 * @brief Set the corresponding floating point parameter of the corresponding task.
 *
 * This function set floating point parameter value at target task.
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
uem_result UKTaskParameter_SetFloat (IN int nCallerTaskId, IN char *pszTaskName, IN char *pszParamName, IN double dbParamVal);

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_INCLUDE_UKTASKPARAMETER_H_ */
