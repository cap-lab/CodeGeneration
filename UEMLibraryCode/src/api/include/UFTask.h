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
 * @brief
 *
 * This function
 *
 * @param nCallerTaskId
 * @param pszTaskName
 * @param pszParamName
 * @param[out] pnParamVal
 *
 * @return
 */
uem_result UFTask_GetIntegerParameter (IN int nCallerTaskId, IN char *pszTaskName, IN char *pszParamName, OUT int *pnParamVal);

/**
 * @brief
 *
 * This function
 *
 * @param nCallerTaskId
 * @param pszTaskName
 * @param pszParamName
 * @param nParamVal
 *
 * @return
 */
uem_result UFTask_SetIntegerParameter (IN int nCallerTaskId, IN char *pszTaskName, IN char *pszParamName, IN int nParamVal);

/**
 * @brief
 *
 * This function
 *
 * @param nCallerTaskId
 * @param pszTaskName
 * @param pszParamName
 * @param[out] pdbParamVal
 *
 * @return
 */
uem_result UFTask_GetFloatParameter (IN int nCallerTaskId, IN char *pszTaskName, IN char *pszParamName, OUT double *pdbParamVal);

/**
 * @brief
 *
 * This function
 *
 * @param nCallerTaskId
 * @param pszTaskName
 * @param pszParamName
 * @param dbParamVal
 *
 * @return
 */
uem_result UFTask_SetFloatParameter (IN int nCallerTaskId, IN char *pszTaskName, IN char *pszParamName, IN double dbParamVal);

/**
 * @brief
 *
 * This function
 *
 * @param nCallerTaskId
 * @param pszTaskName
 * @param[out] penTaskState
 *
 * @return
 */
uem_result UFTask_GetState (IN int nCallerTaskId, IN char *pszTaskName, OUT ETaskState *penTaskState);

#ifndef API_LITE
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
uem_result UFTask_SetThroughput (IN int nCallerTaskId, IN char *pszTaskName, IN char *pszValue, IN char *pszUnit);

/**
 * @brief
 *
 * This function
 *
 * @param nCallerTaskId
 * @param pszTaskName
 * @param[out] ppszModeName
 *
 * @return
 */
uem_result UFTask_GetCurrentModeName (IN int nCallerTaskId, IN char *pszTaskName, OUT char **ppszModeName);

/**
 * @brief
 *
 * This function
 *
 * @param nCallerTaskId
 * @param pszTaskName
 * @param pszParamName
 * @param nParamVal
 *
 * @return
 */
uem_result UFTask_SetModeIntegerParameter (IN int nCallerTaskId, IN char *pszTaskName, IN char *pszParamName, IN int nParamVal);

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
uem_result UFTask_UpdateMode (IN int nCallerTaskId, IN char *pszTaskName);
#endif

#ifdef __cplusplus
}
#endif

#endif /* SRC_API_INCLUDE_UFTASK_H_ */
