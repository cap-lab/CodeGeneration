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
 * @brief (Deprecated)
 *
 * This function
 *
 * @param nCallerTaskId
 * @param pszTaskName
 * @param pszParamName
 *
 * @return
 */
long SYS_REQ_GET_PARAM_INT(int nCallerTaskId, char *pszTaskName, char *pszParamName);

/**
 * @brief (Deprecated)
 *
 * This function
 *
 * @param nCallerTaskId
 * @param pszTaskName
 * @param pszParamName
 * @param lParamVal
 *
 * @return
 */
void SYS_REQ_SET_PARAM_INT(int nCallerTaskId, char *pszTaskName, char *pszParamName, long lParamVal);

/**
 * @brief (Deprecated)
 *
 * This function
 *
 * @param nCallerTaskId
 * @param pszTaskName
 * @param pszParamName
 *
 * @return
 */
double SYS_REQ_GET_PARAM_FLOAT(int nCallerTaskId, char *pszTaskName, char *pszParamName);

/**
 * @brief (Deprecated)
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
void SYS_REQ_SET_PARAM_FLOAT(int nCallerTaskId, char *pszTaskName, char *pszParamName, double dbParamVal);

/**
 * @brief (Deprecated)
 *
 * This function
 *
 * @param nCallerTaskId
 * @param pszTaskName
 *
 * @return
 */
int SYS_REQ_CHECK_TASK_STATE(int nCallerTaskId, char *pszTaskName);

#ifndef API_LITE
/**
 * @brief (Deprecated)
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
void SYS_REQ_SET_THROUGHPUT(int nCallerTaskId, char *pszTaskName, char *pszValue, char *pszUnit);

/**
 * @brief (Deprecated)
 *
 * This function
 *
 * @param nCallerTaskId
 * @param pszTaskName
 *
 * @return
 */
void SYS_REQ_EXECUTE_TRANSITION(int nCallerTaskId, char *pszTaskName);

/**
 * @brief (Deprecated)
 *
 * This function
 *
 * @param nCallerTaskId
 * @param pszTaskName
 * @param pszParamName
 * @param lParamVal
 *
 * @return
 */
void SYS_REQ_SET_MTM_PARAM_INT(int nCallerTaskId, char *pszTaskName, char *pszParamName, long lParamVal);

/**
 * @brief (Deprecated)
 *
 * This function
 *
 * @param nCallerTaskId
 * @param pszTaskName
 *
 * @return
 */
char *SYS_REQ_GET_MODE(int nCallerTaskId, char *pszTaskName);

#define SYS_REQ_GET_CURRENT_MODE_NAME(a) SYS_REQ_GET_MODE(THIS_TASK_ID, TASK_NAME)
#endif

#ifdef __cplusplus
}
#endif

#endif /* SRC_API_INCLUDE_UFTASK_DEPRECATED_H_ */
