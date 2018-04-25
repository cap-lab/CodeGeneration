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

void SYS_REQ_SET_THROUGHPUT(int nCallerTaskId, char *pszTaskName, char *pszValue, char *pszUnit);
long SYS_REQ_GET_PARAM_INT(int nCallerTaskId, char *pszTaskName, char *pszParamName);
void SYS_REQ_SET_PARAM_INT(int nCallerTaskId, char *pszTaskName, char *pszParamName, long lParamVal);
double SYS_REQ_GET_PARAM_FLOAT(int nCallerTaskId, char *pszTaskName, char *pszParamName);
void SYS_REQ_SET_PARAM_FLOAT(int nCallerTaskId, char *pszTaskName, char *pszParamName, double dbParamVal);
int SYS_REQ_CHECK_TASK_STATE(int nCallerTaskId, char *pszTaskName);

void SYS_REQ_EXECUTE_TRANSITION(int nCallerTaskId, char *pszTaskName);
void SYS_REQ_SET_MTM_PARAM_INT(int nCallerTaskId, char *pszTaskName, char *pszParamName, long lParamVal);

char *SYS_REQ_GET_MODE(int nCallerTaskId, char *pszTaskName);

#define SYS_REQ_GET_CURRENT_MODE_NAME(a) SYS_REQ_GET_MODE(THIS_TASK_ID, TASK_NAME)

#ifdef __cplusplus
}
#endif

#endif /* SRC_API_INCLUDE_UFTASK_DEPRECATED_H_ */
