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

uem_result UFTask_GetIntegerParameter (IN char *pszTaskName, IN char *pszParamName, OUT int *pnParamVal);
uem_result UFTask_SetIntegerParameter (IN char *pszTaskName, IN char *pszParamName, IN int nParamVal);
uem_result UFTask_GetFloatParameter (IN char *pszTaskName, IN char *pszParamName, OUT double *pdbParamVal);
uem_result UFTask_SetFloatParameter (IN char *pszTaskName, IN char *pszParamName, IN double dbParamVal);
uem_result UFTask_SetThroughput (IN char *pszTaskName, IN char *pszValue, IN char *pszUnit);
uem_result UFTask_GetState (IN char *pszTaskName, OUT ETaskState *penTaskState);

uem_result UFTask_GetCurrentModeName (IN char *pszTaskName, OUT char **ppszModeName);
uem_result UFTask_SetModeIntegerParameter (IN char *pszTaskName, IN char *pszParamName, IN int nParamVal);
uem_result UFTask_UpdateMode (IN char *pszTaskName);

#ifdef __cplusplus
}
#endif

#endif /* SRC_API_INCLUDE_UFTASK_H_ */