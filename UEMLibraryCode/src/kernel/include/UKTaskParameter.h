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

uem_result UKTaskParameter_GetInteger (IN int nCallerTaskId, IN char *pszTaskName, IN char *pszParamName, OUT int *pnParamVal);
uem_result UKTaskParameter_SetInteger (IN int nCallerTaskId, IN char *pszTaskName, IN char *pszParamName, IN int nParamVal);
uem_result UKTaskParameter_GetFloat (IN int nCallerTaskId, IN char *pszTaskName, IN char *pszParamName, OUT double *pdbParamVal);
uem_result UKTaskParameter_SetFloat (IN int nCallerTaskId, IN char *pszTaskName, IN char *pszParamName, IN double dbParamVal);

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_INCLUDE_UKTASKPARAMETER_H_ */
