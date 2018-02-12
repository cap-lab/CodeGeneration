/*
 * UKTaskParameter.h
 *
 *  Created on: 2018. 2. 12.
 *      Author: jej
 */

#ifndef SRC_KERNEL_INCLUDE_UKTASKPARAMETER_H_
#define SRC_KERNEL_INCLUDE_UKTASKPARAMETER_H_

#include <uem_common.h>

#include <uem_data.h>

#ifdef __cplusplus
extern "C"
{
#endif

uem_result UKTask_GetIntegerParameter (IN char *pszTaskName, IN char *pszParamName, OUT int *pnParamVal);
uem_result UKTask_SetIntegerParameter (IN char *pszTaskName, IN char *pszParamName, IN int nParamVal);
uem_result UKTask_GetFloatParameter (IN char *pszTaskName, IN char *pszParamName, OUT double *pdbParamVal);
uem_result UKTask_SetFloatParameter (IN char *pszTaskName, IN char *pszParamName, IN double dbParamVal);

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_INCLUDE_UKTASKPARAMETER_H_ */
