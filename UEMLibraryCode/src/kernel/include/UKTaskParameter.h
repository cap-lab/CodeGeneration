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
 * @brief
 *
 * This function
 *
 * @param nCallerTaskId
 * @param pszTaskName
 * @param pszParamName
 * @param pnParamVal [out]
 *
 * @return
 */
uem_result UKTaskParameter_GetInteger (IN int nCallerTaskId, IN char *pszTaskName, IN char *pszParamName, OUT int *pnParamVal);

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
uem_result UKTaskParameter_SetInteger (IN int nCallerTaskId, IN char *pszTaskName, IN char *pszParamName, IN int nParamVal);

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
uem_result UKTaskParameter_GetFloat (IN int nCallerTaskId, IN char *pszTaskName, IN char *pszParamName, OUT double *pdbParamVal);

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
uem_result UKTaskParameter_SetFloat (IN int nCallerTaskId, IN char *pszTaskName, IN char *pszParamName, IN double dbParamVal);

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_INCLUDE_UKTASKPARAMETER_H_ */
