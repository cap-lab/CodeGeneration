/*
 * UKModeTransition.h
 *
 *  Created on: 2017. 8. 28.
 *      Author: jej
 */

#ifndef SRC_KERNEL_INCLUDE_UKMODETRANSITION_H_
#define SRC_KERNEL_INCLUDE_UKMODETRANSITION_H_

#include <uem_common.h>

#ifdef __cplusplus
extern "C"
{
#endif

uem_result UKModeTransition_Initialize();
uem_result UKModeTransition_Finalize();

uem_result UKModeTransition_GetCurrentModeName (IN char *pszTaskName, OUT char **ppszModeName);
uem_result UKModeTransition_SetModeIntegerParameter (IN char *pszTaskName, IN char *pszParamName, IN int nParamVal);
uem_result UKModeTransition_UpdateMode (IN char *pszTaskName);

int UKModeTransition_GetModeIndexByModeId(SModeTransitionMachine *pstModeTransition, int nModeId);
int UKModeTransition_GetVariableIndexByName(SModeTransitionMachine *pstModeTransition, char *pszVariableName);


#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_INCLUDE_UKMODETRANSITION_H_ */
