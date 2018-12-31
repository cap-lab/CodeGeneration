/*
 * UKModeTransition.h
 *
 *  Created on: 2017. 8. 28.
 *      Author: jej
 */

#ifndef SRC_KERNEL_INCLUDE_UKMODETRANSITION_H_
#define SRC_KERNEL_INCLUDE_UKMODETRANSITION_H_

#include <uem_common.h>

#include <uem_data.h>

#ifdef __cplusplus
extern "C"
{
#endif

uem_result UKModeTransition_Initialize();
uem_result UKModeTransition_Finalize();

uem_result UKModeTransition_GetCurrentModeName (IN int nCallerTaskId, IN char *pszTaskName, OUT char **ppszModeName);
uem_result UKModeTransition_SetModeIntegerParameter (IN int nCallerTaskId, IN char *pszTaskName, IN char *pszParamName, IN int nParamVal);
uem_result UKModeTransition_UpdateMode (IN int nCallerTaskId, IN char *pszTaskName);

EModeState UKModeTransition_GetModeState(int nTaskId);

int UKModeTransition_GetModeIndexByModeId(SModeTransitionMachine *pstModeTransition, int nModeId);
int UKModeTransition_GetVariableIndexByName(SModeTransitionMachine *pstModeTransition, char *pszVariableName);
uem_result UKModeTransition_GetCurrentModeIndexByIteration(SModeTransitionMachine *pstModeTransition, int nCurrentIteration, int *pnModeIndex);
uem_result UKModeTransition_GetNextModeStartIndexByIteration(SModeTransitionMachine *pstModeTransition, int nCurrentIteration, OUT int *pnModeIndex, OUT int *pnStartIteration);

uem_result UKModeTransition_Clear(SModeTransitionMachine *pstModeTransition);
EModeState UKModeTransition_GetModeStateInternal(SModeTransitionMachine *pstModeTransition);
EModeState UKModeTransition_UpdateModeStateInternal(SModeTransitionMachine *pstModeTransition, EModeState enModeState, int nIteration);


#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_INCLUDE_UKMODETRANSITION_H_ */
