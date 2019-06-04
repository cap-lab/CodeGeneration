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

/**
 * @brief
 *
 * This function
 *
 * @return
 */
uem_result UKModeTransition_Initialize();

/**
 * @brief
 *
 * This function
 *
 * @return
 */
uem_result UKModeTransition_Finalize();

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
uem_result UKModeTransition_GetCurrentModeName (IN int nCallerTaskId, IN char *pszTaskName, OUT char **ppszModeName);

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
uem_result UKModeTransition_SetModeIntegerParameter (IN int nCallerTaskId, IN char *pszTaskName, IN char *pszParamName, IN int nParamVal);

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
uem_result UKModeTransition_UpdateMode (IN int nCallerTaskId, IN char *pszTaskName);

/**
 * @brief
 *
 * This function
 *
 * @param nTaskId
 *
 * @return
 */
EModeState UKModeTransition_GetModeState(int nTaskId);

/**
 * @brief
 *
 * This function
 *
 * @param pstModeTransition
 * @param nModeId
 *
 * @return
 */
int UKModeTransition_GetModeIndexByModeId(SModeTransitionMachine *pstModeTransition, int nModeId);

/**
 * @brief
 *
 * This function
 *
 * @param pstModeTransition
 * @param pszVariableName
 *
 * @return
 */
int UKModeTransition_GetVariableIndexByName(SModeTransitionMachine *pstModeTransition, char *pszVariableName);

/**
 * @brief
 *
 * This function
 *
 * @param pstModeTransition
 * @param nCurrentIteration
 * @param[out] pnModeIndex
 *
 * @return
 */
uem_result UKModeTransition_GetCurrentModeIndexByIteration(SModeTransitionMachine *pstModeTransition, int nCurrentIteration, OUT int *pnModeIndex);

/**
 * @brief
 *
 * This function
 *
 * @param pstModeTransition
 * @param nCurrentIteration
 * @param[out] pnModeIndex
 * @param[out] pnStartIteration
 *
 * @return
 */
uem_result UKModeTransition_GetNextModeStartIndexByIteration(SModeTransitionMachine *pstModeTransition, int nCurrentIteration, OUT int *pnModeIndex, OUT int *pnStartIteration);

/**
 * @brief
 *
 * This function
 *
 * @param pstModeTransition
 *
 * @return
 */
uem_result UKModeTransition_Clear(SModeTransitionMachine *pstModeTransition);

/**
 * @brief
 *
 * This function
 *
 * @param pstModeTransition
 *
 * @return
 */
EModeState UKModeTransition_GetModeStateInternal(SModeTransitionMachine *pstModeTransition);

/**
 * @brief
 *
 * This function
 *
 * @param pstModeTransition
 * @param enModeState
 * @param nIteration
 *
 * @return
 */
EModeState UKModeTransition_UpdateModeStateInternal(SModeTransitionMachine *pstModeTransition, EModeState enModeState, int nIteration);


#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_INCLUDE_UKMODETRANSITION_H_ */
