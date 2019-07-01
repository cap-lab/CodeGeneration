/*
 * UKModeTransition.h
 *
 *  Created on: 2017. 8. 28.
 *      Author: jej
 */

#ifndef SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKMODETRANSITION_H_
#define SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKMODETRANSITION_H_

#include <uem_common.h>

#include <uem_data.h>

#ifdef __cplusplus
extern "C"
{
#endif


/**
 * @brief Get current mode name of a task.
 *
 * This function retrieves the current mode name of task @a pszTaskName.
 *
 * @param nCallerTaskId id of caller task (currently not used).
 * @param pszTaskName target MTM (mode transition machine) task name to get mode name.
 * @param[out] ppszModeName retrieved mode name.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_NO_DATA, @ref ERR_UEM_ILLEGAL_DATA.
 *         @ref ERR_UEM_NO_DATA is occurred when the task name cannot be found.
 *         @ref ERR_UEM_ILLEGAL_DATA is occurred when the task is not an MTM task.
 */
uem_result UKModeTransition_GetCurrentModeName (IN int nCallerTaskId, IN char *pszTaskName, OUT char **ppszModeName);

/**
 * @brief Set Mode transition machine task integer parameter.
 *
 * This function sets Mode transition machine task integer parameter.
 *
 * @param nCallerTaskId id of caller task (currently not used).
 * @param pszTaskName target MTM (mode transition machine) task name to get integer parameter.
 * @param pszParamName integer parameter name.
 * @param nParamVal integer value to set.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_NO_DATA, @ref ERR_UEM_ILLEGAL_DATA.
 *         @ref ERR_UEM_NO_DATA is occurred when the task name cannot be found.
 *         @ref ERR_UEM_ILLEGAL_DATA is occurred when the task is not an MTM task.
 */
uem_result UKModeTransition_SetModeIntegerParameter (IN int nCallerTaskId, IN char *pszTaskName, IN char *pszParamName, IN int nParamVal);

/**
 * @brief Update mode of MTM task.
 *
 * This function updates the mode of MTM task with mode transition function. \n
 * Mode transition may be happened depending on the update of mode variables.
 *
 * @param nCallerTaskId id of caller task (currently not used).
 * @param pszTaskName target MTM (mode transition machine) task name to update mode.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_NO_DATA, @ref ERR_UEM_ILLEGAL_DATA.
 *         @ref ERR_UEM_NO_DATA is occurred when the task name cannot be found.
 *         @ref ERR_UEM_ILLEGAL_DATA is occurred when the task is not an MTM task.
 */
uem_result UKModeTransition_UpdateMode (IN int nCallerTaskId, IN char *pszTaskName);

/**
 * @brief Get Mode state of MTM task with lock protection.
 *
 * This function retrieves the mode state of target MTM task. This function is called by composite MTM task.
 *
 * @param nTaskId target MTM (mode transition machine) task id to get mode state.
 *
 * @return mode state (@ref EModeState).
 */
EModeState UKModeTransition_GetModeState(int nTaskId);

/**
 * @brief Get mode index by mode ID.
 *
 * This function retrieves mode index value by mode ID. Mode index is an index number to access mode information in astModeMap.
 *
 * @param pstModeTransition mode transition machine structure.
 * @param nModeId mode id.
 *
 * @return mode index number.
 */
int UKModeTransition_GetModeIndexByModeId(SModeTransitionMachine *pstModeTransition, int nModeId);

/**
 * @brief Get variable index with variable name.
 *
 * This function retrieves variable index by variable name. Variable index is an index number to access variable value in astVarIntMap.
 *
 * @param pstModeTransition mode transition machine structure.
 * @param pszVariableName mode variable name.
 *
 * @return variable index number.
 */
int UKModeTransition_GetVariableIndexByName(SModeTransitionMachine *pstModeTransition, char *pszVariableName);

/**
 * @brief Get current mode's index by iteration number.
 *
 * This function retrieves current mode's index by current iteration number.
 *
 * @param pstModeTransition mode transition machine structure.
 * @param nCurrentIteration current iteration number.
 * @param[out] pnModeIndex an index value to access mode information in array.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         @ref ERR_UEM_NOT_FOUND is occurred when the corresponding mode index cannot be found \n
 *         (This is occurred when the iteration is 0 or task is going to stop).
 */
uem_result UKModeTransition_GetCurrentModeIndexByIteration(SModeTransitionMachine *pstModeTransition, int nCurrentIteration, OUT int *pnModeIndex);

/**
 * @brief Get next mode's index and new iteration number by current iteration number.
 *
 * This function retrieves next mode's index and new iteration number by current iteration number.
 *
 * @param pstModeTransition mode transition machine structure.
 * @param nCurrentIteration current iteration number.
 * @param[out] pnModeIndex an index value to access mode information in array.
 * @param[out] pnStartIteration next iteration number.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_NOT_FOUND, @ref ERR_UEM_NO_DATA. \n
 *         @ref ERR_UEM_NOT_FOUND is occurred when the corresponding mode index cannot be found. \n
 *         @ref ERR_UEM_NO_DATA is occurred when next mode index is not available yet.
 */
uem_result UKModeTransition_GetNextModeStartIndexByIteration(SModeTransitionMachine *pstModeTransition, int nCurrentIteration, OUT int *pnModeIndex, OUT int *pnStartIteration);

/**
 * @brief Clear mode transition machine structure.
 *
 * This function clears mode transition machine structure.
 *
 * @param pstModeTransition mode transition machine structure.
 *
 * @return It always returns @ref ERR_UEM_NOERROR.
 */
uem_result UKModeTransition_Clear(SModeTransitionMachine *pstModeTransition);

/**
 * @brief Get Mode state of MTM task without lock protection.
 *
 * This function retrieves mode state. This function is called by task managers.
 *
 * @param pstModeTransition mode transition machine structure.
 *
 * @return mode state (@ref EModeState).
 */
EModeState UKModeTransition_GetModeStateInternal(SModeTransitionMachine *pstModeTransition);

/**
 * @brief Update mode state and newly-started mode's iteration number.
 *
 * This function updates mode state and newly-started mode's iteration number. \n
 * When the mode is moved from MODE_STATE_TRANSITING to MODE_STATE_NORMAL.
 *
 * @param pstModeTransition mode transition machine structure.
 * @param enModeState mode state to update.
 * @param nIteration iteration number to update the mode state.
 *
 * @return updated mode state (@ref EModeState).
 */
EModeState UKModeTransition_UpdateModeStateInternal(SModeTransitionMachine *pstModeTransition, EModeState enModeState, int nIteration);


#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKMODETRANSITION_H_ */
