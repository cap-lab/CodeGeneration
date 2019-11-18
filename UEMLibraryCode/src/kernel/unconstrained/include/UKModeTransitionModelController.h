/*
 * UKModeTransitionModelController.h
 *
 *  Created on: 2019. 9. 27.
 *      Author: jej
 */

#ifndef SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKMODETRANSITIONMODELCONTROLLER_H_
#define SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKMODETRANSITIONMODELCONTROLLER_H_

#include <uem_common.h>

#include <uem_data.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief Handle MTM-related control in a composite task manager.
 *
 * This function handles MTM-related control in a composite task manager.
 *
 * @param pstGraph a target task graph to handle.
 * @param pCurrentTaskHandle a task handle.
 * @param pCurrentThreadHandle a task thread handle.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKModeTransitionMachineController_HandleModelComposite(STaskGraph *pstGraph, void *pCurrentTaskHandle, void *pCurrentThreadHandle);


/**
 * @brief Clear mode transition structure.
 *
 * This function clears mode transition structure.
 *
 * @param pstTaskGraph a task graph structure to clear internal structure.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKModeTransitionMachineController_Clear(STaskGraph *pstTaskGraph);

/**
 * @brief Change a thread state based on new task state.
 *
 * This function changes a thread state in a task. \n
 * This function is called in a composite task manager. \n
 * Depending on what kind of model is used, a thread state can be different to a task state.
 *
 * @param pstGraph a target task graph to handle.
 * @param pCurrentTaskHandle a task handle.
 * @param pCurrentThreadHandle a task thread handle.
 * @param enTargetState a task state to be changed.
 * @param[out] penState a new thread state.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKModeTransitionMachineController_ChangeTaskThreadState(STaskGraph *pstGraph, void *pCurrentTaskHandle, void *pCurrentThreadHandle, ECPUTaskState enTargetState, OUT ECPUTaskState *penState);

/**
 * @brief Handle MTM-related control in a general task manager.
 *
 * This function handles MTM-related control in a general task manager.
 *
 * @param pstGraph a target task graph to handle.
 * @param pCurrentTaskHandle a task handle.
 * @param pCurrentThreadHandle a task thread handle.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         @ref ERR_UEM_FOUND_DATA is retrieved when a task needs activations to other tasks. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKModeTransitionMachineController_HandleModelGeneral(STaskGraph *pstGraph, void *pCurrentTaskHandle, void *pCurrentThreadHandle);

/**
 * @brief Change subgraph task states based on new parent task state.
 *
 * This function changes subgraph task states in a parent task graph. \n
 * This function is called in a general task manager. \n
 * Depending on what kind of model is used, a subgraph task state can be different to a parent task graph state.
 *
 * @param pstGraph a target parent task graph to handle.
 * @param pCurrentTaskHandle a task handle to change a state.
 * @param pCurrentThreadHandle not used.
 * @param enTargetState a task state to be changed.
 * @param[out] penState a new subgraph task state.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKModeTransitionMachineController_ChangeSubGraphTaskState(STaskGraph *pstGraph, void *pCurrentTaskHandle, void *pCurrentThreadHandle, ECPUTaskState enTargetState, OUT ECPUTaskState *penState);

/**
 * @brief Handle MTM-related control in a general task manager during TASK_STATE_STOPPING.
 *
 * This function handles MTM-related control in a general task manager during TASK_STATE_STOPPING.
 *
 * @param pstGraph a target task graph to handle.
 * @param pCurrentTaskHandle a task handle.
 * @param pCurrentThreadHandle not used.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         @ref ERR_UEM_ALREADY_DONE is returned when an iteration number is reached to the target iteration. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKModeTransitionMachineController_HandleModelGeneralDuringStopping(STaskGraph *pstGraph, void *pCurrentTaskHandle, void *pCurrentThreadHandle);

#ifdef __cplusplus
}
#endif


#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKMODETRANSITIONMODELCONTROLLER_H_ */
