/*
 * UKLoopTaskController.h
 *
 *  Created on: 2019. 10. 7.
 *      Author: jej
 */

#ifndef SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKLOOPMODELCONTROLLER_H_
#define SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKLOOPMODELCONTROLLER_H_

#include <uem_common.h>

#include <uem_data.h>

#include <UKCPUTaskCommon.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief Handle convergent-loop-related control in a general task manager.
 *
 * This function handles convergent loop in a general task manager.
 *
 * @param pstGraph a target task graph to handle.
 * @param pCurrentTaskHandle a task handle.
 * @param pCurrentThreadHandle a task thread handle.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         @ref ERR_UEM_FOUND_DATA is retrieved when a task needs activations to other tasks. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKLoopModelController_HandleConvergentLoop(STaskGraph *pstGraph, void *pCurrentTaskHandle, void *pCurrentThreadHandle);

/**
 * @brief Handle convergent-loop-related control in a general task manager during TASK_STATE_STOPPING
 *
 * This function handles convergent loop in a general task manager during TASK_STATE_STOPPING.
 *
 * @param pstGraph a target task graph to handle.
 * @param pCurrentTaskHandle a task handle.
 * @param pCurrentThreadHandle not used.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         @ref ERR_UEM_ALREADY_DONE is returned when an iteration number is reached to the target iteration. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKLoopModelController_HandleConvergentLoopDuringStopping(STaskGraph *pstGraph, void *pCurrentTaskHandle, void *pCurrentThreadHandle);

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKLOOPMODELCONTROLLER_H_ */
