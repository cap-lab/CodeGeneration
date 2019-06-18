/*
 * UKTaskScheduler.h
 *
 *  Created on: 2018. 9. 6.
 *      Author: chjej202
 */

#ifndef SRC_KERNEL_CONSTRAINED_INCLUDE_UKTASKSCHEDULER_H_
#define SRC_KERNEL_CONSTRAINED_INCLUDE_UKTASKSCHEDULER_H_

#include <uem_common.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief Perform Task initialization.
 *
 * This function performs TASK_INITIALIZE function on tasks and sets runtimeinfo tNextTime to current time. \n
 * These actions are executed in the order of ControlTask - GeneralTask - Composite Task.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKTaskScheduler_Init();

/**
 * @brief Perform task execution.
 *
 * This function performs task execution. \n
 * The GeneralTask and ControlTask are set to run equal or less than the specified maximum execution times within the time period if the run condition is TIME_DRIVEN. \n
 * In the case of CONTROL_DRIVEN task, the task is terminated with TASK_WRAPUP function after it has been executed only once. \n
 * In the case of composite task, in the case of Control_DRIVEN, the CompositeGo function is called after the wrapup function is called hierarchically. \n
 * * These actions are executed in the order of GeneralTask - Composite Task - ControlTask. \n
 *
 * @return  @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *
 */
uem_result UKTaskScheduler_Run();

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_CONSTRAINED_INCLUDE_UKTASKSCHEDULER_H_ */
