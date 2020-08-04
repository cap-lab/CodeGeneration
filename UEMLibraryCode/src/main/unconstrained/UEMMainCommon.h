/*
 * common.h
 *
 *  Created on: 2020. 01. 08.
 *      Author: urmydata
 */

#ifndef SRC_MAIN_UNCONSTRAINED_NATIVE_UEMMAINCOMMON_H_
#define SRC_MAIN_UNCONSTRAINED_NATIVE_UEMMAINCOMMON_H_

#include <stdio.h>

#include <uem_common.h>

#include <uem_data.h>

#include <UCTime.h>

#include <UKTask.h>
#include <UKCPUTaskManager.h>
#include <UKProcessor.h>
#include <UKTime.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * Create tasks using TaskManager.
 *
 * @param   hManager	a task manager handle.
 *
 * @return	@ref ERR_UEM_NOERROR is returned if there is no error. \n
 * 			Errors to be returned - @ref ERR_UEM_ILLEGAL_DATA, @ref ERR_UEM_NOT_FOUND, @ref ERR_UEM_INVALID_PARAM, @ref ERR_INVALID_HANDLE, @ref ERR_INTERNAL_FAIL, @ref ERR_UEM_OUT_OF_MEMORY.
 */
uem_result UEMMainCommon_CreateTasks(IN OUT HCPUTaskManager hManager);

/**
 * Calculate End time of the execution
 *
 * @param	llStartTime	the start time of the execution.
 *
 * @return	llEndTime	the end time of the execution.
 */
long long UEMMainCommon_GetEndTime(long long llStartTime);

/**
 * Execute tasks. \n
 * It includes all procedure of creation, execution, and destruction of tasks.
 *
 * @param	llStartTime	the start time of the execution.
 *
 * @return	@ref ERR_UEM_NOERROR is returned if there is no error. \n
 * 			All type of Errors to be returned.
 */
uem_result UEMMainCommon_ExecuteTasks();


#ifdef __cplusplus
}
#endif

#endif /* SRC_MAIN_UNCONSTRAINED_NATIVE_UEMMAINCOMMON_H_ */
