/*
 * UKCPUTaskManager.h
 *
 *  Created on: 2017. 9. 20.
 *      Author: jej
 */

#ifndef SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKCPUTASKMANAGER_H_
#define SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKCPUTASKMANAGER_H_

#include <uem_common.h>

#include <uem_data.h>

#include <UCThreadMutex.h>

#include <UKTask.h>
#include <UKCPUTaskCommon.h>

#ifdef __cplusplus
extern "C"
{
#endif

#define CPU_ID_NOT_SET (-1)

typedef struct _SCPUTaskManager *HCPUTaskManager;

// global variable which is used for accessing CPUTaskManager from APIs
extern HCPUTaskManager g_hCPUTaskManager;

/**
 * @brief Create a task manager.
 *
 * This function creates a task manager. \n
 * Task manager is an integrated manager of general task manager and composite task manager. \n
 * Internally, it calls both managers to control tasks.
 *
 * @param[out] phCPUTaskManager a task manager handle to be created.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_MUTEX_ERROR, @ref ERR_UEM_OUT_OF_MEMORY, @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKCPUTaskManager_Create(OUT HCPUTaskManager *phCPUTaskManager);

/**
 * @brief Register a general task.
 *
 * This function registers a general task to the task manager.
 *
 * @param hCPUTaskManager a task manager handle.
 * @param pstMappedTask a general task and mapping info to register.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_OUT_OF_MEMORY, @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKCPUTaskManager_RegisterTask(HCPUTaskManager hCPUTaskManager, SMappedGeneralTaskInfo *pstMappedTask);

/**
 * @brief Register a composite task.
 *
 * This function registers a composite task to the task manager.
 *
 * @param hCPUTaskManager a task manager handle.
 * @param pstMappedTask a composite task and mapping info to register.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_OUT_OF_MEMORY, @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INVALID_PARAM, \n
 *         @ref ERR_UEM_INTERNAL_FAIL.
 */
uem_result UKCPUTaskManager_RegisterCompositeTask(HCPUTaskManager hCPUTaskManager, SMappedCompositeTaskInfo *pstMappedTask);

/**
 * @brief Execute registered tasks.
 *
 * This function runs all the registered tasks. \n
 * First of all, it executes control tasks first and then executes rest tasks except control-driven tasks.
 *
 * @param hCPUTaskManager a task manager handle.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INTERNAL_FAIL, @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_OUT_OF_MEMORY.
 *         @ref ERR_UEM_INTERNAL_FAIL is happened when the thread creation or thread-related operations are failed.
 */
uem_result UKCPUTaskManager_RunRegisteredTasks(HCPUTaskManager hCPUTaskManager);

/**
 * @brief Stop all tasks.
 *
 * This function stops all tasks.
 *
 * @param hCPUTaskManager a task manager handle.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INTERNAL_FAIL. \n
 *          @ref ERR_UEM_INTERNAL_FAIL is occurred when threads in the task is not terminated properly.
 */
uem_result UKCPUTaskManager_StopAllTasks(HCPUTaskManager hCPUTaskManager);

/**
 * @brief Suspend a task.
 *
 * This function suspends a task. the task state must be running.
 *
 * @param hCPUTaskManager a task manager handle.
 * @param nTaskId target task id to suspend.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_NOT_FOUND, @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INVALID_PARAM, \n
 *         @ref ERR_UEM_ILLEGAL_DATA. \n
 *         @ref ERR_UEM_NOT_FOUND is occurred when the target task is not found in the task manager. \n
 *         @ref ERR_UEM_ILLEGAL_DATA is occurred when the target task has a subgraph with process network.
 */
uem_result UKCPUTaskManager_SuspendTask(HCPUTaskManager hCPUTaskManager, int nTaskId);

/**
 * @brief Stop task nicely.
 *
 * This function stops a task nicely. This function tries to wait until the iteration is finished.
 *
 * @param hCPUTaskManager a task manager handle.
 * @param nTaskId target task id to stop.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_NOT_FOUND, @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INVALID_PARAM, \n
 *         @ref ERR_UEM_ILLEGAL_DATA. \n
 *         @ref ERR_UEM_NOT_FOUND is occurred when the target task is not found in the task manager. \n
 *         @ref ERR_UEM_ILLEGAL_DATA is occurred when the target task has a subgraph with process network.
 */
uem_result UKCPUTaskManager_StoppingTask(HCPUTaskManager hCPUTaskManager, int nTaskId);

/**
 * @brief Stop task forcedly.
 *
 * This function stops a task forcedly. This function immediately stops the task.
 * For exception, this function also stop nicely if a task has a dataflow subgraph.
 *
 * @param hCPUTaskManager a task manager handle.
 * @param nTaskId target task id to stop.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_NOT_FOUND, @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INVALID_PARAM, \n
 *         @ref ERR_UEM_ILLEGAL_DATA, @ref ERR_UEM_INTERNAL_FAIL. \n
 *         @ref ERR_UEM_NOT_FOUND is occurred when the target task is not found in the task manager. \n
 *         @ref ERR_UEM_ILLEGAL_DATA is occurred when the target task has a subgraph with process network. \n
 *         @ref ERR_UEM_INTERNAL_FAIL is occurred when threads in the task is not terminated properly.
 */
uem_result UKCPUTaskManager_StopTask(HCPUTaskManager hCPUTaskManager, int nTaskId);

/**
 * @brief Run a task.
 *
 * This function runs a task. the task state must be stopped.
 *
 * @param hCPUTaskManager a task manager handle.
 * @param nTaskId target task id to run.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_NOT_FOUND, @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INVALID_PARAM, \n
 *         @ref ERR_UEM_ILLEGAL_DATA, @ref ERR_UEM_INTERNAL_FAIL. \n
 *         @ref ERR_UEM_NOT_FOUND is occurred when the target task is not found in the task manager. \n
 *         @ref ERR_UEM_ILLEGAL_DATA is occurred when the target task has a subgraph with process network. \n
 *         @ref ERR_UEM_INTERNAL_FAIL is occurred when threads in the task is not started properly.
 */
uem_result UKCPUTaskManager_RunTask(HCPUTaskManager hCPUTaskManager, int nTaskId);

/**
 * @brief Resume a task.
 *
 * This function resumes a task. the task state must be suspended.
 *
 * @param hCPUTaskManager a task manager handle.
 * @param nTaskId target task id to resume.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_NOT_FOUND, @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INVALID_PARAM, \n
 *         @ref ERR_UEM_ILLEGAL_DATA, @ref ERR_UEM_INTERNAL_FAIL. \n
 *         @ref ERR_UEM_NOT_FOUND is occurred when the target task is not found in the task manager. \n
 *         @ref ERR_UEM_ILLEGAL_DATA is occurred when the target task has a subgraph with process network. \n
 *         @ref ERR_UEM_INTERNAL_FAIL is occurred when threads in the task is not started properly.
 */
uem_result UKCPUTaskManager_ResumeTask(HCPUTaskManager hCPUTaskManager, int nTaskId);

/**
 * @brief Get task state.
 *
 * This function retrieves the state of the task.
 *
 * @param hCPUTaskManager a task manager handle.
 * @param nTaskId target task id to get state.
 * @param[out] penTaskState task state.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_NOT_FOUND, @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INVALID_PARAM, \n
 *         @ref ERR_UEM_ILLEGAL_DATA, @ref ERR_UEM_INTERNAL_FAIL. \n
 *         @ref ERR_UEM_NOT_FOUND is occurred when the target task is not found in the task manager. \n
 *         @ref ERR_UEM_ILLEGAL_DATA is occurred when the target task has a subgraph with process network. \n
 *         @ref ERR_UEM_INTERNAL_FAIL is occurred when stopping state task termination is not finished properly.
 */
uem_result UKCPUTaskManager_GetTaskState(HCPUTaskManager hCPUTaskManager, int nTaskId, OUT EInternalTaskState *penTaskState);

/**
 * @brief Check all tasks are stopped.
 *
 * This function checks all the tasks are stopped.
 *
 * @param hCPUTaskManager a task manager handle.
 * @param[out] pbStopped boolean value of all tasks are stopped or not.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_INTERNAL_FAIL. \n
 *         @ref ERR_UEM_INTERNAL_FAIL is occurred when threads in the task is not terminated yet if the task is stopping-state task.
 */
uem_result UKCPUTaskManager_IsAllTaskStopped(HCPUTaskManager hCPUTaskManager, OUT uem_bool *pbStopped);

/**
 * @brief Update the mapping info of the task.
 *
 * This function updates the mapping info of the task.
 *
 * @param hCPUTaskManager a task manager handle.
 * @param nTaskId an id of task to update the mapping info.
 * @param nNewLocalId a new core id to assign.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_ILLEGAL_CONTROL. \n
 *         @ref ERR_UEM_ILLEGAL_CONTROL is occurred when the task is not computational or control type, and the task is statically scheduled.
 */
uem_result UKCPUTaskManager_ChangeMappedCore(HCPUTaskManager hCPUTaskManager, int nTaskId, int nNewLocalId);

/**
 * @brief Destroy a task manager.
 *
 * This function a task manager.
 *
 * @param[in,out] phCPUTaskManager a task manager handle to be destroyed.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INTERNAL_FAIL. \n
 *         @ref ERR_UEM_INTERNAL_FAIL is occurred when threads in the task is not terminated properly.
 */
uem_result UKCPUTaskManager_Destroy(IN OUT HCPUTaskManager *phCPUTaskManager);

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKCPUTASKMANAGER_H_ */
