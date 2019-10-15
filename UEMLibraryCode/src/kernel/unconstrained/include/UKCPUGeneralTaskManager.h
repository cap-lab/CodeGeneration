/*
 * UKCPUGeneralTaskManager.h
 *
 *  Created on: 2018. 2. 6.
 *      Author: jej
 */

#ifndef SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKCPUGENERALTASKMANAGER_H_
#define SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKCPUGENERALTASKMANAGER_H_

#include <uem_common.h>

#include <uem_data.h>

#include <UKCPUTaskCommon.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct _SCPUGeneralTaskManager *HCPUGeneralTaskManager;

typedef uem_result (*CbFnTraverseGeneralTask)(STask *pstTask, IN void *pUserData);

/**
 * @brief Create a general task manager.
 *
 * This function creates a general task manager. \n
 * General task means that one thread is responsible to one task to execute. \n
 * Multiple threads may be used when the task has multiple instances. \n
 * (These tasks can be generated when the scheduling policy is static-assignment for code generation).
 *
 * @param[out] phManager a general task manager handle to be created.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_OUT_OF_MEMORY, @ref ERR_UEM_MUTEX_ERROR. \n
 */
uem_result UKCPUGeneralTaskManager_Create(IN OUT HCPUGeneralTaskManager *phManager);

/**
 * @brief Register a task to general task manager.
 *
 * This function registers a task to the general task manager.
 *
 * @param hManager a general task manager handle.
 * @param pstMappedTask  a general task and mapping info to register.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INVALID_PARAM, \n
 *         @ref ERR_UEM_OUT_OF_MEMORY.
 */
uem_result UKCPUGeneralTaskManager_RegisterTask(HCPUGeneralTaskManager hManager, SMappedGeneralTaskInfo *pstMappedTask);

/**
 * @brief Traverse all the general tasks in general task manager.
 *
 * This function traverses all the general tasks in the general task manager.
 *
 * @param hManager a general task manager handle.
 * @param fnCallback callback function during traversing general tasks.
 * @param pUserData user data used in the callback function.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INVALID_PARAM, and \n
 *         corresponding results from callback function. \n
 *         @ref ERR_UEM_NOT_FOUND is occurred when the target task is not found in the general task manager.
 */
uem_result UKCPUGeneralTaskManager_TraverseGeneralTaskList(HCPUGeneralTaskManager hManager, CbFnTraverseGeneralTask fnCallback, void *pUserData);

/**
 * @brief Create threads in the task.
 *
 * This function creates threads in the task. \n
 * Before calling this function, registered general tasks are not working at all.
 *
 * @param hManager a general task manager handle.
 * @param pstTargetTask target task to create threads.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_NOT_FOUND, @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INVALID_PARAM, \n
 *         @ref ERR_UEM_INTERNAL_FAIL. \n
 *         @ref ERR_UEM_INTERNAL_FAIL is occurred when thread-related operations are failed.
 */
uem_result UKCPUGeneralTaskManager_CreateThread(HCPUGeneralTaskManager hManager, STask *pstTargetTask);

/**
 * @brief Change state of the task.
 *
 * This function changes the task state. \n
 * After calling this function, @ref UKCPUGeneralTaskManager_ActivateThread is also needed because it does not wake up threads.
 *
 * @param hManager a general task manager handle.
 * @param pstTargetTask target task to change state.
 * @param enTaskState new task state.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_NOT_FOUND, @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INVALID_PARAM. \n
 *         @ref ERR_UEM_NOT_FOUND is occurred when the target task is not found in the general task manager.
 */
uem_result UKCPUGeneralTaskManager_ChangeState(HCPUGeneralTaskManager hManager, STask *pstTargetTask, ECPUTaskState enTaskState);

/**
 * @brief Activate threads in the task.
 *
 * This function activates threads. This function wakes up the threads to do jobs according to their task state.
 *
 * @param hManager a general task manager handle.
 * @param pstTargetTask target task to activate threads.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_NOT_FOUND, @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INVALID_PARAM. \n
 *         @ref ERR_UEM_NOT_FOUND is occurred when the target task is not found in the general task manager.
 */
uem_result UKCPUGeneralTaskManager_ActivateThread(HCPUGeneralTaskManager hManager, STask *pstTargetTask);

/**
 * @brief Wait until the task is activated.
 *
 * This function waits until the task is fully activated.
 * "Activated" means the tasks finishes its init() functions and start to run go() function.
 *
 * @param hManager a general task manager handle.
 * @param pstTargetTask target task to wait task activation.
 * @param nTimeoutInMilliSec wait time limit.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_NOT_FOUND, @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INVALID_PARAM,
 *         @ref ERR_UEM_TIME_EXPIRED. \n
 *         @ref ERR_UEM_NOT_FOUND is occurred when the target task is not found in the general task manager. \n
 *         @ref ERR_UEM_TIME_EXPIRED is occurred when the task is not activated until the timeout.
 */
uem_result UKCPUGeneralTaskManager_WaitTaskActivated(HCPUGeneralTaskManager hManager, STask *pstTargetTask, int nTimeoutInMilliSec);

/**
 * @brief Get task state.
 *
 * This function retrieves the state of the task.
 *
 * @param hManager a general task manager handle.
 * @param pstTargetTask target task to get state.
 * @param[out] penTaskState task state.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_NOT_FOUND, @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INVALID_PARAM, \n
 *         @ref ERR_UEM_INTERNAL_FAIL. \n
 *         @ref ERR_UEM_NOT_FOUND is occurred when the target task is not found in the general task manager. \n
 *         @ref ERR_UEM_INTERNAL_FAIL is occurred when threads in the task is not terminated yet if the task is stopping-state task.
 */
uem_result UKCPUGeneralTaskManager_GetTaskState(HCPUGeneralTaskManager hManager, STask *pstTargetTask, OUT ECPUTaskState *penTaskState);

/**
 * @brief Check all the general tasks are stopped.
 *
 * This function checks all the tasks are stopped in the general task manager.
 *
 * @param hManager a general task manager handle.
 * @param[out] pbStopped boolean value of all tasks are stopped or not.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INTERNAL_FAIL. \n
 *         @ref ERR_UEM_INTERNAL_FAIL is occurred when threads in the task is not terminated yet if the task is stopping-state task.
 */
uem_result UKCPUGeneralTaskManager_CheckAllTaskStopped(HCPUGeneralTaskManager hManager, OUT uem_bool *pbStopped);

/**
 * @brief Check all threads in the task are running now.
 *
 * This function checks all threads in the task are running.
 *
 * @param hManager a general task manager handle.
 * @param pstTargetTask target task to check.
 * @param[out] pbStarted boolean value of task is started or not.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_NOT_FOUND, @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INVALID_PARAM. \n
 *         @ref ERR_UEM_NOT_FOUND is occurred when the target task is not found in the general task manager.
 */
uem_result UKCPUGeneralTaskManager_CheckTaskStarted(HCPUGeneralTaskManager hManager, STask *pstTargetTask, OUT uem_bool *pbStarted);

/**
 * @brief Destroy threads in the task.
 *
 * This function destroys threads in the task. \n
 * After calling this function, @ref UKCPUGeneralTaskManager_CreateThread is needed to re-execute tasks.
 *
 * @param hManager a general task manager handle.
 * @param pstTargetTask target task to destroy threads.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_NOT_FOUND, @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INVALID_PARAM, \n
 *         @ref ERR_UEM_INTERNAL_FAIL. \n
 *         @ref ERR_UEM_NOT_FOUND is occurred when the target task is not found in the general task manager.
 *         @ref ERR_UEM_INTERNAL_FAIL is occurred when threads in the task is not terminated properly.
 */
uem_result UKCPUGeneralTaskManager_DestroyThread(HCPUGeneralTaskManager hManager, STask *pstTargetTask);

/**
 * @brief Check all the general tasks are stopped.
 *
 * This function destroys a general task manager.
 *
 * @param[in,out] phManager a general task manager handle to be destroyed.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INTERNAL_FAIL. \n
 *         @ref ERR_UEM_INTERNAL_FAIL is occurred when threads in the task is not terminated properly.
 */
uem_result UKCPUGeneralTaskManager_Destroy(IN OUT HCPUGeneralTaskManager *phManager);


uem_result UKCPUGeneralTaskManagerCB_IsSourceTask(void *pTaskHandle, OUT uem_bool *pbIsSourceTask);
uem_result UKCPUGeneralTaskManagerCB_GetCurrentTaskStructure(void *pTaskHandle, OUT STask **ppstTask);
uem_result UKCPUGeneralTaskManagerCB_ChangeTaskState(void *pTaskHandle, ECPUTaskState enState);
uem_result UKCPUGeneralTaskManagerCB_GetCurrentTaskState(void *pTaskHandle, OUT ECPUTaskState *penState);
uem_result UKCPUGeneralTaskManagerCB_GetManagerHandle(void *pTaskHandle, OUT HCPUGeneralTaskManager *phManager);
uem_result UKCPUGeneralTaskManagerCB_ActivateTask(void *pTaskHandle);

uem_result UKCPUGeneralTaskManagerCB_ClearLoopIndex(void *pTaskHandle);
uem_result UKCPUGeneralTaskManagerCB_GetLoopIndex(void *pTaskHandle, OUT int *pnLoopIndex);
uem_result UKCPUGeneralTaskManagerCB_SetLoopIndex(void *pTaskHandle, OUT int nLoopIndex);
uem_result UKCPUGeneralTaskManagerCB_GetFunctionCalled(void *pThreadHandle, OUT uem_bool *pbFunctionCalled);
uem_result UKCPUGeneralTaskManagerCB_GetThreadIndex(void *pThreadHandle, OUT int *pnThreadIndex);
uem_result UKCPUGeneralTaskManagerCB_GetTaskGraphLock(void *pTaskHandle, OUT HThreadMutex *phMutex);

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKCPUGENERALTASKMANAGER_H_ */
