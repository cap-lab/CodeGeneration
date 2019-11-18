/*
 * UKCPUCompositeTaskManager.h
 *
 *  Created on: 2018. 2. 6.
 *      Author: jej
 */

#ifndef SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKCPUCOMPOSITETASKMANAGER_H_
#define SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKCPUCOMPOSITETASKMANAGER_H_

#include <uem_common.h>

#include <uem_data.h>

#include <UKCPUTaskCommon.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct _SCPUCompositeTaskManager *HCPUCompositeTaskManager;

typedef uem_result (*CbFnTraverseCompositeTask)(STask *pstTask, IN void *pUserData);
typedef uem_result (*FnTaskThreadTraverse)(void *pCurrentTaskHandle, void *pCurrentThreadHandle, void *pUserData, OUT uem_bool *pbActivateThread);

/**
 * @brief Create a composite task manager.
 *
 * This function creates a composite task manager. \n
 * Composite task means that one or multiple tasks are combined to a single thread to be executed. \n
 * Composite task is created when the schedule order is already defined before generating codes. \n
 * (These tasks can be generated when the scheduling policy is self-timed for code generation).
 *
 * @param[out] phManager a composite task manager handle to be created.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_MUTEX_ERROR, @ref ERR_UEM_OUT_OF_MEMORY.
 */
uem_result UKCPUCompositeTaskManager_Create(IN OUT HCPUCompositeTaskManager *phManager);

/**
 * @brief Register a task to composite task manager.
 *
 * This function registers a task to the composite task manager.
 *
 * @param hManager a composite task manager handle.
 * @param pstMappedTask a composite task and mapping info to register.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_OUT_OF_MEMORY, \n
 *         @ref ERR_UEM_INTERNAL_FAIL.
 */
uem_result UKCPUCompositeTaskManager_RegisterTask(HCPUCompositeTaskManager hManager, SMappedCompositeTaskInfo *pstMappedTask);

/**
 * @brief Traverse all the composite tasks in composite task manager.
 *
 * This function traverses all the composite tasks in the composite task manager.
 *
 * @param hManager a composite task manager handle.
 * @param fnCallback callback function during traversing composite tasks.
 * @param pUserData user data used in the callback function.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_INVALID_HANDLE, and \n
 *         corresponding results from callback function.
 */
uem_result UKCPUCompositeTaskManager_TraverseCompositeTaskList(HCPUCompositeTaskManager hManager, CbFnTraverseCompositeTask fnCallback, void *pUserData);

/**
 * @brief Create threads in the task.
 *
 * This function creates threads in the task. \n
 * Before calling this function, registered composite tasks are not working at all.
 *
 * @param hManager a composite task manager handle.
 * @param pstTargetTask target task to create threads.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_NOT_FOUND, @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_OUT_OF_MEMORY, \n
 *         @ref ERR_UEM_INTERNAL_FAIL. \n
 *         @ref ERR_UEM_NOT_FOUND is occurred when the target task is not found in the composite task manager. \n
 *         @ref ERR_UEM_INTERNAL_FAIL is happened when the thread creation or thread-related operations are failed.
 */
uem_result UKCPUCompositeTaskManager_CreateThread(HCPUCompositeTaskManager hManager, STask *pstTargetTask);

/**
 * @brief Change state of the task.
 *
 * This function changes the task state. \n
 * After calling this function, @ref UKCPUCompositeTaskManager_ActivateThread is also needed because it does not wake up threads.
 *
 * @param hManager a composite task manager handle.
 * @param pstTargetTask target task to change state.
 * @param enTaskState new task state.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_NOT_FOUND, @ref ERR_UEM_INVALID_HANDLE. \n
 *         @ref ERR_UEM_NOT_FOUND is occurred when the target task is not found in the composite task manager.
 */
uem_result UKCPUCompositeTaskManager_ChangeState(HCPUCompositeTaskManager hManager, STask *pstTargetTask, ECPUTaskState enTaskState);

/**
 * @brief Activate threads in the task.
 *
 * This function activates threads. This function wakes up the threads to do jobs according to their task state.
 *
 * @param hManager a composite task manager handle.
 * @param pstTargetTask target task to activate threads.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_NOT_FOUND, @ref ERR_UEM_INVALID_HANDLE. \n
 *         @ref ERR_UEM_NOT_FOUND is occurred when the target task is not found in the composite task manager.
 */
uem_result UKCPUCompositeTaskManager_ActivateThread(HCPUCompositeTaskManager hManager, STask *pstTargetTask);

/**
 * @brief Get task state.
 *
 * This function retrieves the state of the task.
 *
 * @param hManager a composite task manager handle.
 * @param pstTargetTask target task to get task state.
 * @param[out] penTaskState task state.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_NOT_FOUND, @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INTERNAL_FAIL. \n
 *         @ref ERR_UEM_NOT_FOUND is occurred when the target task is not found in the composite task manager. \n
 *         @ref ERR_UEM_INTERNAL_FAIL is occurred when threads in the task is not terminated yet if the task is control-driven or stopping-state task.
 */
uem_result UKCPUCompositeTaskManager_GetTaskState(HCPUCompositeTaskManager hManager, STask *pstTargetTask, OUT ECPUTaskState *penTaskState);

/**
 * @brief Destroy threads in the task.
 *
 * This function destroys threads in the task. \n
 * After calling this function, @ref UKCPUCompositeTaskManager_CreateThread is needed to re-execute tasks.
 *
 * @param hManager a composite task manager handle.
 * @param pstTargetTask target task to destroy threads.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_NOT_FOUND, @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INTERNAL_FAIL. \n
 *         @ref ERR_UEM_NOT_FOUND is occurred when the target task is not found in the composite task manager. \n
 *         @ref ERR_UEM_INTERNAL_FAIL is occurred when threads in the task is not terminated properly.
 */
uem_result UKCPUCompositeTaskManager_DestroyThread(HCPUCompositeTaskManager hManager, STask *pstTargetTask);

/**
 * @brief Destroy a composite task manager.
 *
 * This function destroys a composite task manager.
 *
 * @param[in,out] phManager a composite task manager handle to be destroyed.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INTERNAL_FAIL. \n
 *         @ref ERR_UEM_INTERNAL_FAIL is occurred when threads in the task is not terminated properly.
 */
uem_result UKCPUCompositeTaskManager_Destroy(IN OUT HCPUCompositeTaskManager *phManager);

/**
 * @brief Check all the composite tasks are stopped.
 *
 * This function checks all the tasks are stopped in the composite task manager.
 *
 * @param hManager a composite task manager handle.
 * @param[out] pbStopped boolean value of all tasks are stopped or not.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INTERNAL_FAIL. \n
 *         @ref ERR_UEM_INTERNAL_FAIL is occurred when threads in the task is not terminated yet if the task is control-driven or stopping-state task.
 */
uem_result UKCPUCompositeTaskManager_CheckAllTaskStopped(HCPUCompositeTaskManager hManager, OUT uem_bool *pbStopped);

/**
 * @brief Get task state. (Model controller callback function)
 *
 * This function retrieves a state of a task. \n
 * This function is called inside a model controller which needs state information.
 *
 * @param pTaskHandle a composite task handle.
 * @param[out] penState  task state.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKCPUCompositeTaskManagerCB_GetTaskState(void *pTaskHandle, OUT ECPUTaskState *penState);

/**
 * @brief Change task state. (Model controller callback function)
 *
 * This function changes a state of a task and a state of all thread in the task. \n
 * This function is called inside a model controller which needs to change state information. \n
 *
 * @param pTaskHandle a composite task handle.
 * @param enState a target task state to chane.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, ERR_UEM_ILLEGAL_CONTROL.
 */
uem_result UKCPUCompositeTaskManagerCB_ChangeTaskState(void *pTaskHandle, ECPUTaskState enState);

/**
 * @brief Update task state based on control request. (Model controller callback function)
 *
 * This function handles a external task state update request and updates a task state.
 *
 * @warning The request is always suspend because only suspend request is set by UKCPUTaskManager. \n
 * Otherwise, the task state is immediately updated.
 *
 * @param pTaskHandle a composite task handle.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKCPUCompositeTaskManagerCB_HandleControlRequest(void *pTaskHandle);

/**
 * @brief Traverse all threads in the task. (Model controller callback function)
 *
 * This function traverses all threads in the composite task.
 *
 * @param pTaskHandle a composite task handle.
 * @param fnCallback callback function during traversing threads in the composite task.
 * @param pUserData user data used in the callback function.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKCPUCompositeTaskManagerCB_TraverseThreadsInTask(void *pTaskHandle, FnTaskThreadTraverse fnCallback, void *pUserData);

/**
 * @brief Activate all threads in the task. (Model controller callback function)
 *
 * This function activates all threads in the task.
 *
 * @param pTaskHandle a composite task handle.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKCPUCompositeTaskManagerCB_WakeUpTask(void *pTaskHandle);

/**
 * @brief Get thread state. (Model controller callback function)
 *
 * This function retrieves a state of a task. \n
 * This function is called inside a model controller which needs state information.
 *
 * @param pThreadHandle a composite task thread handle.
 * @param[out] penState a retrieved thread state
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKCPUCompositeTaskManagerCB_GetThreadState(void *pThreadHandle, OUT ECPUTaskState *penState);

/**
 * @brief Change thread state. (Model controller callback function)
 *
 * This function changes a thread state.
 *
 * @param pThreadHandle a composite task thread handle.
 * @param enState a thread state to be changed.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKCPUCompositeTaskManagerCB_SetThreadState(void *pThreadHandle, ECPUTaskState enState);

/**
 * @brief Get a thread mode ID. (Model controller callback function)
 *
 * This function retrieves a thread mode id which belongs to a specific mode.
 *
 * @param pThreadHandle a composite task thread handle.
 * @param[out] pnModeId a retrieved mode ID.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKCPUCompositeTaskManagerCB_GetThreadModeId(void *pThreadHandle, OUT int *pnModeId);

/**
 * @brief Get thread target throughput. (Model controller callback function)
 *
 * This function retrieves a target thread throughput. \n
 * Because a thread set in the composite task is divided based on thread throughput, \n
 * each thread is belong to a specific target throughput value.
 *
 * @param pThreadHandle a composite task thread handle.
 * @param[out] pnTargetThroughput a retrieved throughput value.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKCPUCompositeTaskManagerCB_GetThreadTargetThroughput(void *pThreadHandle, OUT int *pnTargetThroughput);

/**
 * @brief Get thread iteration. (Model controller callback function)
 *
 * This function retrieves a current iteration of a thread.
 *
 * @param pThreadHandle a composite task thread handle.
 * @param[out] pnThreadIteration a retrieved thread iteration value.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKCPUCompositeTaskManagerCB_GetThreadIteration(void *pThreadHandle, OUT int *pnThreadIteration);

/**
 * @brief Check a thread contains a source task. (Model controller callback function)
 *
 * This function checks a thread contains a source task. \n
 * A task with no inputs in the located task graph is considered as a source task, \n
 * so a source task can have inputs from upper level task graphs.
 *
 * @param pThreadHandle a composite task thread handle.
 * @param[out] pbHasSourceTask a boolean value whether a task is a source task or not.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKCPUCompositeTaskManagerCB_HasSourceTask(void *pThreadHandle, OUT uem_bool *pbHasSourceTask);


#ifdef __cplusplus
}
#endif


#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKCPUCOMPOSITETASKMANAGER_H_ */
