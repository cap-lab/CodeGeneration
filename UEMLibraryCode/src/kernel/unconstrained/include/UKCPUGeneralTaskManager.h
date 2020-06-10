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
 * @brief Update the mapping info in the task.
 *
 * This function updates the mapping info in the task. \n
 *
 * @param hManager a general task manager handle.
 * @param pstTargetTask target task to update the mapping info.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_NOT_FOUND, @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INVALID_PARAM, \n
 *         @ref ERR_UEM_ILLEGAL_CONTROL. \n
 *         @ref ERR_UEM_NOT_FOUND is occurred when the target task is not found in the general task manager.
 *         @ref ERR_UEM_ILLEGAL_CONTROL is occurred when the processor of the task is not a CPU.
 */
uem_result UKCPUGeneralTaskManager_UpdateTaskMappingInfo(HCPUGeneralTaskManager hManager, STask *pstTargetTask, int nNewLocalId);

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

/**
 * @brief Request to change a task state. (Model controller callback function)
 *
 * This function requests to change a task state. \n
 * This function does not directly changes a task state, but set a flag to change a task state. \n
 * When the target task is called, it will change the requested task state. \n
 * This function is used to change a state of other tasks
 *
 * @param hManager a general task manager handle.
 * @param pstTargetTask a target task to change a state.
 * @param enTaskState a task state to be changed.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_NOT_FOUND.
 *         @ref ERR_UEM_NOT_FOUND is returned when the target task is not available.
 */
uem_result UKCPUGeneralTaskManager_RequestTaskState(HCPUGeneralTaskManager hManager, STask *pstTargetTask, ECPUTaskState enTaskState);

/**
 * @brief Check that a task is a source task. (Model controller callback function)
 *
 * This function checks a current task is a source task.
 *
 * @param pTaskHandle a general task handle.
 * @param[out] pbIsSourceTask a boolean value whether the task is a source task or not.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKCPUGeneralTaskManagerCB_IsSourceTask(void *pTaskHandle, OUT uem_bool *pbIsSourceTask);

/**
 * @brief Retrieve @ref STask structure of a current task. (Model controller callback function)
 *
 * This function retrieves @ref STask structure of a current task.
 *
 * @param pTaskHandle a general task handle.
 * @param[out] ppstTask retrieved @ref STask structure.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKCPUGeneralTaskManagerCB_GetCurrentTaskStructure(void *pTaskHandle, OUT STask **ppstTask);

/**
 * @brief Changes a current task's state. (Model controller callback function)
 *
 * This function changes a current task state.
 *
 * @param pTaskHandle a general task handle.
 * @param enState a state value to be changed.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, errors occurred from FnTraverseModelControllerFunctions callback function.
 */
uem_result UKCPUGeneralTaskManagerCB_ChangeTaskState(void *pTaskHandle, ECPUTaskState enState);

/**
 * @brief Get task state. (Model controller callback function)
 *
 * This function retrieves a state of a current task.
 *
 * @param pTaskHandle a general task handle.
 * @param[out] penState retrieved task state.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKCPUGeneralTaskManagerCB_GetCurrentTaskState(void *pTaskHandle, OUT ECPUTaskState *penState);

/**
 * @brief Retrieve a manager handle. (Model controller callback function)
 *
 * This function retrieves a general task manager handle. \n
 * This function is used to utilize functions provided by a general task manager handle.
 *
 * @param pTaskHandle a general task handle.
 * @param[out] phManager a retrieved general task manager handle.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKCPUGeneralTaskManagerCB_GetManagerHandle(void *pTaskHandle, OUT HCPUGeneralTaskManager *phManager);

/**
 * @brief Activate a current task. (Model controller callback function)
 *
 * This function activates all the threads which belong to a current task.
 *
 * @param pTaskHandle a general task handle.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKCPUGeneralTaskManagerCB_ActivateTask(void *pTaskHandle);

/**
 * @brief Retrieve loop index value. (Model controller callback function)
 *
 * This function retrieves a loop index value which is managed for counting a iteration number \n
 * of each thread in a current task.
 *
 * @param pTaskHandle a general task handle.
 * @param[out] pnLoopIndex retrieved loop index value.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKCPUGeneralTaskManagerCB_GetLoopIndex(void *pTaskHandle, OUT int *pnLoopIndex);

/**
 * @brief Set loop index value. (Model controller callback function)
 *
 * This function sets a new loop index value.
 *
 * @param pTaskHandle a general task handle.
 * @param nLoopIndex a loop index value to be set.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKCPUGeneralTaskManagerCB_SetLoopIndex(void *pTaskHandle, OUT int nLoopIndex);


/**
 * @brief Retrieve a task graph lock handle. (Model controller callback function)
 *
 * This function retrieves a highest task graph lock handle based on a current task.
 *
 * @param pTaskHandle a general task handle.
 * @param[out] phMutex a retrieved lock handle.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKCPUGeneralTaskManagerCB_GetTaskGraphLock(void *pTaskHandle, OUT HThreadMutex *phMutex);

/**
 * @brief Check whether a current task function is called (Model controller callback function)
 *
 * This function checks whether a current task function is called or not. \n
 * A retrieved value is TRUE if go() function is called before calling the model controller function.
 *
 * @param pThreadHandle a general task thread handle.
 * @param[out] pbFunctionCalled a retrieved boolean value whether go() function is called or not.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKCPUGeneralTaskManagerCB_GetFunctionCalled(void *pThreadHandle, OUT uem_bool *pbFunctionCalled);

/**
 * @brief Check whether a task is resumed by external control command. (Model controller callback function)
 *
 * This function checks whether a task is resumed by external control command or not.
 * A retrieved value is TRUE if a suspended task is resumed by calling \ref UKCPUTaskManager_ResumeTask.
 *
 * @param pTaskHandle a general task handle.
 * @param[out] pbResumedByControl a retrieved boolean value whether a task is resumed by external control or not.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKCPUGeneralTaskManagerCB_IsResumedByControl(void *pTaskHandle, OUT uem_bool *pbResumedByControl);

/**
 * @brief Check whether a task is awaken from a suspend state. (Model controller callback function)
 *
 * This function checks whether a task is awaken from a suspend state.
 * A retrieved value is TRUE if a model controller is called after changing from TASK_STATE_SUSPEND to TASK_STATE_RUNNING.
 *
 * @param pThreadHandle a general task thread handle.
 * @param[out] pbRestarted a retrieved boolean value whether a task is awaken from a suspend state or not.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKCPUGeneralTaskManagerCB_GetRestarted(void *pThreadHandle, OUT uem_bool *pbRestarted);

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKCPUGENERALTASKMANAGER_H_ */
