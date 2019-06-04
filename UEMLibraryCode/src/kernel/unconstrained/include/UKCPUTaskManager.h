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
 * @brief
 *
 * This function
 *
 * @param[out] phCPUTaskManager
 *
 * @return
 */
uem_result UKCPUTaskManager_Create(OUT HCPUTaskManager *phCPUTaskManager);

/**
 * @brief
 *
 * This function
 *
 * @param hCPUTaskManager
 * @param pstMappedTask
 *
 * @return
 */
uem_result UKCPUTaskManager_RegisterTask(HCPUTaskManager hCPUTaskManager, SMappedGeneralTaskInfo *pstMappedTask);

/**
 * @brief
 *
 * This function
 *
 * @param hCPUTaskManager
 * @param pstMappedTask
 *
 * @return
 */
uem_result UKCPUTaskManager_RegisterCompositeTask(HCPUTaskManager hCPUTaskManager, SMappedCompositeTaskInfo *pstMappedTask);

/**
 * @brief
 *
 * This function
 *
 * @param hCPUTaskManager
 *
 * @return
 */
uem_result UKCPUTaskManager_RunRegisteredTasks(HCPUTaskManager hCPUTaskManager);

/**
 * @brief
 *
 * This function
 *
 * @param hCPUTaskManager
 *
 * @return
 */
uem_result UKCPUTaskManager_StopAllTasks(HCPUTaskManager hCPUTaskManager);

/**
 * @brief
 *
 * This function
 *
 * @param hCPUTaskManager
 * @param nTaskId
 *
 * @return
 */
uem_result UKCPUTaskManager_SuspendTask(HCPUTaskManager hCPUTaskManager, int nTaskId);

/**
 * @brief
 *
 * This function
 *
 * @param hCPUTaskManager
 * @param nTaskId
 *
 * @return
 */
uem_result UKCPUTaskManager_StoppingTask(HCPUTaskManager hCPUTaskManager, int nTaskId);

/**
 * @brief
 *
 * This function
 *
 * @param hCPUTaskManager
 * @param nTaskId
 *
 * @return
 */
uem_result UKCPUTaskManager_StopTask(HCPUTaskManager hCPUTaskManager, int nTaskId);

/**
 * @brief
 *
 * This function
 *
 * @param hCPUTaskManager
 * @param nTaskId
 *
 * @return
 */
uem_result UKCPUTaskManager_RunTask(HCPUTaskManager hCPUTaskManager, int nTaskId);

/**
 * @brief
 *
 * This function
 *
 * @param hCPUTaskManager
 * @param nTaskId
 *
 * @return
 */
uem_result UKCPUTaskManager_ResumeTask(HCPUTaskManager hCPUTaskManager, int nTaskId);

/**
 * @brief
 *
 * This function
 *
 * @param hCPUTaskManager
 * @param nTaskId
 * @param[out] penTaskState
 *
 * @return
 */
uem_result UKCPUTaskManager_GetTaskState(HCPUTaskManager hCPUTaskManager, int nTaskId, EInternalTaskState *penTaskState);

/**
 * @brief
 *
 * This function
 *
 * @param hCPUTaskManager
 * @param[out] pbStopped
 *
 * @return
 */
uem_result UKCPUTaskManager_IsAllTaskStopped(HCPUTaskManager hCPUTaskManager, uem_bool *pbStopped);

/**
 * @brief
 *
 * This function
 *
 * @param[in,out] phCPUTaskManager
 *
 * @return
 */
uem_result UKCPUTaskManager_Destroy(IN OUT HCPUTaskManager *phCPUTaskManager);

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKCPUTASKMANAGER_H_ */
