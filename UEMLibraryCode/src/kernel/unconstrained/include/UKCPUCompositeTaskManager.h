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

/**
 * @brief
 *
 * This function
 *
 * @param[out] phManager
 *
 * @return
 */
uem_result UKCPUCompositeTaskManager_Create(IN OUT HCPUCompositeTaskManager *phManager);

/**
 * @brief
 *
 * This function
 *
 * @param hManager
 * @param pstMappedTask
 *
 * @return
 */
uem_result UKCPUCompositeTaskManager_RegisterTask(HCPUCompositeTaskManager hManager, SMappedCompositeTaskInfo *pstMappedTask);

/**
 * @brief
 *
 * This function
 *
 * @param hManager
 * @param fnCallback
 * @param pUserData
 *
 * @return
 */
uem_result UKCPUCompositeTaskManager_TraverseCompositeTaskList(HCPUCompositeTaskManager hManager, CbFnTraverseCompositeTask fnCallback, void *pUserData);

/**
 * @brief
 *
 * This function
 *
 * @param hManager
 * @param pstTargetTask
 *
 * @return
 */
uem_result UKCPUCompositeTaskManager_CreateThread(HCPUCompositeTaskManager hManager, STask *pstTargetTask);

/**
 * @brief
 *
 * This function
 *
 * @param hManager
 * @param pstTargetTask
 * @param enTaskState
 *
 * @return
 */
uem_result UKCPUCompositeTaskManager_ChangeState(HCPUCompositeTaskManager hManager, STask *pstTargetTask, ECPUTaskState enTaskState);

/**
 * @brief
 *
 * This function
 *
 * @param hManager
 * @param pstTargetTask
 *
 * @return
 */
uem_result UKCPUCompositeTaskManager_ActivateThread(HCPUCompositeTaskManager hManager, STask *pstTargetTask);

/**
 * @brief
 *
 * This function
 *
 * @param hManager
 * @param pstTargetTask
 * @param[out] penTaskState
 *
 * @return
 */
uem_result UKCPUCompositeTaskManager_GetTaskState(HCPUCompositeTaskManager hManager, STask *pstTargetTask, ECPUTaskState *penTaskState);

/**
 * @brief
 *
 * This function
 *
 * @param hManager
 * @param pstTargetTask
 *
 * @return
 */
uem_result UKCPUCompositeTaskManager_DestroyThread(HCPUCompositeTaskManager hManager, STask *pstTargetTask);

/**
 * @brief
 *
 * This function
 *
 * @param phManager
 *
 * @return
 */
uem_result UKCPUCompositeTaskManager_Destroy(IN OUT HCPUCompositeTaskManager *phManager);

/**
 * @brief
 *
 * This function
 *
 * @param hManager
 * @param[out] pbStopped
 *
 * @return
 */
uem_result UKCPUCompositeTaskManager_CheckAllTaskStopped(HCPUCompositeTaskManager hManager, uem_bool *pbStopped);

#ifdef __cplusplus
}
#endif


#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKCPUCOMPOSITETASKMANAGER_H_ */
