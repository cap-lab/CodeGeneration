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
 * @brief
 *
 * This function
 *
 * @param[out] phManager
 *
 * @return
 */
uem_result UKCPUGeneralTaskManager_Create(IN OUT HCPUGeneralTaskManager *phManager);

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
uem_result UKCPUGeneralTaskManager_RegisterTask(HCPUGeneralTaskManager hManager, SMappedGeneralTaskInfo *pstMappedTask);

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
uem_result UKCPUGeneralTaskManager_TraverseGeneralTaskList(HCPUGeneralTaskManager hManager, CbFnTraverseGeneralTask fnCallback, void *pUserData);

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
uem_result UKCPUGeneralTaskManager_CreateThread(HCPUGeneralTaskManager hManager, STask *pstTargetTask);

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
uem_result UKCPUGeneralTaskManager_ChangeState(HCPUGeneralTaskManager hManager, STask *pstTargetTask, ECPUTaskState enTaskState);

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
uem_result UKCPUGeneralTaskManager_ActivateThread(HCPUGeneralTaskManager hManager, STask *pstTargetTask);

/**
 * @brief
 *
 * This function
 *
 * @param hManager
 * @param pstTargetTask
 * @param nTimeoutInMilliSec
 *
 * @return
 */
uem_result UKCPUGeneralTaskManager_WaitTaskActivated(HCPUGeneralTaskManager hManager, STask *pstTargetTask, int nTimeoutInMilliSec);

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
uem_result UKCPUGeneralTaskManager_GetTaskState(HCPUGeneralTaskManager hManager, STask *pstTargetTask, OUT ECPUTaskState *penTaskState);

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
uem_result UKCPUGeneralTaskManager_CheckAllTaskStopped(HCPUGeneralTaskManager hManager, uem_bool *pbStopped);

/**
 * @brief
 *
 * This function
 *
 * @param hManager
 * @param pstTargetTask
 * @param[out] pbStarted
 *
 * @return
 */
uem_result UKCPUGeneralTaskManager_CheckTaskStarted(HCPUGeneralTaskManager hManager, STask *pstTargetTask, uem_bool *pbStarted);

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
uem_result UKCPUGeneralTaskManager_DestroyThread(HCPUGeneralTaskManager hManager, STask *pstTargetTask);

/**
 * @brief
 *
 * This function
 *
 * @param phManager
 *
 * @return
 */
uem_result UKCPUGeneralTaskManager_Destroy(IN OUT HCPUGeneralTaskManager *phManager);

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKCPUGENERALTASKMANAGER_H_ */
