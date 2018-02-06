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

uem_result UKCPUGeneralTaskManager_Create(IN OUT HCPUGeneralTaskManager *phManager);
uem_result UKCPUGeneralTaskManager_RegisterTask(HCPUGeneralTaskManager hManager, SMappedGeneralTaskInfo *pstMappedTask);
uem_result UKCPUGeneralTaskManager_CreateThread(HCPUGeneralTaskManager hManager, STask *pstTargetTask);
uem_result UKCPUGeneralTaskManager_ChangeState(HCPUGeneralTaskManager hManager, STask *pstTargetTask, ECPUTaskState enTaskState);
uem_result UKCPUGeneralTaskManager_ActivateThread(HCPUGeneralTaskManager hManager, STask *pstTargetTask);
uem_result UKCPUGeneralTaskManager_GetTaskState(HCPUGeneralTaskManager hManager, STask *pstTargetTask, ECPUTaskState *penTaskState);
uem_result UKCPUGeneralTaskManager_DestroyThread(HCPUGeneralTaskManager hManager, STask *pstTargetTask);
uem_result UKCPUGeneralTaskManager_Destroy(IN OUT HCPUGeneralTaskManager *phManager);

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKCPUGENERALTASKMANAGER_H_ */
