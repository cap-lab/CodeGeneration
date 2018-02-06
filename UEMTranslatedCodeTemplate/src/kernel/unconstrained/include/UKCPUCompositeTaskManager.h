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

uem_result UKCPUCompositeTaskManager_Create(IN OUT HCPUCompositeTaskManager *phManager);
uem_result UKCPUCompositeTaskManager_RegisterTask(HCPUCompositeTaskManager hManager, SMappedCompositeTaskInfo *pstMappedTask);
uem_result UKCPUCompositeTaskManager_CreateThread(HCPUCompositeTaskManager hManager, STask *pstTargetTask);
uem_result UKCPUCompositeTaskManager_ChangeState(HCPUCompositeTaskManager hManager, STask *pstTargetTask, ECPUTaskState enTaskState);
uem_result UKCPUCompositeTaskManager_ActivateThread(HCPUCompositeTaskManager hManager, STask *pstTargetTask);
uem_result UKCPUCompositeTaskManager_GetTaskState(HCPUCompositeTaskManager hManager, STask *pstTargetTask, ECPUTaskState *penTaskState);
uem_result UKCPUCompositeTaskManager_DestroyThread(HCPUCompositeTaskManager hManager, STask *pstTargetTask);
uem_result UKCPUCompositeTaskManager_Destroy(IN OUT HCPUCompositeTaskManager *phManager);

#ifdef __cplusplus
}
#endif


#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKCPUCOMPOSITETASKMANAGER_H_ */
