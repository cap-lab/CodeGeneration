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

uem_result UKCPUTaskManager_Create(OUT HCPUTaskManager *phCPUTaskManager);
//uem_result UKCPUTaskManager_RegisterTask(HCPUTaskManager hCPUTaskManager, STask *pstTask, int nCPUId);
uem_result UKCPUTaskManager_RegisterTask(HCPUTaskManager hCPUTaskManager, SMappedGeneralTaskInfo *pstMappedTask);
//uem_result UKCPUTaskManager_RegisterCompositeTask(HCPUTaskManager hCPUTaskManager, SScheduledTasks *pstScheduledTasks, int nCPUId);
uem_result UKCPUTaskManager_RegisterCompositeTask(HCPUTaskManager hCPUTaskManager, SMappedCompositeTaskInfo *pstMappedTask);
uem_result UKCPUTaskManager_RunRegisteredTasks(HCPUTaskManager hCPUTaskManager);
uem_result UKCPUTaskManager_StopAllTasks(HCPUTaskManager hCPUTaskManager);
uem_result UKCPUTaskManager_SuspendTask(HCPUTaskManager hCPUTaskManager, int nTaskId);
uem_result UKCPUTaskManager_StoppingTask(HCPUTaskManager hCPUTaskManager, int nTaskId);
uem_result UKCPUTaskManager_StopTask(HCPUTaskManager hCPUTaskManager, int nTaskId);
uem_result UKCPUTaskManager_RunTask(HCPUTaskManager hCPUTaskManager, int nTaskId);
uem_result UKCPUTaskManager_ResumeTask(HCPUTaskManager hCPUTaskManager, int nTaskId);
uem_result UKCPUTaskManager_GetTaskState(HCPUTaskManager hCPUTaskManager, int nTaskId, EInternalTaskState *penTaskState);
uem_result UKCPUTaskManager_Destroy(IN OUT HCPUTaskManager *phCPUTaskManager);

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKCPUTASKMANAGER_H_ */
