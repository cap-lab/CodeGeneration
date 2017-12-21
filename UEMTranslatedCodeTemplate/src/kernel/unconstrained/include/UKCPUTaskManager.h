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

#ifdef __cplusplus
extern "C"
{
#endif

#define CPU_ID_NOT_SET (-1)

typedef struct _SCPUTaskManager *HCPUTaskManager;

uem_result UKCPUTaskManager_Create(OUT HCPUTaskManager *phCPUThreadPool);
uem_result UKCPUTaskManager_RegisterTask(HCPUTaskManager hCPUThreadPool, STask *pstTask, int nCPUId);
uem_result UKCPUTaskManager_RegisterCompositeTask(HCPUTaskManager hCPUThreadPool, SScheduledTasks *pstScheduledTasks, int nCPUId);
uem_result UKCPUTaskManager_SuspendTask(HCPUTaskManager hCPUThreadPool, int nTaskId);
uem_result UKCPUTaskManager_StopTask(HCPUTaskManager hCPUThreadPool, int nTaskId);
uem_result UKCPUTaskManager_RunTask(HCPUTaskManager hCPUThreadPool, int nTaskId);
uem_result UKCPUTaskManager_ResumeTask(HCPUTaskManager hCPUThreadPool, int nTaskId);
uem_result UKCPUTaskManager_Destroy(IN OUT HCPUTaskManager *phCPUThreadPool);

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKCPUTASKMANAGER_H_ */
