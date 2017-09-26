/*
 * UKCPUTaskManager.c
 *
 *  Created on: 2017. 9. 19.
 *      Author: jej
 */


#include <uem_common.h>

#include <UCThreadMutex.h>

#include <uem_data.h>

#include <UKCPUTaskManager.h>

typedef enum _ETaskState {
	TASK_STATE_STOP,
	TASK_STATE_RUNNING,
	TASK_STATE_SUSPEND,
} ETaskState;

typedef union _UTargetTask {
	STask *pstTask;
	SScheduledTasks *pstScheduledTasks;
} UTargetTask;

typedef union _UMappedCPUList {
	int *anCPUId;
	HLinkedList hMappedCPUList;
} UMappedCPUList;


typedef struct _STaskThread {
	HThread hThread;
	HThreadEvent hEvent;
	UTargetTask uTargetTask;
	ETaskState enTaskState;
	UMappedCPUList uMappedCPUList;
} STaskThread;

typedef union _UTaskList {
	STaskThread *astTaskThread;
	HLinkedList hTaskList;
} UTaskList;

typedef struct _SCPUTaskManager {
	EUemModuleId enId;
	UTaskList uTaskList;
	uem_bool bListStatic;
	HThreadMutex hMutex;
} SCPUTaskManager;


uem_result UKCPUTaskManager_Create(OUT HCPUTaskManager *phCPUThreadPool)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKCPUTaskManager_RegisterTask(HCPUTaskManager hCPUThreadPool, STask *pstTask, int nCPUId)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKCPUTaskManager_RegisterCompositeTask(HCPUTaskManager hCPUThreadPool, SScheduledTasks *pstScheduledTasks, int nCPUId)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKCPUTaskManager_SuspendTask(HCPUTaskManager hCPUThreadPool, int nTaskId)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKCPUTaskManager_StopTask(HCPUTaskManager hCPUThreadPool, int nTaskId)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKCPUTaskManager_RunTask(HCPUTaskManager hCPUThreadPool, int nTaskId)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKCPUTaskManager_ResumeTask(HCPUTaskManager hCPUThreadPool, int nTaskId)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKCPUTaskManager_Destroy(IN OUT HCPUTaskManager *phCPUThreadPool)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


