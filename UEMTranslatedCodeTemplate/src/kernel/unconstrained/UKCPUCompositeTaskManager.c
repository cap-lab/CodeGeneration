/*
 * UKCPUCompositeTaskManager.c
 *
 *  Created on: 2018. 1. 16.
 *      Author: jej
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <UKCPUTaskManager.h>

typedef struct _SCPUCompositeTaskManager *HCPUCompositeTaskManager;

typedef struct _STaskThread {
	int nSeqId;
	int nModeId;
	int nThroughputConstraint;
	int nProcId;
	FnUemTaskGo fnCompositeGo;
	ECPUTaskState enTaskState;
	HCPUCompositeTaskManager hManager; // handle for accessing higher data structures
} SCompositeTaskThread;


typedef struct _STaskThread {
	STask *pstParentTask;
	HLinkedList hThreadList;
	HThreadEvent hEvent;
	HThreadMutex hMutex;
	int nWaitingThreadNum;
	int nFinishedThreadNum;
	SScheduledTasks *pstScheduledTasks;
	ECPUTaskState enTaskState;
} SCompositeTask;


typedef struct _SCPUCompositeTaskManager {
	EUemModuleId enId;
	HLinkedList hTaskList;
	uem_bool bListStatic;
	HThreadMutex hMutex;
} SCPUCompositeTaskManager;


static uem_result destroyCompositeTaskStruct(IN OUT SCompositeTask **ppstCompositeTask)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCompositeTask *pstCompositeTask = NULL;

	pstCompositeTask = *ppstCompositeTask;

	// TODO: Traverse all thread to be destroyed

	if(pstCompositeTask->hThreadList != NULL)
	{
		UCDynamicLinkedList_Destroy(&(pstCompositeTask->hThreadList));
	}

	if(pstCompositeTask->hEvent != NULL)
	{
		UCThreadEvent_Destroy(&(pstCompositeTask->hEvent));
	}

	if(pstCompositeTask->hMutex != NULL)
	{
		UCThreadMutex_Destroy(&(pstCompositeTask->hMutex));
	}

	SAFEMEMFREE(pstCompositeTask);

	*ppstCompositeTask = NULL;

	result = ERR_UEM_NOERROR;

	return result;
}


static uem_result createCompositeTaskStruct(HCPUTaskManager hCPUTaskManager, SMappedCompositeTaskInfo *pstMappedInfo, OUT SCompositeTask **ppstCompositeTask)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCompositeTask *pstCompositeTask = NULL;

	pstCompositeTask = UC_malloc(sizeof(SCompositeTask));
	ERRMEMGOTO(pstCompositeTask, result, _EXIT);

	pstCompositeTask->hEvent = NULL;
	pstCompositeTask->hMutex = NULL;
	pstCompositeTask->hThreadList = NULL;
	pstCompositeTask->enTaskState = TASK_STATE_STOP;
	pstCompositeTask->nFinishedThreadNum = 0;
	pstCompositeTask->nWaitingThreadNum = 0;
	pstCompositeTask->pstParentTask = pstMappedInfo;

	result = UCThreadEvent_Create(&(pstCompositeTask->hEvent));
	ERRIFGOTO(result, _EXIT);

	result = UCThreadMutex_Create(&(pstCompositeTask->hMutex));
	ERRIFGOTO(result, _EXIT);

	result = UCDynamicLinkedList_Create(&(pstCompositeTask->hThreadList));
	ERRIFGOTO(result, _EXIT);

	//pstCompositeTask->hManager = hCPUTaskManager;

	*ppstCompositeTask = pstCompositeTask;

	result = ERR_UEM_NOERROR;
_EXIT:
	if(result != ERR_UEM_NOERROR && pstCompositeTask != NULL)
	{
		destroyCompositeTaskStruct(&pstCompositeTask);
	}
	return result;
}


uem_result UKCPUCompositeTaskManager_Create(IN OUT HCPUCompositeTaskManager *phManager)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUCompositeTaskManager *pstTaskManager = NULL;

#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(phManager, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	pstTaskManager = UC_malloc(sizeof(SCPUCompositeTaskManager));
	ERRMEMGOTO(pstTaskManager, result, _EXIT);

	pstTaskManager->enId = ID_UEM_CPU_COMPOSITE_TASK_MANAGER;
	pstTaskManager->bListStatic = FALSE;
	pstTaskManager->hMutex = NULL;
	pstTaskManager->hTaskList = NULL;

	result = UCThreadMutex_Create(&(pstTaskManager->hMutex));
	ERRIFGOTO(result, _EXIT);

	result = UCDynamicLinkedList_Create(&(pstTaskManager->hTaskList));
	ERRIFGOTO(result, _EXIT);

	*phManager = pstTaskManager;

	result = ERR_UEM_NOERROR;
_EXIT:
	if(pstTaskManager != NULL && result != ERR_UEM_NOERROR)
	{
		UCDynamicLinkedList_Destroy(&(pstTaskManager->hTaskList));
		UCThreadMutex_Destroy(&(pstTaskManager->hMutex));
		SAFEMEMFREE(pstTaskManager);
	}
	return result;
}



static uem_result traverseCompositeTaskList(IN int nOffset, IN void *pData, IN void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCompositeTask *pstTaskStruct = NULL;
	SMappedCompositeTaskInfo *pstUserData = NULL;

	pstTaskStruct = (SCompositeTask *) pData;
	pstUserData = (SMappedCompositeTaskInfo *) pUserData;

	if(pstTaskStruct->pstParentTask->nTaskId == pstUserData->pstScheduledTasks->pstParentTask->nTaskId)
	{
		UEMASSIGNGOTO(result, ERR_UEM_FOUND_DATA, _EXIT);
	}

    result = ERR_UEM_NOERROR;
_EXIT:
    return result;
}

uem_result UKCPUCompositeTaskManager_RegisterTask(HCPUCompositeTaskManager hManager, SMappedCompositeTaskInfo *pstMappedTask)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUCompositeTaskManager *pstTaskManager = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(pstMappedTask, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	if (IS_VALID_HANDLE(hManager, ID_UEM_CPU_COMPOSITE_TASK_MANAGER) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}
#endif
	pstTaskManager = hManager;

	result = UCDynamicLinkedList_Traverse(pstTaskManager->hTaskList, traverseCompositeTaskList, pstMappedTask);
	ERRIFGOTO(result, _EXIT);



	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}



uem_result UKCPUCompositeTaskManager_CreateThread(HCPUCompositeTaskManager hManager, STask *pstTargetTask)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUCompositeTaskManager *pstTaskManager = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(pstTargetTask, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	if (IS_VALID_HANDLE(hManager, ID_UEM_CPU_COMPOSITE_TASK_MANAGER) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}
#endif
	pstTaskManager = hManager;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKCPUCompositeTaskManager_ChangeState(HCPUCompositeTaskManager hManager, STask *pstTargetTask, , EInternalTaskState enTaskState)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUCompositeTaskManager *pstTaskManager = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(pstTargetTask, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	if (IS_VALID_HANDLE(hManager, ID_UEM_CPU_COMPOSITE_TASK_MANAGER) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}
#endif
	pstTaskManager = hManager;


	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKCPUCompositeTaskManager_ActivateThread(HCPUCompositeTaskManager hManager, STask *pstTargetTask)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUCompositeTaskManager *pstTaskManager = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(pstTargetTask, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	if (IS_VALID_HANDLE(hManager, ID_UEM_CPU_COMPOSITE_TASK_MANAGER) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}
#endif
	pstTaskManager = hManager;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKCPUCompositeTaskManager_GetTaskState(HCPUCompositeTaskManager hManager, STask *pstTargetTask, EInternalTaskState *penTaskState)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUCompositeTaskManager *pstTaskManager = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(pstTargetTask, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	if (IS_VALID_HANDLE(hManager, ID_UEM_CPU_COMPOSITE_TASK_MANAGER) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}
#endif
	pstTaskManager = hManager;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKCPUCompositeTaskManager_DestroyThread(HCPUCompositeTaskManager hManager, STask *pstTargetTask)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUCompositeTaskManager *pstTaskManager = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(pstTargetTask, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	if (IS_VALID_HANDLE(hManager, ID_UEM_CPU_COMPOSITE_TASK_MANAGER) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}
#endif
	pstTaskManager = hManager;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}



uem_result UKCPUCompositeTaskManager_Destroy(IN OUT HCPUCompositeTaskManager *phManager)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUCompositeTaskManager *pstTaskManager = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(phManager, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	if (IS_VALID_HANDLE(*phManager, ID_UEM_CPU_COMPOSITE_TASK_MANAGER) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}
#endif
	pstTaskManager = *phManager;

	//result = UKCPUTaskManager_StopAllTasks(*phManager);
	//ERRIFGOTO(result, _EXIT);

	//result = UCDynamicLinkedList_Traverse(pstTaskManager->hTaskList, traverseAndDestroyTaskThread, NULL);
	//ERRIFGOTO(result, _EXIT);

	result = UCDynamicLinkedList_Destroy(&(pstTaskManager->hTaskList));
	ERRIFGOTO(result, _EXIT);

	result = UCThreadMutex_Destroy(&(pstTaskManager->hMutex));
	ERRIFGOTO(result, _EXIT);

	SAFEMEMFREE(pstTaskManager);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}
