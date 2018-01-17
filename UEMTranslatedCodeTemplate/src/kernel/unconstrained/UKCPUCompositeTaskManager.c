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

typedef struct _SCompositeTaskThread {
	int nSeqId;
	int nModeId;
	int nThroughputConstraint;
	int nProcId;
	FnUemTaskGo fnCompositeGo;
	ECPUTaskState enTaskState;
	HCPUCompositeTaskManager hManager; // handle for accessing higher data structures
} SCompositeTaskThread;


typedef struct _SCompositeTask {
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

struct _SCompositeTaskSearchData {
	SMappedCompositeTaskInfo *pstMappedInfo;
	SCompositeTask *pstMatchingCompositeTask;
};

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

static uem_result createCompositeTaskThreadStructPerSchedule(SMappedCompositeTaskInfo *pstMappedInfo, int nScheduleIndex, OUT SCompositeTaskThread **ppstCompositeTaskThread)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCompositeTaskThread *pstCompositeTaskThread = NULL;

	pstCompositeTaskThread = UC_malloc(sizeof(SCompositeTaskThread));
	ERRMEMGOTO(pstCompositeTaskThread, result, _EXIT);

	pstCompositeTaskThread->enTaskState = TASK_STATE_STOP;
	pstCompositeTaskThread->fnCompositeGo = pstMappedInfo->pstScheduledTasks->astScheduleList[nScheduleIndex].fnCompositeGo;
	pstCompositeTaskThread->nThroughputConstraint = pstMappedInfo->pstScheduledTasks->astScheduleList[nScheduleIndex].nThroughputConstraint;
	pstCompositeTaskThread->hManager = NULL;
	pstCompositeTaskThread->nModeId = pstMappedInfo->pstScheduledTasks->nModeId;
	pstCompositeTaskThread->nProcId = pstMappedInfo->nProcessorId;
	pstCompositeTaskThread->nSeqId = 0;

	*ppstCompositeTaskThread = pstCompositeTaskThread;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result createCompositeTaskThreadStructs(SMappedCompositeTaskInfo *pstMappedInfo, IN OUT HLinkedList hTaskList)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCompositeTaskThread *pstCompositeTaskThread = NULL;
	int nLoop = 0;

	for(nLoop = 0 ; nLoop < pstMappedInfo->pstScheduledTasks->nScheduleNum ; nLoop++)
	{
		result = createCompositeTaskThreadStructPerSchedule(pstMappedInfo, nLoop, &pstCompositeTaskThread);
		ERRIFGOTO(result, _EXIT);

		result = UCDynamicLinkedList_Add(hTaskList, LINKED_LIST_OFFSET_FIRST, 0, pstCompositeTaskThread);
		ERRIFGOTO(result, _EXIT);
	}



	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

// TODO: use hCPUTaskManager?
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
	pstCompositeTask->pstParentTask = pstMappedInfo->pstScheduledTasks->pstParentTask;

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



static uem_result findCompositeTask(IN int nOffset, IN void *pData, IN void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCompositeTask *pstTaskStruct = NULL;
	struct _SCompositeTaskSearchData *pstSearchData = NULL;

	pstTaskStruct = (SCompositeTask *) pData;
	pstSearchData = (struct _SCompositeTaskSearchData *) pUserData;

	if(pstTaskStruct->pstParentTask->nTaskId == pstSearchData->pstMappedInfo->pstScheduledTasks->pstParentTask->nTaskId)
	{
		pstSearchData->pstMatchingCompositeTask = pstTaskStruct;
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
	SCompositeTask *pstCompositeTask = NULL;
	struct _SCompositeTaskSearchData stSearchData;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(pstMappedTask, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	if (IS_VALID_HANDLE(hManager, ID_UEM_CPU_COMPOSITE_TASK_MANAGER) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}
#endif
	pstTaskManager = hManager;

	stSearchData.pstMappedInfo = pstMappedTask;
	stSearchData.pstMatchingCompositeTask = NULL;

	result = UCDynamicLinkedList_Traverse(pstTaskManager->hTaskList, findCompositeTask, &stSearchData);
	ERRIFGOTO(result, _EXIT);

	if(result == ERR_UEM_FOUND_DATA)
	{
		pstCompositeTask = stSearchData.pstMatchingCompositeTask;
	}
	else
	{
		result = createCompositeTaskStruct(hManager, pstMappedTask, &pstCompositeTask);
		ERRIFGOTO(result, _EXIT);

		result = UCDynamicLinkedList_Add(pstTaskManager->hTaskList, LINKED_LIST_OFFSET_FIRST, 0, pstCompositeTask);
		ERRIFGOTO(result, _EXIT);
	}

	result = createCompositeTaskThreadStructs(pstMappedInfo, pstTaskManager->hTaskList);
	ERRIFGOTO(result, _EXIT);

	//result = UCDynamicLinkedList_Add(pstTaskManager->hTaskList, LINKED_LIST_OFFSET_FIRST, 0, pstTaskThread);g
	//ERRIFGOTO(result, _EXIT);

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

uem_result UKCPUCompositeTaskManager_ChangeState(HCPUCompositeTaskManager hManager, STask *pstTargetTask, EInternalTaskState enTaskState)
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
