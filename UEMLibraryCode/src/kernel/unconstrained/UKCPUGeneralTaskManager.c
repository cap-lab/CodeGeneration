/*
 * UKCPUGeneralTaskManager.c
 *
 *  Created on: 2018. 1. 16.
 *      Author: jej
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <UCBasic.h>
#include <UCAlloc.h>
#include <UCThreadMutex.h>
#include <UCThread.h>
#include <UCDynamicLinkedList.h>
#include <UCTime.h>

#include <uem_data.h>

#include <UKCPUTaskCommon.h>
#include <UKTime.h>
#include <UKChannel_internal.h>
#include <UKModeTransition.h>

#include <UKProcessor.h>

#include <UKCPUGeneralTaskManager.h>

#define THREAD_DESTROY_TIMEOUT (3000)

typedef struct _SGeneralTaskThread {
	int nProcId;
	int nTaskFuncId;
	HThread hThread; // modified
	uem_bool bIsThreadFinished; // modified
	HThreadEvent hEvent;
	uem_bool bSuspended; // modified
} SGeneralTaskThread;

typedef struct _SGeneralTask {
	STask *pstTask;
	HLinkedList hThreadList; // modified
	HThreadMutex hMutex;
	ECPUTaskState enTaskState; // modified
	uem_bool bCreated;
	uem_bool bIsModeTransition;
	STask *pstMTMParentTask;
	uem_bool bMTMSourceTask;
	uem_bool bIsSubLoop;
	uem_bool bIsSubConvergentLoop;
	STask *pstLoopParentTask;
	uem_bool bLoopDesignatedTask;
	int nProcessorId;
	SGenericMapProcessor *pstMapProcessorAPI;
	HCPUGeneralTaskManager hManager;
	int nCurLoopIndex;
} SGeneralTask;

typedef struct _SCPUGeneralTaskManager {
	EUemModuleId enId;
	HLinkedList hTaskList;
	uem_bool bListStatic;
	HThreadMutex hMutex;
	SGeneralTask *pstCachedTask;
} SCPUGeneralTaskManager;


struct _SGeneralTaskCreateData {
	SGeneralTask *pstGeneralTask;
};

struct _SGeneralTaskSearchData {
	int nTargetTaskId;
	SGeneralTask *pstMatchingTask;
};

struct _SGeneralTaskThreadData {
	SGeneralTaskThread *pstTaskThread;
	SGeneralTask *pstGeneralTask;
};

struct _SGeneralTaskStopCheck {
	uem_bool bAllStop;
	SGeneralTask *pstGeneralTask;
	SCPUGeneralTaskManager *pstManager;
};

struct _SGeneralTaskTraverse {
	CbFnTraverseGeneralTask fnCallback;
	void *pUserData;
	SCPUGeneralTaskManager *pstManager;
};

struct _SWaitTaskTraverse {
	long long llLeftTime;
	SCPUGeneralTaskManager *pstManager;
};

struct _STaskThreadDestroyTraverse {
	SGeneralTask *pstGeneralTask;
	SCPUGeneralTaskManager *pstManager;
};


struct _SModeTransitionSetEventCheck {
	SGeneralTask *pstCallerTask;
	int nNewStartIteration;
	int nPrevModeIndex;
	int nNewModeIndex;
	uem_bool bModeChanged;
};

uem_result UKCPUGeneralTaskManager_Create(IN OUT HCPUGeneralTaskManager *phManager)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUGeneralTaskManager *pstManager = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(phManager, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	pstManager = UCAlloc_malloc(sizeof(SCPUGeneralTaskManager));
	ERRMEMGOTO(pstManager, result, _EXIT);

	pstManager->enId = ID_UEM_CPU_GENERAL_TASK_MANAGER;
	pstManager->bListStatic = FALSE;
	pstManager->hMutex = NULL;
	pstManager->hTaskList = NULL;
	pstManager->pstCachedTask = NULL;

	result = UCThreadMutex_Create(&(pstManager->hMutex));
	ERRIFGOTO(result, _EXIT);

	result = UCDynamicLinkedList_Create(&(pstManager->hTaskList));
	ERRIFGOTO(result, _EXIT);

	*phManager = pstManager;

	result = ERR_UEM_NOERROR;
_EXIT:
	if(result != ERR_UEM_NOERROR && pstManager != NULL)
	{
		UCThreadMutex_Destroy(&(pstManager->hMutex));
		SAFEMEMFREE(pstManager);
	}
	return result;
}



static uem_result getCachedTask(SCPUGeneralTaskManager *pstTaskManager, int nTaskId, OUT SGeneralTask **ppstGeneralTask)
{
	SGeneralTask *pstGeneralTask = NULL;
	uem_result result = ERR_UEM_UNKNOWN;

	pstGeneralTask = pstTaskManager->pstCachedTask;

	if(pstGeneralTask == NULL)
	{
		*ppstGeneralTask = NULL;
		result = ERR_UEM_NOT_FOUND;
	}
	else if(pstGeneralTask->pstTask->nTaskId == nTaskId)
	{
		*ppstGeneralTask = pstGeneralTask;
		result = ERR_UEM_NOERROR;
	}
	else
	{
		*ppstGeneralTask = NULL;
		result = ERR_UEM_NOT_FOUND;
	}

	return result;
}

static uem_result findTask(IN int nOffset, IN void *pData, IN void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SGeneralTask *pstTaskStruct = NULL;
	struct _SGeneralTaskSearchData *pstSearchData = NULL;

	pstTaskStruct = (SGeneralTask *) pData;
	pstSearchData = (struct _SGeneralTaskSearchData *) pUserData;

	if(pstTaskStruct->pstTask->nTaskId == pstSearchData->nTargetTaskId)
	{
		pstSearchData->pstMatchingTask = pstTaskStruct;
		UEMASSIGNGOTO(result, ERR_UEM_FOUND_DATA, _EXIT);
	}

    result = ERR_UEM_NOERROR;
_EXIT:
    return result;
}

static uem_result findMatchingGeneralTask(SCPUGeneralTaskManager *pstTaskManager, int nTaskId, OUT SGeneralTask **ppstGeneralTask)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SGeneralTask *pstGeneralTask = NULL;
	struct _SGeneralTaskSearchData stSearchData;

	// check cache
	result = getCachedTask(pstTaskManager, nTaskId, &pstGeneralTask);
	if(result == ERR_UEM_NOT_FOUND)
	{
		stSearchData.nTargetTaskId = nTaskId;
		stSearchData.pstMatchingTask = NULL;

		result = UCDynamicLinkedList_Traverse(pstTaskManager->hTaskList, findTask, &stSearchData);
		ERRIFGOTO(result, _EXIT);

		if(result == ERR_UEM_FOUND_DATA)
		{
			pstGeneralTask = stSearchData.pstMatchingTask;
			pstTaskManager->pstCachedTask = pstGeneralTask;
		}
		else
		{
			UEMASSIGNGOTO(result, ERR_UEM_NOT_FOUND, _EXIT);
		}
	}
	ERRIFGOTO(result, _EXIT);

	*ppstGeneralTask = pstGeneralTask;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static uem_result destroyGeneralTaskThreadStruct(IN int nOffset, IN void *pData, IN void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SGeneralTaskThread *pstGeneralTaskThread = NULL;

	pstGeneralTaskThread = (SGeneralTaskThread *) pData;

	if(pstGeneralTaskThread->hEvent != NULL)
	{
		result = UCThreadEvent_Destroy(&(pstGeneralTaskThread->hEvent));
		ERRIFGOTO(result, _EXIT);
	}

	SAFEMEMFREE(pstGeneralTaskThread);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result destroyGeneralTaskStruct(IN OUT SGeneralTask **ppstGeneralTask)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SGeneralTask *pstGeneralTask = NULL;

	pstGeneralTask = *ppstGeneralTask;

	if(pstGeneralTask->hThreadList != NULL)
	{
		// Traverse all thread to be destroyed
		UCDynamicLinkedList_Traverse(pstGeneralTask->hThreadList, destroyGeneralTaskThreadStruct, NULL);

		UCDynamicLinkedList_Destroy(&(pstGeneralTask->hThreadList));
	}

	if(pstGeneralTask->hMutex != NULL)
	{
		UCThreadMutex_Destroy(&(pstGeneralTask->hMutex));
	}

	SAFEMEMFREE(pstGeneralTask);

	*ppstGeneralTask = NULL;

	result = ERR_UEM_NOERROR;

	return result;
}

static uem_bool isModeTransitionTask(STask *pstTask, OUT STask **ppstMTMTask)
{
	STask *pstCurrentTask = NULL;
	uem_bool bIsModeTransition = FALSE;

	pstCurrentTask = pstTask->pstParentGraph->pstParentTask;

	while(pstCurrentTask != NULL)
	{
		if(pstCurrentTask->pstMTMInfo != NULL && pstCurrentTask->pstMTMInfo->nNumOfModes > 1)
		{
			*ppstMTMTask = pstCurrentTask;
			bIsModeTransition = TRUE;
			break;
		}

		pstCurrentTask = pstCurrentTask->pstParentGraph->pstParentTask;
	}

	if(bIsModeTransition == FALSE)
	{
		*ppstMTMTask = NULL;
	}

	return bIsModeTransition;
}

static uem_bool isSubLoopTask(STask *pstTask, OUT STask **ppstLoopTask)
{
	STask *pstCurrentTask = NULL;
	uem_bool bIsSubLoop = FALSE;

	pstCurrentTask = pstTask->pstParentGraph->pstParentTask;

	while(pstCurrentTask != NULL)
	{
		if(pstCurrentTask->pstLoopInfo != NULL)
		{
			*ppstLoopTask = pstCurrentTask;
			bIsSubLoop = TRUE;
			break;
		}

		pstCurrentTask = pstCurrentTask->pstParentGraph->pstParentTask;
	}

	if(bIsSubLoop == FALSE)
	{
		*ppstLoopTask = NULL;
	}

	return bIsSubLoop;
}

static uem_bool isSubConvergentLoopTask(STask *pstTask)
{
	STask *pstCurrentTask = NULL;
	uem_bool bIsSubLoop = FALSE;

	pstCurrentTask = pstTask->pstParentGraph->pstParentTask;

	while(pstCurrentTask != NULL)
	{
		if(pstCurrentTask->pstLoopInfo != NULL && pstCurrentTask->pstLoopInfo->enType == LOOP_TYPE_CONVERGENT)
		{
			bIsSubLoop = TRUE;
			break;
		}

		pstCurrentTask = pstCurrentTask->pstParentGraph->pstParentTask;
	}

	return bIsSubLoop;
}


static uem_result createGeneralTaskStruct(HCPUGeneralTaskManager hCPUTaskManager, SMappedGeneralTaskInfo *pstMappedInfo, OUT SGeneralTask **ppstGeneralTask)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SGeneralTask *pstGeneralTask = NULL;

	pstGeneralTask = UCAlloc_malloc(sizeof(SGeneralTask));
	ERRMEMGOTO(pstGeneralTask, result, _EXIT);

	pstGeneralTask->hMutex = NULL;
	pstGeneralTask->hThreadList = NULL;
	pstGeneralTask->enTaskState = TASK_STATE_STOP;
	pstGeneralTask->bCreated = FALSE;
	pstGeneralTask->pstMTMParentTask = NULL;
	pstGeneralTask->bIsModeTransition = isModeTransitionTask(pstMappedInfo->pstTask, &(pstGeneralTask->pstMTMParentTask));
	pstGeneralTask->bIsSubLoop = isSubLoopTask(pstMappedInfo->pstTask, &(pstGeneralTask->pstLoopParentTask));
	pstGeneralTask->bIsSubConvergentLoop = isSubConvergentLoopTask(pstMappedInfo->pstTask);
	pstGeneralTask->pstTask = pstMappedInfo->pstTask;
	pstGeneralTask->nProcessorId = pstMappedInfo->nProcessorId;
	pstGeneralTask->pstMapProcessorAPI = pstMappedInfo->pstMapProcessorAPI;
	pstGeneralTask->nCurLoopIndex = 0;

	if(pstGeneralTask->bIsModeTransition == TRUE)
	{
		pstGeneralTask->bMTMSourceTask = UKChannel_IsTaskSourceTask(pstGeneralTask->pstTask->nTaskId);
	}
	else
	{
		pstGeneralTask->bMTMSourceTask = FALSE;
	}

	if(pstGeneralTask->bIsSubLoop == TRUE && pstGeneralTask->pstTask->nTaskId == pstGeneralTask->pstLoopParentTask->pstLoopInfo->nDesignatedTaskId){
		pstGeneralTask->bLoopDesignatedTask = TRUE;
	}
	else
	{
		pstGeneralTask->bLoopDesignatedTask = FALSE;
	}

	if(pstMappedInfo->pstTask->enRunCondition == RUN_CONDITION_DATA_DRIVEN ||
		pstMappedInfo->pstTask->enRunCondition == RUN_CONDITION_TIME_DRIVEN)
	{
		pstGeneralTask->enTaskState = TASK_STATE_RUNNING;
	}
	else
	{
		pstGeneralTask->enTaskState = TASK_STATE_STOP;
	}

	result = UCThreadMutex_Create(&(pstGeneralTask->hMutex));
	ERRIFGOTO(result, _EXIT);

	result = UCDynamicLinkedList_Create(&(pstGeneralTask->hThreadList));
	ERRIFGOTO(result, _EXIT);

	pstGeneralTask->hManager = hCPUTaskManager;

	*ppstGeneralTask = pstGeneralTask;

	result = ERR_UEM_NOERROR;
_EXIT:
	if(result != ERR_UEM_NOERROR && pstGeneralTask != NULL)
	{
		destroyGeneralTaskStruct(&pstGeneralTask);
	}
	return result;
}


static uem_result createGeneralTaskThreadStructs(SMappedGeneralTaskInfo *pstMappedInfo, OUT SGeneralTaskThread **ppstGeneralTaskThread)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SGeneralTaskThread *pstGeneralTaskThread = NULL;

	pstGeneralTaskThread = UCAlloc_malloc(sizeof(SGeneralTaskThread));
	ERRMEMGOTO(pstGeneralTaskThread, result, _EXIT);

	pstGeneralTaskThread->nProcId = pstMappedInfo->nLocalId;
	pstGeneralTaskThread->hThread = NULL;
	pstGeneralTaskThread->bIsThreadFinished = TRUE;
	pstGeneralTaskThread->hEvent = NULL;
	pstGeneralTaskThread->bSuspended = FALSE;
	pstGeneralTaskThread->nTaskFuncId = 0;

	*ppstGeneralTaskThread = pstGeneralTaskThread;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKCPUGeneralTaskManager_RegisterTask(HCPUGeneralTaskManager hManager, SMappedGeneralTaskInfo *pstMappedTask)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUGeneralTaskManager *pstTaskManager = NULL;
	SGeneralTask *pstGeneralTask = NULL;
	SGeneralTaskThread *pstGeneralTaskThread = NULL;
	int nLen = 0;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(pstMappedTask, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	if (IS_VALID_HANDLE(hManager, ID_UEM_CPU_GENERAL_TASK_MANAGER) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}
#endif
	pstTaskManager = hManager;

	result = findMatchingGeneralTask(pstTaskManager, pstMappedTask->pstTask->nTaskId, &pstGeneralTask);
	if(result == ERR_UEM_NOT_FOUND)
	{
		result = createGeneralTaskStruct(hManager, pstMappedTask, &pstGeneralTask);
		ERRIFGOTO(result, _EXIT);

		result = UCDynamicLinkedList_Add(pstTaskManager->hTaskList, LINKED_LIST_OFFSET_FIRST, 0, pstGeneralTask);
	}
	ERRIFGOTO(result, _EXIT);

	result = createGeneralTaskThreadStructs(pstMappedTask, &pstGeneralTaskThread);
	ERRIFGOTO(result, _EXIT);

	result = UCDynamicLinkedList_GetLength(pstGeneralTask->hThreadList, &nLen);
	ERRIFGOTO(result, _EXIT);

	pstGeneralTaskThread->nTaskFuncId = nLen;

	result = UCDynamicLinkedList_Add(pstGeneralTask->hThreadList, LINKED_LIST_OFFSET_FIRST, 0, pstGeneralTaskThread);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result traverseGeneralTaskList(IN int nOffset, IN void *pData, IN void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SGeneralTask *pstGeneralTask = NULL;
	struct _SGeneralTaskTraverse *pstUserData = NULL;
	SCPUGeneralTaskManager *pstTaskManager = NULL;

	pstGeneralTask = (SGeneralTask *) pData;
	pstUserData = (struct _SGeneralTaskTraverse *) pUserData;

	pstTaskManager = (SCPUGeneralTaskManager *) pstUserData->pstManager;

	result = UCThreadMutex_Unlock(pstTaskManager->hMutex);
	ERRIFGOTO(result, _EXIT);

	result = pstUserData->fnCallback(pstGeneralTask->pstTask, pstUserData->pUserData);
	ERRIFGOTO(result, _EXIT_LOCK);

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
	UCThreadMutex_Lock(pstTaskManager->hMutex);
_EXIT:
	return result;
}


uem_result UKCPUGeneralTaskManager_TraverseGeneralTaskList(HCPUGeneralTaskManager hManager, CbFnTraverseGeneralTask fnCallback, void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUGeneralTaskManager *pstTaskManager = NULL;
	struct _SGeneralTaskTraverse stUserData;

#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(fnCallback, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	if (IS_VALID_HANDLE(hManager, ID_UEM_CPU_GENERAL_TASK_MANAGER) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}
#endif
	pstTaskManager = hManager;

	stUserData.fnCallback = fnCallback;
	stUserData.pUserData = pUserData;
	stUserData.pstManager = pstTaskManager;

	result = UCThreadMutex_Lock(pstTaskManager->hMutex);
	ERRIFGOTO(result, _EXIT);

	result = UCDynamicLinkedList_Traverse(pstTaskManager->hTaskList, traverseGeneralTaskList, &stUserData);
	ERRIFGOTO(result, _EXIT_LOCK);

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
	UCThreadMutex_Unlock(pstTaskManager->hMutex);
_EXIT:
	return result;
}



static uem_result waitRunSignal(SGeneralTask *pstGeneralTask, SGeneralTaskThread *pstTaskThread, uem_bool bStartWait, OUT long long *pllNextTime, OUT int *pnNextMaxRunCount)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstCurrentTask = NULL;
	long long llCurTime = 0;

	pstCurrentTask = pstGeneralTask->pstTask;

	result = UCThreadEvent_WaitEvent(pstTaskThread->hEvent);
	ERRIFGOTO(result, _EXIT);

	if(pstGeneralTask->enTaskState == TASK_STATE_RUNNING || bStartWait == TRUE)
	{
		result = UCThreadMutex_Lock(pstGeneralTask->hMutex);
		ERRIFGOTO(result, _EXIT);

		pstTaskThread->bSuspended = FALSE;

		result = UCThreadMutex_Unlock(pstGeneralTask->hMutex);
		ERRIFGOTO(result, _EXIT);

		if(pstCurrentTask != NULL && pstCurrentTask->enRunCondition == RUN_CONDITION_TIME_DRIVEN)
		{
			result = UCTime_GetCurTickInMilliSeconds(&llCurTime);
			ERRIFGOTO(result, _EXIT);

			result = UKTime_GetNextTimeByPeriod(llCurTime, pstCurrentTask->nPeriod, pstCurrentTask->enPeriodMetric,
																pllNextTime, pnNextMaxRunCount);
			ERRIFGOTO(result, _EXIT);
		}
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result traverseAndSetEventToTemporarySuspendedMTMTask(STask *pstTask, void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SModeTransitionSetEventCheck *pstNewModeData = NULL;
	HCPUGeneralTaskManager hManager = NULL;
	ECPUTaskState enState;
	char *pszOldModeName = NULL;
	char *pszCurModeName = NULL;
	uem_bool bCurrentPortAvailable = FALSE;

	pstNewModeData = (struct _SModeTransitionSetEventCheck *) pUserData;

	hManager = pstNewModeData->pstCallerTask->hManager;

	result = UKCPUGeneralTaskManager_GetTaskState(hManager, pstTask, &enState);
	ERRIFGOTO(result, _EXIT);

	if(enState == TASK_STATE_SUSPEND && pstNewModeData->pstCallerTask->pstTask->nTaskId != pstTask->nTaskId)
	{
		pszCurModeName = pstNewModeData->pstCallerTask->pstMTMParentTask->pstMTMInfo->astModeMap[pstNewModeData->nNewModeIndex].pszModeName;

		bCurrentPortAvailable = UKChannel_IsPortRateAvailableTask(pstTask->nTaskId, pszCurModeName);
		//UEM_DEBUG_PRINT("task: %s, available: %d, mode_name: %s\n", pstTask->pszTaskName, bCurrentPortAvailable, pszCurModeName);

		if(pstNewModeData->bModeChanged == TRUE)
		{
			pszOldModeName = pstNewModeData->pstCallerTask->pstMTMParentTask->pstMTMInfo->astModeMap[pstNewModeData->nPrevModeIndex].pszModeName;

			if(UKChannel_IsPortRateAvailableTask(pstTask->nTaskId, pszOldModeName) == FALSE &&
				bCurrentPortAvailable == TRUE)
			{
				//UEM_DEBUG_PRINT("new task: %s, previous_iteration: %d, new_iteration: %d\n", pstTask->pszTaskName, pstTask->nCurIteration, pstNewModeData->nNewStartIteration);

				pstTask->nCurIteration = pstNewModeData->nNewStartIteration;
			}
		}

		if(bCurrentPortAvailable == TRUE)
		{
			result = UKCPUGeneralTaskManager_ChangeState(hManager, pstTask, TASK_STATE_RUNNING);
			ERRIFGOTO(result, _EXIT);

			result = UKCPUGeneralTaskManager_ActivateThread(hManager, pstTask);
			ERRIFGOTO(result, _EXIT);
		}
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result traverseAndSetIsSuspended(IN int nOffset, IN void *pData, IN void *pUserData)
{
	SGeneralTaskThread *pstTaskThread = NULL;

	pstTaskThread = (SGeneralTaskThread *) pData;
	pstTaskThread->bSuspended = TRUE;

	return ERR_UEM_NOERROR;
}


static uem_result changeTaskState(SGeneralTask *pstGeneralTask, ECPUTaskState enTaskState)
{
	uem_result result = ERR_UEM_UNKNOWN;
	char *pszModeName = NULL;
	int nCurModeIndex = 0;

	result = UCThreadMutex_Lock(pstGeneralTask->hMutex);
	ERRIFGOTO(result, _EXIT);

	result = UKCPUTaskCommon_CheckTaskState(pstGeneralTask->enTaskState, enTaskState);
	ERRIFGOTO(result, _EXIT_LOCK);

	if(result != ERR_UEM_NOERROR)
	{
		UEMASSIGNGOTO(result, ERR_UEM_NOERROR, _EXIT_LOCK);
	}

	// change task state to suspend when the target task is included in MTM task graph and is not a source task.
	if(pstGeneralTask->bIsModeTransition == TRUE && enTaskState == TASK_STATE_RUNNING &&
	UKModeTransition_GetModeStateInternal(pstGeneralTask->pstMTMParentTask->pstMTMInfo) == MODE_STATE_TRANSITING &&
	pstGeneralTask->bMTMSourceTask == FALSE)
	{
		enTaskState = TASK_STATE_SUSPEND;
	}

	if(enTaskState == TASK_STATE_SUSPEND)
	{
		result = UCDynamicLinkedList_Traverse(pstGeneralTask->hThreadList, traverseAndSetIsSuspended, NULL);
		ERRIFGOTO(result, _EXIT_LOCK);
	}

	if(pstGeneralTask->enTaskState == TASK_STATE_SUSPEND && enTaskState == TASK_STATE_STOPPING && pstGeneralTask->bIsModeTransition == TRUE)
	{
		result = UKModeTransition_GetCurrentModeIndexByIteration(pstGeneralTask->pstMTMParentTask->pstMTMInfo, pstGeneralTask->pstTask->nCurIteration, &nCurModeIndex);
		if(result == ERR_UEM_NOT_FOUND)
		{
			pszModeName = pstGeneralTask->pstMTMParentTask->pstMTMInfo->astModeMap[nCurModeIndex].pszModeName;
			if(UKChannel_IsPortRateAvailableTask(pstGeneralTask->pstTask->nTaskId, pszModeName) == FALSE)
			{
				pstGeneralTask->pstTask->nCurIteration = pstGeneralTask->pstTask->nTargetIteration;
			}
		}
	}

	pstGeneralTask->enTaskState = enTaskState;

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
	UCThreadMutex_Unlock(pstGeneralTask->hMutex);
_EXIT:
	return result;
}


static uem_result updateCurrentIteration(SModeTransitionMachine *pstMTMInfo, STask *pstTask)
{
	uem_result result = ERR_UEM_UNKNOWN;

	int nModeIndex;
	char *pszCurModeName = NULL;
	uem_bool bCurrentPortAvailable = FALSE;

	result = UKModeTransition_GetCurrentModeIndexByIteration(pstMTMInfo, pstTask->nCurIteration, &nModeIndex);
	ERRIFGOTO(result, _EXIT);

	while(bCurrentPortAvailable == FALSE)
	{
		pszCurModeName = pstMTMInfo->astModeMap[nModeIndex].pszModeName;

		bCurrentPortAvailable = UKChannel_IsPortRateAvailableTask(pstTask->nTaskId, pszCurModeName);
		if(bCurrentPortAvailable == FALSE)
		{
			result = UKModeTransition_GetNextModeStartIndexByIteration(pstMTMInfo, pstTask->nCurIteration,
																		&nModeIndex, &(pstTask->nCurIteration));
			if(result == ERR_UEM_NO_DATA)
			{
				pstTask->nCurIteration = pstMTMInfo->nCurrentIteration;
				break;
			}
			ERRIFGOTO(result, _EXIT);
		}
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result handleTaskModeTransition(SGeneralTaskThread *pstTaskThread, SGeneralTask *pstGeneralTask)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstMTMTask = NULL;
	struct _SModeTransitionSetEventCheck stNewModeData;
	EModeState enModeState;

	pstMTMTask = pstGeneralTask->pstMTMParentTask;

	result = UCThreadMutex_Lock(pstMTMTask->hMutex);
	ERRIFGOTO(result, _EXIT);

	if(pstGeneralTask->bMTMSourceTask == TRUE)
	{
		pstMTMTask->pstMTMInfo->fnTransition(pstMTMTask->pstMTMInfo);

		enModeState = UKModeTransition_GetModeStateInternal(pstMTMTask->pstMTMInfo);

		stNewModeData.nPrevModeIndex = pstMTMTask->pstMTMInfo->nCurModeIndex;

		if(enModeState == MODE_STATE_TRANSITING)
		{
			enModeState = UKModeTransition_UpdateModeStateInternal(pstMTMTask->pstMTMInfo, MODE_STATE_NORMAL, pstGeneralTask->pstTask->nCurIteration-1);

			stNewModeData.bModeChanged = TRUE;
		}
		else
		{
			stNewModeData.bModeChanged = FALSE;
		}

		stNewModeData.nNewModeIndex = pstMTMTask->pstMTMInfo->nCurModeIndex;

		stNewModeData.pstCallerTask = pstGeneralTask;
		stNewModeData.nNewStartIteration = pstGeneralTask->pstTask->nCurIteration-1;

		result = UKCPUTaskCommon_TraverseSubGraphTasks(pstMTMTask, traverseAndSetEventToTemporarySuspendedMTMTask, &stNewModeData);
		ERRIFGOTO(result, _EXIT_LOCK);

		pstMTMTask->pstMTMInfo->nCurrentIteration = pstGeneralTask->pstTask->nCurIteration;
	}
	else
	{
		if(pstMTMTask->pstMTMInfo->nCurrentIteration <= pstGeneralTask->pstTask->nCurIteration)
		{
			result = UCThreadMutex_Unlock(pstMTMTask->hMutex);
			ERRIFGOTO(result, _EXIT);

			result = changeTaskState(pstGeneralTask, TASK_STATE_SUSPEND);
			ERRIFGOTO(result, _EXIT);

			result = UCThreadMutex_Lock(pstMTMTask->hMutex);
			ERRIFGOTO(result, _EXIT);
		}
		else
		{
			result = updateCurrentIteration(pstMTMTask->pstMTMInfo, pstGeneralTask->pstTask);
			ERRIFGOTO(result, _EXIT_LOCK);

			if(pstMTMTask->pstMTMInfo->nCurrentIteration <= pstGeneralTask->pstTask->nCurIteration)
			{
				result = UCThreadMutex_Unlock(pstMTMTask->hMutex);
				ERRIFGOTO(result, _EXIT);

				result = changeTaskState(pstGeneralTask, TASK_STATE_SUSPEND);
				ERRIFGOTO(result, _EXIT);

				result = UCThreadMutex_Lock(pstMTMTask->hMutex);
				ERRIFGOTO(result, _EXIT);
			}
		}
	}

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
	UCThreadMutex_Unlock(pstMTMTask->hMutex);
_EXIT:
	return result;
}


static uem_result traverseAndSetEventToTemporarySuspendedTask(STask *pstTask, void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	HCPUGeneralTaskManager hManager = NULL;
	ECPUTaskState enState;

	hManager = (HCPUGeneralTaskManager) pUserData;

	result = UKCPUGeneralTaskManager_GetTaskState(hManager, pstTask, &enState);
	ERRIFGOTO(result, _EXIT);

	if(enState == TASK_STATE_SUSPEND)
	{
		result = UKCPUGeneralTaskManager_ChangeState(hManager, pstTask, TASK_STATE_RUNNING);
		ERRIFGOTO(result, _EXIT);

		result = UKCPUGeneralTaskManager_ActivateThread(hManager, pstTask);
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static uem_result traverseAndSetEventToStopTask(STask *pstTask, void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	HCPUGeneralTaskManager hManager = NULL;
	ECPUTaskState enState;

	hManager = (HCPUGeneralTaskManager) pUserData;

	result = UKCPUGeneralTaskManager_GetTaskState(hManager, pstTask, &enState);
	ERRIFGOTO(result, _EXIT);

	if(pstTask->nCurIteration >= pstTask->nTargetIteration && enState == TASK_STATE_SUSPEND)
	{
		result = UKCPUGeneralTaskManager_ChangeState(hManager, pstTask, TASK_STATE_STOP);
		ERRIFGOTO(result, _EXIT);

		result = UKCPUGeneralTaskManager_ActivateThread(hManager, pstTask);
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static uem_result setLoopTaskCurrentIteration(STask *pstTask, void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstParentTask = NULL;
	SLoopInfo *pstLoopInfo = NULL;
	int nSavedIteration = 1;
	int nLoop = 0;
	int nHistoryEnd;
	int nCheckNum = 0;
	STask *pstParentLoopTask = NULL;
	int nNumOfDataToPop = 0;

	pstParentTask = pstTask->pstParentGraph->pstParentTask;
	pstParentLoopTask = (STask *) pUserData;

	while(pstParentTask != NULL )
	{
		if(pstParentTask->pstLoopInfo != NULL)
		{
			if(pstParentTask->pstLoopInfo->enType == LOOP_TYPE_CONVERGENT)
			{
				pstLoopInfo = pstParentTask->pstLoopInfo;

				nHistoryEnd = pstLoopInfo->nCurHistoryStartIndex + pstLoopInfo->nCurHistoryLen - 1;

				if(nHistoryEnd >= LOOP_HISTORY_ARRAY_SIZE)
				{
					nHistoryEnd = nHistoryEnd - LOOP_HISTORY_ARRAY_SIZE;
				}

				result = UCThreadMutex_Lock(pstTask->hMutex);
				ERRIFGOTO(result, _EXIT);
	
				for(nLoop = nHistoryEnd; nCheckNum < pstLoopInfo->nCurHistoryLen ; nLoop--)
				{
					//UEM_DEBUG_PRINT("pstLoopInfo->astLoopIteration[%d]: prev: %d, next: %d, nCurrentIteration: %d\n", nLoop, pstTask->nCurIteration == pstLoopInfo->astLoopIteration[nLoop].nPrevIteration, pstTask->nCurIteration = pstLoopInfo->astLoopIteration[nLoop].nNextIteration, nCurrentIteration);
					if (pstTask->nCurIteration > pstLoopInfo->astLoopIteration[nLoop].nPrevIteration * nSavedIteration &&
						pstTask->nCurIteration < pstLoopInfo->astLoopIteration[nLoop].nNextIteration * nSavedIteration) {
						nNumOfDataToPop = pstLoopInfo->astLoopIteration[nLoop].nNextIteration * nSavedIteration - pstTask->nCurIteration;
						pstTask->nCurIteration = pstLoopInfo->astLoopIteration[nLoop].nNextIteration * nSavedIteration;
						break;
					}
					else if(pstTask->nCurIteration >= pstLoopInfo->astLoopIteration[nLoop].nNextIteration * nSavedIteration)
					{
						break;
					}

					if (nLoop <= 0) {
						nLoop = LOOP_HISTORY_ARRAY_SIZE;
					}
					nCheckNum++;
				}

				result = UCThreadMutex_Unlock(pstTask->hMutex);
				ERRIFGOTO(result, _EXIT);

			}
			nSavedIteration = nSavedIteration * pstParentTask->pstLoopInfo->nLoopCount;
		}
		pstParentTask = pstParentTask->pstParentGraph->pstParentTask;
	}

	if(pstParentLoopTask->nTaskId != pstTask->nTaskId && nNumOfDataToPop > 0)
	{
		result = UKChannel_PopLoopTaskBroadcastingDataFromQueueByTaskId(pstParentLoopTask->nTaskId, pstTask->nTaskId, nNumOfDataToPop);
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static uem_result updateLoopIterationHistory(SGeneralTask *pstGeneralTask)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SLoopInfo *pstLoopInfo = NULL;
	int nHistoryEnd;
	int nLoopCount = 0;
	int nCurIteration = 0;
	int nTargetIteration = 0;

	pstLoopInfo = pstGeneralTask->pstLoopParentTask->pstLoopInfo;
	nCurIteration = pstLoopInfo->nCurrentIteration;
	nTargetIteration = pstGeneralTask->pstTask->nTargetIteration;
	nLoopCount = pstLoopInfo->nLoopCount;

	if (pstLoopInfo->nCurHistoryLen < LOOP_HISTORY_ARRAY_SIZE) {
		pstLoopInfo->nCurHistoryLen++;
	}
	else {
		pstLoopInfo->nCurHistoryStartIndex++;
		if (pstLoopInfo->nCurHistoryStartIndex >= LOOP_HISTORY_ARRAY_SIZE) {
			pstLoopInfo->nCurHistoryStartIndex = 0;
		}
	}

	nHistoryEnd = pstLoopInfo->nCurHistoryStartIndex + pstLoopInfo->nCurHistoryLen - 1;
	if (nHistoryEnd >= LOOP_HISTORY_ARRAY_SIZE) {
		nHistoryEnd -= LOOP_HISTORY_ARRAY_SIZE;
	}

	pstLoopInfo->astLoopIteration[nHistoryEnd].nPrevIteration = nCurIteration;
	pstLoopInfo->astLoopIteration[nHistoryEnd].nNextIteration = nCurIteration - (nCurIteration % nLoopCount) + nLoopCount;

	if(pstLoopInfo->astLoopIteration[nHistoryEnd].nNextIteration > nTargetIteration)
	{
		pstLoopInfo->astLoopIteration[nHistoryEnd].nNextIteration = nTargetIteration;
	}

	if(pstLoopInfo->astLoopIteration[nHistoryEnd].nPrevIteration > pstLoopInfo->astLoopIteration[nHistoryEnd].nNextIteration)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
	}
	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result clearGeneralTaskData(SGeneralTask *pstGeneralTask)
{
	uem_result result = ERR_UEM_UNKNOWN;

	pstGeneralTask->nCurLoopIndex = 0;

	result = ERR_UEM_NOERROR;

	return result;
}


static uem_bool compareIterationtoAllParentLoopTask(STask *pstTask)
{
	STask *pstCurrentTask = NULL;
	uem_bool bNeedtoSuspend = FALSE;
	int nCurIteration = 0;

	pstCurrentTask = pstTask->pstParentGraph->pstParentTask;
	nCurIteration = pstTask->nCurIteration;

	while(pstCurrentTask != NULL)
	{
		if(pstCurrentTask->pstLoopInfo != NULL)
		{
			if(pstCurrentTask->pstLoopInfo->enType == LOOP_TYPE_CONVERGENT)
			{
				if(pstCurrentTask->pstLoopInfo->nCurrentIteration < nCurIteration)
				{
					bNeedtoSuspend = TRUE;
					break;
				}
			}
			nCurIteration = nCurIteration / pstCurrentTask->pstLoopInfo->nLoopCount;
		}

		pstCurrentTask = pstCurrentTask->pstParentGraph->pstParentTask;
	}

	return bNeedtoSuspend;
}

static uem_result handleLoopTaskIteration(SGeneralTaskThread *pstTaskThread, SGeneralTask *pstGeneralTask)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstCurrentTask = NULL;
	STask *pstParentTask = NULL;
	int nLoopCount = 0;
	int nCurIteration = 0;
	ECPUTaskState enNewState;
	uem_bool bStateChangeNeeded = FALSE;


	pstCurrentTask = pstGeneralTask->pstTask;
	pstParentTask = pstGeneralTask->pstLoopParentTask;

	result = UCThreadMutex_Lock(pstParentTask->hMutex);
	ERRIFGOTO(result, _EXIT);

	if(pstParentTask->pstLoopInfo->enType == LOOP_TYPE_CONVERGENT && pstParentTask->pstLoopInfo->nDesignatedTaskId == pstCurrentTask->nTaskId)
	{
		if(pstParentTask->pstLoopInfo->bDesignatedTaskState == TRUE)
		{
			//stop loop task iteration
			//set iteration to target iteration
			result = changeTaskState(pstGeneralTask, TASK_STATE_SUSPEND);
			ERRIFGOTO(result, _EXIT);

			result = updateLoopIterationHistory(pstGeneralTask);
			ERRIFGOTO(result, _EXIT_LOCK);

			nCurIteration = pstParentTask->pstLoopInfo->nCurrentIteration;
			nLoopCount = pstParentTask->pstLoopInfo->nLoopCount;
			pstParentTask->pstLoopInfo->nCurrentIteration = nCurIteration - (nCurIteration % nLoopCount) + nLoopCount;

			result = UKCPUTaskCommon_TraverseSubGraphTasks(pstParentTask, setLoopTaskCurrentIteration, pstParentTask);
			ERRIFGOTO(result, _EXIT_LOCK);

			result = UKCPUTaskCommon_TraverseSubGraphTasks(pstParentTask, traverseAndSetEventToStopTask, pstGeneralTask->hManager);
			ERRIFGOTO(result, _EXIT_LOCK);

			pstParentTask->pstLoopInfo->bDesignatedTaskState = FALSE;

			result = clearGeneralTaskData(pstGeneralTask);
			ERRIFGOTO(result, _EXIT_LOCK);
		}
		else
		{
			//run next iteration
			pstParentTask->pstLoopInfo->nCurrentIteration++;
			//pstParentTask->pstLoopInfo->nCurrentIteration = pstGeneralTask->pstTask->nCurIteration + 1;
		}
		result = UKCPUTaskCommon_TraverseSubGraphTasks(pstParentTask, traverseAndSetEventToTemporarySuspendedTask, pstGeneralTask->hManager);
		ERRIFGOTO(result, _EXIT_LOCK);
	}

	result = setLoopTaskCurrentIteration(pstCurrentTask, pstParentTask);
	ERRIFGOTO(result, _EXIT);

	if(pstCurrentTask->nTargetIteration > 0 && pstCurrentTask->nCurIteration >= pstCurrentTask->nTargetIteration)
	{
		enNewState = TASK_STATE_STOP;
		bStateChangeNeeded = TRUE;
	}
	else if(compareIterationtoAllParentLoopTask(pstCurrentTask) == TRUE)
	{
		enNewState = TASK_STATE_SUSPEND;
		bStateChangeNeeded = TRUE;
	}

	if(bStateChangeNeeded == TRUE)
	{
		result = changeTaskState(pstGeneralTask, enNewState);
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
	UCThreadMutex_Unlock(pstParentTask->hMutex);
_EXIT:
	return result;
}

static uem_result setTaskThreadIteration(SGeneralTask *pstGeneralTask, SGeneralTaskThread *pstTaskThread)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstCurrentTask = NULL;
	int nIndex = 0;

	nIndex = pstTaskThread->nTaskFuncId;
	pstCurrentTask = pstGeneralTask->pstTask;

	result = UCThreadMutex_Lock(pstGeneralTask->hMutex);
	ERRIFGOTO(result, _EXIT);

	pstCurrentTask->astThreadContext[nIndex].nCurRunIndex = pstGeneralTask->nCurLoopIndex;

	pstGeneralTask->nCurLoopIndex++;

	result = UCThreadMutex_Unlock(pstGeneralTask->hMutex);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static uem_result handleTaskMainRoutine(SGeneralTask *pstGeneralTask, SGeneralTaskThread *pstTaskThread, FnUemTaskGo fnGo)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstCurrentTask = NULL;
	long long llNextTime = 0;
	int nMaxRunCount = 0;
	int nRunCount = 0;
	ERunCondition enRunCondition;
	uem_bool bFunctionCalled = FALSE;
	int nExecutionCount = 0;
	uem_bool bTargetIterationReached = FALSE;

	pstCurrentTask = pstGeneralTask->pstTask;

	enRunCondition = pstCurrentTask->enRunCondition;

	result = waitRunSignal(pstGeneralTask, pstTaskThread, TRUE, &llNextTime, &nMaxRunCount);
	ERRIFGOTO(result, _EXIT);

	result = setTaskThreadIteration(pstGeneralTask, pstTaskThread);
	ERRIFGOTO(result, _EXIT);

	// if nSeqId is changed, it means this thread is detached or stopped from the CPU task manager.
	// So, end this thread
	while(pstGeneralTask->enTaskState != TASK_STATE_STOP)
	{
		if(bFunctionCalled == TRUE && pstGeneralTask->enTaskState == TASK_STATE_RUNNING)
		{
			if(pstGeneralTask->bIsModeTransition == TRUE)
			{
				result = handleTaskModeTransition(pstTaskThread, pstGeneralTask);
				ERRIFGOTO(result, _EXIT);
			}
			if(pstGeneralTask->bIsSubConvergentLoop == TRUE)
			{
				result = handleLoopTaskIteration(pstTaskThread, pstGeneralTask);
				ERRIFGOTO(result, _EXIT);
			}

			result = setTaskThreadIteration(pstGeneralTask, pstTaskThread);
			ERRIFGOTO(result, _EXIT);
		}
		switch(pstGeneralTask->enTaskState)
		{
		case TASK_STATE_RUNNING:
			switch(enRunCondition)
			{
			case RUN_CONDITION_TIME_DRIVEN:
				result = UKCPUTaskCommon_HandleTimeDrivenTask(pstCurrentTask, fnGo, &llNextTime, &nRunCount, &nMaxRunCount, &bFunctionCalled);
				ERRIFGOTO(result, _EXIT);
				break;
			case RUN_CONDITION_DATA_DRIVEN:
			case RUN_CONDITION_CONTROL_DRIVEN:
				fnGo(pstCurrentTask->nTaskId);
				bFunctionCalled = TRUE;
				break;
			default:
				ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
				break;
			}
			if(bFunctionCalled == TRUE)
			{
				nExecutionCount++;
				result = UKTask_IncreaseRunCount(pstCurrentTask, pstTaskThread->nTaskFuncId, &bTargetIterationReached);
				if(result != ERR_UEM_NOERROR)
					UEM_DEBUG_PRINT("%s (Proc: %d, func_id: %d, current iteration: %d, reached: %d)\n", pstCurrentTask->pszTaskName, pstTaskThread->nProcId, pstTaskThread->nTaskFuncId, pstCurrentTask->nCurIteration, bTargetIterationReached);
				ERRIFGOTO(result, _EXIT);

				if(bTargetIterationReached == TRUE)
				{
					result = changeTaskState(pstGeneralTask, TASK_STATE_STOPPING);
					ERRIFGOTO(result, _EXIT);
				}
			}
			if(enRunCondition == RUN_CONDITION_CONTROL_DRIVEN) // run once for control-driven leaf task
			{
				UEMASSIGNGOTO(result, ERR_UEM_NOERROR, _EXIT);
			}
			break;
		case TASK_STATE_STOPPING:
			// check one more time to handle suspended tasks
			result = UKTask_CheckIterationRunCount(pstCurrentTask, pstTaskThread->nTaskFuncId, &bTargetIterationReached);
			ERRIFGOTO(result, _EXIT);
			// run until iteration count;
			while(bTargetIterationReached == FALSE)
			{
				if(pstGeneralTask->bIsModeTransition == TRUE)
				{
					result = UCThreadMutex_Lock(pstGeneralTask->pstMTMParentTask->hMutex);
					ERRIFGOTO(result, _EXIT);

					result = updateCurrentIteration(pstGeneralTask->pstMTMParentTask->pstMTMInfo, pstGeneralTask->pstTask);

					UCThreadMutex_Unlock(pstGeneralTask->pstMTMParentTask->hMutex); // ignore error to preserve previous result value
					ERRIFGOTO(result, _EXIT);

					if(pstGeneralTask->pstMTMParentTask->pstMTMInfo->nCurrentIteration <= pstGeneralTask->pstTask->nCurIteration)
					{
						break;
					}
				}
				result = setTaskThreadIteration(pstGeneralTask, pstTaskThread);
				ERRIFGOTO(result, _EXIT);
				fnGo(pstCurrentTask->nTaskId);
				//UEM_DEBUG_PRINT("%s (stopping-driven, Proc: %d, func_id: %d, current iteration: %d)\n", pstCurrentTask->pszTaskName, pstTaskThread->nProcId, pstTaskThread->nTaskFuncId, pstCurrentTask->nCurIteration);
				nExecutionCount++;
				result = UKTask_IncreaseRunCount(pstCurrentTask, pstTaskThread->nTaskFuncId, &bTargetIterationReached);
				ERRIFGOTO(result, _EXIT);
			}
			UEMASSIGNGOTO(result, ERR_UEM_NOERROR, _EXIT);
			break;
		case TASK_STATE_STOP:
			// do nothing
			break;
		case TASK_STATE_SUSPEND:
			result = UCThreadMutex_Lock(pstGeneralTask->hMutex);
			ERRIFGOTO(result, _EXIT);

			pstTaskThread->bSuspended = TRUE;

			result = UCThreadMutex_Unlock(pstGeneralTask->hMutex);
			ERRIFGOTO(result, _EXIT);

			result = waitRunSignal(pstGeneralTask, pstTaskThread, FALSE, &llNextTime, &nMaxRunCount);
			ERRIFGOTO(result, _EXIT);
			break;
		default:
			ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_CONTROL, _EXIT);
			break;
		}
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	UEM_DEBUG_PRINT("pstCurrentTask out : %s (count: %d)\n", pstCurrentTask->pszTaskName, nExecutionCount);
//	{
//		int nLoop = 0;
//		for(nLoop = 0; nLoop < g_nChannelNum; nLoop++)
//		{
//			UEM_DEBUG_PRINT("g_astChannels[%d]: size: %d, dataLen: %d\n", nLoop, g_astChannels[nLoop].nBufSize, g_astChannels[nLoop].nDataLen);
//		}
//	}
	return result;
}


static void *taskThreadRoutine(void *pData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SGeneralTaskThreadData *pstThreadData = NULL;
	SGeneralTaskThread *pstTaskThread = NULL;
	SGeneralTask *pstGeneralTask = NULL;
	STask *pstCurrentTask = NULL;
	int nIndex = 0;
	uem_bool bIsCPU = FALSE;
	SGenericMapProcessor *pstProcessorAPI = NULL;

	pstThreadData = (struct _SGeneralTaskThreadData *) pData;

	pstTaskThread = pstThreadData->pstTaskThread;
	pstGeneralTask = pstThreadData->pstGeneralTask;
	pstCurrentTask = pstGeneralTask->pstTask;
	nIndex = pstTaskThread->nTaskFuncId;

	pstProcessorAPI = pstGeneralTask->pstMapProcessorAPI;

	result = UKProcessor_IsCPUByProcessorId(pstGeneralTask->nProcessorId, &bIsCPU);
	ERRIFGOTO(result, _EXIT);

	if(bIsCPU == FALSE)
	{
		result = pstProcessorAPI->fnMapProcessor(pstTaskThread->hThread, pstGeneralTask->nProcessorId, pstTaskThread->nProcId);
		ERRIFGOTO(result, _EXIT);
	}

	pstCurrentTask->astTaskThreadFunctions[nIndex].fnInit(pstCurrentTask->nTaskId);

	result = UKChannel_FillInitialDataBySourceTaskId(pstCurrentTask->nTaskId);
	ERRIFGOTO(result, _EXIT);

	result = handleTaskMainRoutine(pstGeneralTask, pstTaskThread, pstCurrentTask->astTaskThreadFunctions[nIndex].fnGo);
	ERRIFGOTO(result, _EXIT);

_EXIT:
	pstCurrentTask->astTaskThreadFunctions[nIndex].fnWrapup();
	if(pstTaskThread != NULL)
	{
		UCThreadMutex_Lock(pstGeneralTask->hMutex);
		pstTaskThread->bIsThreadFinished = TRUE;
		UCThreadMutex_Unlock(pstGeneralTask->hMutex);
	}

	SAFEMEMFREE(pstThreadData);
	return NULL;
}

static uem_result createGeneralTaskThread(IN int nOffset, IN void *pData, IN void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SGeneralTaskThread *pstTaskThread = NULL;
	SGeneralTask *pstGeneralTask = NULL;
	struct _SGeneralTaskCreateData *pstCreateData = NULL;
	struct _SGeneralTaskThreadData *pstTaskThreadData = NULL;
	uem_bool bIsCPU = FALSE;
	// int nModeId = INVALID_MODE_ID;

	pstTaskThread = (SGeneralTaskThread *) pData;
	pstCreateData = (struct _SGeneralTaskCreateData *) pUserData;
	pstGeneralTask = pstCreateData->pstGeneralTask;

	result = UCThreadMutex_Lock(pstGeneralTask->hMutex);
	ERRIFGOTO(result, _EXIT);

	result = UCThreadEvent_Create(&(pstTaskThread->hEvent));
	if(result == ERR_UEM_NOERROR)
	{
		pstTaskThread->bIsThreadFinished = FALSE;
		pstTaskThread->bSuspended = TRUE;
		result = UCThreadMutex_Unlock(pstGeneralTask->hMutex);
	}
	else
	{
		UCThreadMutex_Unlock(pstGeneralTask->hMutex);
	}
	ERRIFGOTO(result, _EXIT);

	pstTaskThreadData = UCAlloc_malloc(sizeof(struct _SGeneralTaskThreadData));
	ERRMEMGOTO(pstTaskThreadData, result, _EXIT);

	pstTaskThreadData->pstGeneralTask = pstCreateData->pstGeneralTask;
	pstTaskThreadData->pstTaskThread = pstTaskThread;

	result = UCThread_Create(taskThreadRoutine, pstTaskThreadData, &(pstTaskThread->hThread));
	ERRIFGOTO(result, _EXIT);

	result = UKProcessor_IsCPUByProcessorId(pstGeneralTask->nProcessorId, &bIsCPU);
	ERRIFGOTO(result, _EXIT);

	if(bIsCPU == TRUE)
	{
		result = pstGeneralTask->pstMapProcessorAPI->fnMapProcessor(pstTaskThread->hThread, pstGeneralTask->nProcessorId, pstTaskThread->nProcId);
		ERRIFGOTO(result, _EXIT);
	}

	pstTaskThreadData = NULL;

    result = ERR_UEM_NOERROR;
_EXIT:
	if(result != ERR_UEM_NOERROR && pstTaskThread->hThread != NULL)
	{
		UCThread_Destroy(&(pstTaskThread->hThread), FALSE, THREAD_DESTROY_TIMEOUT);
		pstTaskThread->bIsThreadFinished = TRUE;
	}
	SAFEMEMFREE(pstTaskThreadData);
    return result;
}


static uem_result traverseAndCheckStoppingThread(IN int nOffset, IN void *pData, IN void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SGeneralTaskThread *pstTaskThread = NULL;
	struct _SGeneralTaskStopCheck *pstStopCheck = NULL;
	SCPUGeneralTaskManager *pstManager = NULL;
	HThread hThread = NULL;

	pstTaskThread = (SGeneralTaskThread *) pData;
	pstStopCheck = (struct _SGeneralTaskStopCheck *) pUserData;
	pstManager = pstStopCheck->pstManager;

	if(pstTaskThread->hThread != NULL && pstTaskThread->bIsThreadFinished == TRUE)
	{
		hThread = pstTaskThread->hThread;
		pstTaskThread->hThread = NULL;
		result = UCThreadMutex_Unlock(pstManager->hMutex);
		ERRIFGOTO(result, _EXIT);

		result = UCThread_Destroy(&hThread, FALSE, THREAD_DESTROY_TIMEOUT);
		UCThreadMutex_Lock(pstManager->hMutex);
		ERRIFGOTO(result, _EXIT);
	}

	if(pstTaskThread->bIsThreadFinished != TRUE)
	{
		pstStopCheck->bAllStop = FALSE;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	if(result != ERR_UEM_NOERROR && pstStopCheck != NULL)
	{
		UEM_DEBUG_PRINT("Error is happened during checking stopping task.\n");
		pstStopCheck->bAllStop = FALSE;
	}
	return result;
}


static uem_result checkAndStopStoppingThread(SCPUGeneralTaskManager *pstTaskManager, SGeneralTask *pstGeneralTask)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SGeneralTaskStopCheck stStopCheck;

	stStopCheck.bAllStop = TRUE;
	stStopCheck.pstGeneralTask = pstGeneralTask;
	stStopCheck.pstManager = pstTaskManager;

	result = UCDynamicLinkedList_Traverse(pstGeneralTask->hThreadList, traverseAndCheckStoppingThread, &stStopCheck);
	ERRIFGOTO(result, _EXIT);

	if(stStopCheck.bAllStop == TRUE)
	{
		result = UCThreadMutex_Lock(pstGeneralTask->hMutex);
		ERRIFGOTO(result, _EXIT);

		pstGeneralTask->enTaskState = TASK_STATE_STOP;
		pstGeneralTask->bCreated = FALSE;

		result = UCThreadMutex_Unlock(pstGeneralTask->hMutex);
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKCPUGeneralTaskManager_CreateThread(HCPUGeneralTaskManager hManager, STask *pstTargetTask)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUGeneralTaskManager *pstTaskManager = NULL;
	SGeneralTask *pstGeneralTask = NULL;
	struct _SGeneralTaskCreateData stCreateData;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(pstTargetTask, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	if (IS_VALID_HANDLE(hManager, ID_UEM_CPU_GENERAL_TASK_MANAGER) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}
#endif
	pstTaskManager = hManager;

	result = UCThreadMutex_Lock(pstTaskManager->hMutex);
	ERRIFGOTO(result, _EXIT);

	result = findMatchingGeneralTask(pstTaskManager, pstTargetTask->nTaskId, &pstGeneralTask);
	ERRIFGOTO(result, _EXIT_LOCK);

	if(pstGeneralTask->enTaskState == TASK_STATE_STOPPING || pstGeneralTask->pstTask->enRunCondition == RUN_CONDITION_CONTROL_DRIVEN)
	{
		result = checkAndStopStoppingThread(pstTaskManager, pstGeneralTask);
		ERRIFGOTO(result, _EXIT_LOCK);
	}

	stCreateData.pstGeneralTask = pstGeneralTask;

	result = clearGeneralTaskData(pstGeneralTask);
	ERRIFGOTO(result, _EXIT);

	if(pstGeneralTask->bCreated == FALSE)
	{
		result = UCDynamicLinkedList_Traverse(pstGeneralTask->hThreadList, createGeneralTaskThread, &stCreateData);
		ERRIFGOTO(result, _EXIT_LOCK);

		pstGeneralTask->bCreated = TRUE;
	}

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
	UCThreadMutex_Unlock(pstTaskManager->hMutex);
_EXIT:
	return result;
}


uem_result UKCPUGeneralTaskManager_ChangeState(HCPUGeneralTaskManager hManager, STask *pstTargetTask, ECPUTaskState enTaskState)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUGeneralTaskManager *pstTaskManager = NULL;
	SGeneralTask *pstGeneralTask = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(pstTargetTask, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	if (IS_VALID_HANDLE(hManager, ID_UEM_CPU_GENERAL_TASK_MANAGER) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}
#endif
	pstTaskManager = hManager;

	result = UCThreadMutex_Lock(pstTaskManager->hMutex);
	ERRIFGOTO(result, _EXIT);

	result = findMatchingGeneralTask(pstTaskManager, pstTargetTask->nTaskId, &pstGeneralTask);
	ERRIFGOTO(result, _EXIT_LOCK);

	result = changeTaskState(pstGeneralTask, enTaskState);
	ERRIFGOTO(result, _EXIT_LOCK);

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
	UCThreadMutex_Unlock(pstTaskManager->hMutex);
_EXIT:
	return result;
}


static uem_result traverseAndSetEventToTaskThread(IN int nOffset, IN void *pData, IN void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SGeneralTaskThread *pstTaskThread = NULL;

	pstTaskThread = (SGeneralTaskThread *) pData;

	if(pstTaskThread->bSuspended == TRUE)
	{
		result = UCThreadEvent_SetEvent(pstTaskThread->hEvent);
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result traverseAndWaitActivation(IN int nOffset, IN void *pData, IN void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SGeneralTaskThread *pstTaskThread = NULL;
	SCPUGeneralTaskManager *pstTaskManager = NULL;
	struct _SWaitTaskTraverse *pstUserData = NULL;
	long long llCurTime;
	long long llNextTime;

	pstTaskThread = (SGeneralTaskThread *) pData;
	pstUserData = (struct _SWaitTaskTraverse *) pUserData;
	pstTaskManager = (SCPUGeneralTaskManager *) pstUserData->pstManager;

	result = UCTime_GetCurTickInMilliSeconds(&llCurTime);
	ERRIFGOTO(result, _EXIT);

	llNextTime = llCurTime + pstUserData->llLeftTime + 1; // extra 1 ms for minimum time check

	result = UCThreadMutex_Unlock(pstTaskManager->hMutex);
	ERRIFGOTO(result, _EXIT);

	while(pstTaskThread->bSuspended == TRUE && llCurTime < llNextTime)
	{
		UCThread_Yield();
		UCTime_GetCurTickInMilliSeconds(&llCurTime);
	}

	result = UCThreadMutex_Lock(pstTaskManager->hMutex);
	ERRIFGOTO(result, _EXIT);

	if(pstTaskThread->bSuspended == FALSE)
	{
		pstUserData->llLeftTime = llNextTime - llCurTime;
	}
	else
	{
		ERRASSIGNGOTO(result, ERR_UEM_TIME_EXPIRED, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKCPUGeneralTaskManager_ActivateThread(HCPUGeneralTaskManager hManager, STask *pstTargetTask)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUGeneralTaskManager *pstTaskManager = NULL;
	SGeneralTask *pstGeneralTask = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(pstTargetTask, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	if (IS_VALID_HANDLE(hManager, ID_UEM_CPU_GENERAL_TASK_MANAGER) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}
#endif
	pstTaskManager = hManager;

	result = UCThreadMutex_Lock(pstTaskManager->hMutex);
	ERRIFGOTO(result, _EXIT);

	result = findMatchingGeneralTask(pstTaskManager, pstTargetTask->nTaskId, &pstGeneralTask);
	ERRIFGOTO(result, _EXIT_LOCK);

	result = UCDynamicLinkedList_Traverse(pstGeneralTask->hThreadList, traverseAndSetEventToTaskThread, NULL);
	ERRIFGOTO(result, _EXIT_LOCK);

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
	UCThreadMutex_Unlock(pstTaskManager->hMutex);
_EXIT:
	return result;
}


uem_result UKCPUGeneralTaskManager_WaitTaskActivated(HCPUGeneralTaskManager hManager, STask *pstTargetTask, int nTimeoutInMilliSec)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUGeneralTaskManager *pstTaskManager = NULL;
	SGeneralTask *pstGeneralTask = NULL;
	struct _SWaitTaskTraverse stUserData;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(pstTargetTask, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	if (IS_VALID_HANDLE(hManager, ID_UEM_CPU_GENERAL_TASK_MANAGER) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}
#endif
	pstTaskManager = hManager;

	result = UCThreadMutex_Lock(pstTaskManager->hMutex);
	ERRIFGOTO(result, _EXIT);

	result = findMatchingGeneralTask(pstTaskManager, pstTargetTask->nTaskId, &pstGeneralTask);
	ERRIFGOTO(result, _EXIT_LOCK);

	stUserData.llLeftTime = nTimeoutInMilliSec;
	stUserData.pstManager = pstTaskManager;

	result = UCDynamicLinkedList_Traverse(pstGeneralTask->hThreadList, traverseAndWaitActivation, &stUserData);
	ERRIFGOTO(result, _EXIT_LOCK);

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
	UCThreadMutex_Unlock(pstTaskManager->hMutex);
_EXIT:
	return result;
}


uem_result UKCPUGeneralTaskManager_GetTaskState(HCPUGeneralTaskManager hManager, STask *pstTargetTask, ECPUTaskState *penTaskState)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUGeneralTaskManager *pstTaskManager = NULL;
	SGeneralTask *pstGeneralTask = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(pstTargetTask, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	if (IS_VALID_HANDLE(hManager, ID_UEM_CPU_GENERAL_TASK_MANAGER) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}
#endif
	pstTaskManager = hManager;

	result = UCThreadMutex_Lock(pstTaskManager->hMutex);
	ERRIFGOTO(result, _EXIT);

	result = findMatchingGeneralTask(pstTaskManager, pstTargetTask->nTaskId, &pstGeneralTask);
	ERRIFGOTO(result, _EXIT_LOCK);

	if(pstGeneralTask->enTaskState == TASK_STATE_STOPPING)
	{
		result = checkAndStopStoppingThread(pstTaskManager, pstGeneralTask);
		ERRIFGOTO(result, _EXIT_LOCK);
	}

	*penTaskState = pstGeneralTask->enTaskState;

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
	UCThreadMutex_Unlock(pstTaskManager->hMutex);
_EXIT:
	return result;
}


static uem_result traverseAndDestroyThread(IN int nOffset, IN void *pData, IN void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SGeneralTaskThread *pstTaskThread = (SGeneralTaskThread *) pData;
	struct _STaskThreadDestroyTraverse *pstDestroyData = NULL;
	SGeneralTask *pstGeneralTask = NULL;
	SCPUGeneralTaskManager *pstTaskManager = NULL;
	HThread hThread = NULL;

	pstDestroyData = (struct _STaskThreadDestroyTraverse *) pUserData;
	pstTaskManager = pstDestroyData->pstManager;
	pstGeneralTask = pstDestroyData->pstGeneralTask;

	if(pstTaskThread->hThread != NULL)
	{
		hThread = pstTaskThread->hThread;
		pstTaskThread->hThread = NULL;
		result = UCThreadMutex_Unlock(pstTaskManager->hMutex);
		ERRIFGOTO(result, _EXIT);

		result = UCThread_Destroy(&hThread, FALSE, THREAD_DESTROY_TIMEOUT);
		UCThreadMutex_Lock(pstTaskManager->hMutex);
		ERRIFGOTO(result, _EXIT);

		result = UCThreadEvent_Destroy(&(pstTaskThread->hEvent));
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	if(result != ERR_UEM_NOERROR)
	{
		if(pstGeneralTask->pstTask != NULL)
		{
			UEM_DEBUG_PRINT("Failed to destroy general task of [%s] (%d)\n", pstGeneralTask->pstTask->pszTaskName, pstTaskThread->bIsThreadFinished);
		}
		else
		{
			UEM_DEBUG_PRINT("Failed to destroy general task of whole task graph (%d)\n", pstTaskThread->bIsThreadFinished);
		}
	}
	return result;
}


static uem_result destroyGeneralTaskThread(SGeneralTask *pstGeneralTask, SCPUGeneralTaskManager *pstTaskManager)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _STaskThreadDestroyTraverse stDestroyData;

	pstGeneralTask->enTaskState = TASK_STATE_STOP;

	if(pstGeneralTask->bCreated == TRUE)
	{
		result = UKChannel_SetExitByTaskId(pstGeneralTask->pstTask->nTaskId);
		ERRIFGOTO(result, _EXIT);

		result = UCDynamicLinkedList_Traverse(pstGeneralTask->hThreadList, traverseAndSetEventToTaskThread, NULL);
		ERRIFGOTO(result, _EXIT);

		stDestroyData.pstGeneralTask = pstGeneralTask;
		stDestroyData.pstManager = pstTaskManager;

		result = UCDynamicLinkedList_Traverse(pstGeneralTask->hThreadList, traverseAndDestroyThread, &stDestroyData);
		ERRIFGOTO(result, _EXIT);

		result = UKChannel_ClearExitByTaskId(pstGeneralTask->pstTask->nTaskId);
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}



uem_result UKCPUGeneralTaskManager_DestroyThread(HCPUGeneralTaskManager hManager, STask *pstTargetTask)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUGeneralTaskManager *pstTaskManager = NULL;
	SGeneralTask *pstGeneralTask = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(pstTargetTask, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	if (IS_VALID_HANDLE(hManager, ID_UEM_CPU_GENERAL_TASK_MANAGER) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}
#endif
	pstTaskManager = hManager;

	result = UCThreadMutex_Lock(pstTaskManager->hMutex);
	ERRIFGOTO(result, _EXIT);

	result = findMatchingGeneralTask(pstTaskManager, pstTargetTask->nTaskId, &pstGeneralTask);
	if(result == ERR_UEM_NOT_FOUND)
	{
		UEM_DEBUG_PRINT("cannot find matching task: %s\n", pstTargetTask->pszTaskName);
	}
	ERRIFGOTO(result, _EXIT_LOCK);

	result = destroyGeneralTaskThread(pstGeneralTask, pstTaskManager);
	ERRIFGOTO(result, _EXIT_LOCK);

	pstGeneralTask->bCreated = FALSE;

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
	UCThreadMutex_Unlock(pstTaskManager->hMutex);
_EXIT:
	return result;
}


static uem_result traverseAndDestroyGeneralTask(IN int nOffset, IN void *pData, IN void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SGeneralTask *pstGeneralTask = NULL;
	SCPUGeneralTaskManager *pstTaskManager = NULL;

	pstGeneralTask = (SGeneralTask *) pData;
	pstTaskManager = (SCPUGeneralTaskManager *) pUserData;

	result = destroyGeneralTaskThread(pstGeneralTask, pstTaskManager);
	ERRIFGOTO(result, _EXIT);

	result = destroyGeneralTaskStruct(&pstGeneralTask);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result checkGeneralTaskStillRunning(IN int nOffset, IN void *pData, IN void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SGeneralTask *pstGeneralTask = NULL;
	SCPUGeneralTaskManager *pstTaskManager = NULL;

	pstGeneralTask = (SGeneralTask *) pData;
	pstTaskManager = (SCPUGeneralTaskManager *) pUserData;

	if(pstGeneralTask->enTaskState == TASK_STATE_STOPPING)
	{
		result = checkAndStopStoppingThread(pstTaskManager, pstGeneralTask);
		ERRIFGOTO(result, _EXIT);
	}

	if(pstGeneralTask->enTaskState != TASK_STATE_STOP)
	{
		result = ERR_UEM_FOUND_DATA;
	}
	else
	{
		result = ERR_UEM_NOERROR;
	}

_EXIT:
    return result;
}


static uem_result checkTaskThreadRunning(IN int nOffset, IN void *pData, IN void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SGeneralTaskThread *pstTaskThread = NULL;
	int *pnStartCount = 0;

	pstTaskThread = (SGeneralTaskThread *) pData;
	pnStartCount = (int *) pUserData;

	if(pstTaskThread->bSuspended == FALSE)
	{
		*pnStartCount = *pnStartCount + 1;
	}

    result = ERR_UEM_NOERROR;
_EXIT:
    return result;
}


uem_result UKCPUGeneralTaskManager_CheckTaskStarted(HCPUGeneralTaskManager hManager, STask *pstTargetTask, uem_bool *pbStarted)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUGeneralTaskManager *pstTaskManager = NULL;
	SGeneralTask *pstGeneralTask = NULL;
	int nTotalThreadNum = 0;
	int nStartCount = 0;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(pbStarted, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	IFVARERRASSIGNGOTO(pstTargetTask, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	if (IS_VALID_HANDLE(hManager, ID_UEM_CPU_GENERAL_TASK_MANAGER) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}
#endif

	pstTaskManager = hManager;

	result = UCThreadMutex_Lock(pstTaskManager->hMutex);
	ERRIFGOTO(result, _EXIT);

	result = findMatchingGeneralTask(pstTaskManager, pstTargetTask->nTaskId, &pstGeneralTask);
	ERRIFGOTO(result, _EXIT_LOCK);

	result = UCDynamicLinkedList_Traverse(pstGeneralTask->hThreadList, checkTaskThreadRunning, &nStartCount);
	ERRIFGOTO(result, _EXIT_LOCK);

	result = UCDynamicLinkedList_GetLength(pstGeneralTask->hThreadList, &nTotalThreadNum);
	ERRIFGOTO(result, _EXIT_LOCK);

	if(nStartCount == nTotalThreadNum) // All tasks are started
	{
		*pbStarted = TRUE;
	}
	else
	{
		*pbStarted = FALSE;
	}

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
	UCThreadMutex_Unlock(pstTaskManager->hMutex);
_EXIT:
	return result;
}


uem_result UKCPUGeneralTaskManager_CheckAllTaskStopped(HCPUGeneralTaskManager hManager, uem_bool *pbStopped)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUGeneralTaskManager *pstTaskManager = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(pbStopped, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	if (IS_VALID_HANDLE(hManager, ID_UEM_CPU_GENERAL_TASK_MANAGER) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}
#endif

	pstTaskManager = hManager;

	result = UCThreadMutex_Lock(pstTaskManager->hMutex);
	ERRIFGOTO(result, _EXIT);

	result = UCDynamicLinkedList_Traverse(pstTaskManager->hTaskList, checkGeneralTaskStillRunning, pstTaskManager);
	ERRIFGOTO(result, _EXIT_LOCK);

	if(result == ERR_UEM_FOUND_DATA) // there are some tasks still running
	{
		*pbStopped = FALSE;
	}
	else
	{
		*pbStopped = TRUE;
	}

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
	UCThreadMutex_Unlock(pstTaskManager->hMutex);
_EXIT:
	return result;
}


uem_result UKCPUGeneralTaskManager_Destroy(IN OUT HCPUGeneralTaskManager *phManager)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUGeneralTaskManager *pstTaskManager = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(phManager, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	if (IS_VALID_HANDLE(*phManager, ID_UEM_CPU_GENERAL_TASK_MANAGER) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}
#endif

	pstTaskManager = *phManager;

	result = UCThreadMutex_Lock(pstTaskManager->hMutex);
	ERRIFGOTO(result, _EXIT);

	result = UCDynamicLinkedList_Traverse(pstTaskManager->hTaskList, traverseAndDestroyGeneralTask, pstTaskManager);
	UCThreadMutex_Unlock(pstTaskManager->hMutex); // ignore result
	ERRIFGOTO(result, _EXIT);

	result = UCDynamicLinkedList_Destroy(&(pstTaskManager->hTaskList));
	ERRIFGOTO(result, _EXIT);

	result = UCThreadMutex_Destroy(&(pstTaskManager->hMutex));
	ERRIFGOTO(result, _EXIT);

	SAFEMEMFREE(pstTaskManager);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}
