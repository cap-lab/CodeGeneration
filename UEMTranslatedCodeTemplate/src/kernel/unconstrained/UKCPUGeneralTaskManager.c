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
#include <UCThreadMutex.h>
#include <UCThread.h>
#include <UCDynamicLinkedList.h>
#include <UCTime.h>

#include <uem_data.h>

#include <UKCPUTaskCommon.h>
#include <UKTime.h>
#include <UKChannel.h>
#include <UKModeTransition.h>

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
	HCPUGeneralTaskManager hManager;
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
	pstManager = UC_malloc(sizeof(SCPUGeneralTaskManager));
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


static uem_result createGeneralTaskStruct(HCPUGeneralTaskManager hCPUTaskManager, SMappedGeneralTaskInfo *pstMappedInfo, OUT SGeneralTask **ppstGeneralTask)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SGeneralTask *pstGeneralTask = NULL;

	pstGeneralTask = UC_malloc(sizeof(SGeneralTask));
	ERRMEMGOTO(pstGeneralTask, result, _EXIT);

	pstGeneralTask->hMutex = NULL;
	pstGeneralTask->hThreadList = NULL;
	pstGeneralTask->enTaskState = TASK_STATE_STOP;
	pstGeneralTask->bCreated = FALSE;
	pstGeneralTask->pstMTMParentTask = NULL;
	pstGeneralTask->bIsModeTransition = isModeTransitionTask(pstMappedInfo->pstTask, &(pstGeneralTask->pstMTMParentTask));
	pstGeneralTask->pstTask = pstMappedInfo->pstTask;

	if(pstGeneralTask->bIsModeTransition == TRUE)
	{
		pstGeneralTask->bMTMSourceTask = UKChannel_IsTaskSourceTask(pstGeneralTask->pstTask->nTaskId);
	}
	else
	{
		pstGeneralTask->bMTMSourceTask = FALSE;
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

	pstGeneralTaskThread = UC_malloc(sizeof(SGeneralTaskThread));
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


static uem_result traverseAndSetEventToTemporarySuspendedTask(STask *pstTask, void *pUserData)
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
		//printf("task: %s, available: %d, mode_name: %s\n", pstTask->pszTaskName, bCurrentPortAvailable, pszCurModeName);

		if(pstNewModeData->bModeChanged == TRUE)
		{
			pszOldModeName = pstNewModeData->pstCallerTask->pstMTMParentTask->pstMTMInfo->astModeMap[pstNewModeData->nPrevModeIndex].pszModeName;

			if(UKChannel_IsPortRateAvailableTask(pstTask->nTaskId, pszOldModeName) == FALSE &&
				bCurrentPortAvailable == TRUE)
			{
				//printf("new task: %s, previous_iteration: %d, new_iteration: %d\n", pstTask->pszTaskName, pstTask->nCurIteration, pstNewModeData->nNewStartIteration);

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

		result = UKCPUTaskCommon_TraverseSubGraphTasks(pstMTMTask, traverseAndSetEventToTemporarySuspendedTask, &stNewModeData);
		ERRIFGOTO(result, _EXIT_LOCK);

		pstMTMTask->pstMTMInfo->nCurrentIteration = pstGeneralTask->pstTask->nCurIteration;
	}
	else
	{
		if(pstMTMTask->pstMTMInfo->nCurrentIteration < pstGeneralTask->pstTask->nCurIteration)
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

			if(pstMTMTask->pstMTMInfo->nCurrentIteration < pstGeneralTask->pstTask->nCurIteration)
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

	// if nSeqId is changed, it means this thread is detached or stopped from the CPU task manager.
	// So, end this thread
	while(pstGeneralTask->enTaskState != TASK_STATE_STOP)
	{
		if(bFunctionCalled == TRUE && pstGeneralTask->bIsModeTransition == TRUE && pstGeneralTask->enTaskState == TASK_STATE_RUNNING)
		{
			result = handleTaskModeTransition(pstTaskThread, pstGeneralTask);
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
				result = UKTask_IncreaseRunCount(pstCurrentTask, &bTargetIterationReached);
				ERRIFGOTO(result, _EXIT);

				//printf("%s (Proc: %d, func_id: %d, current iteration: %d, reached: %d)\n", pstCurrentTask->pszTaskName, pstTaskThread->nProcId, pstTaskThread->nTaskFuncId, pstCurrentTask->nCurIteration, bTargetIterationReached);
			}
			if(enRunCondition == RUN_CONDITION_CONTROL_DRIVEN) // run once for control-driven leaf task
			{
				UEMASSIGNGOTO(result, ERR_UEM_NOERROR, _EXIT);
			}
			break;
		case TASK_STATE_STOPPING:
			// check one more time to handle suspended tasks
			result = UKTask_CheckIterationRunCount(pstCurrentTask, &bTargetIterationReached);
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

					if(pstGeneralTask->pstMTMParentTask->pstMTMInfo->nCurrentIteration < pstGeneralTask->pstTask->nCurIteration)
					{
						break;
					}
				}
				fnGo(pstCurrentTask->nTaskId);
				//printf("%s (stopping-driven, Proc: %d, func_id: %d, current iteration: %d)\n", pstCurrentTask->pszTaskName, pstTaskThread->nProcId, pstTaskThread->nTaskFuncId, pstCurrentTask->nCurIteration);
				nExecutionCount++;
				result = UKTask_IncreaseRunCount(pstCurrentTask, &bTargetIterationReached);
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
	printf("pstCurrentTask out : %s (count: %d)\n", pstCurrentTask->pszTaskName, nExecutionCount);
//	{
//		int nLoop = 0;
//		for(nLoop = 0; nLoop < g_nChannelNum; nLoop++)
//		{
//			printf("g_astChannels[%d]: size: %d, dataLen: %d\n", nLoop, g_astChannels[nLoop].nBufSize, g_astChannels[nLoop].nDataLen);
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

	pstThreadData = (struct _SGeneralTaskThreadData *) pData;

	pstTaskThread = pstThreadData->pstTaskThread;
	pstGeneralTask = pstThreadData->pstGeneralTask;
	pstCurrentTask = pstGeneralTask->pstTask;
	nIndex = pstTaskThread->nTaskFuncId;

	pstCurrentTask->astTaskFunctions[nIndex].fnInit(pstCurrentTask->nTaskId);

	result = handleTaskMainRoutine(pstGeneralTask, pstTaskThread, pstCurrentTask->astTaskFunctions[nIndex].fnGo);
	ERRIFGOTO(result, _EXIT);

_EXIT:
	pstCurrentTask->astTaskFunctions[nIndex].fnWrapup();
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

	pstTaskThreadData = UC_malloc(sizeof(struct _SGeneralTaskThreadData));
	ERRMEMGOTO(pstTaskThreadData, result, _EXIT);

	pstTaskThreadData->pstGeneralTask = pstCreateData->pstGeneralTask;
	pstTaskThreadData->pstTaskThread = pstTaskThread;

	result = UCThread_Create(taskThreadRoutine, pstTaskThreadData, &(pstTaskThread->hThread));
	ERRIFGOTO(result, _EXIT);

	pstTaskThreadData = NULL;

	if(pstTaskThread->nProcId != MAPPING_NOT_SPECIFIED)
	{
		result = UCThread_SetMappedCPU(pstTaskThread->hThread, pstTaskThread->nProcId);
		ERRIFGOTO(result, _EXIT);
	}

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
		printf("Error is happened during checking stopping task.\n");
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
			printf("Failed to destroy general task of [%s] (%d)\n", pstGeneralTask->pstTask->pszTaskName, pstTaskThread->bIsThreadFinished);
		}
		else
		{
			printf("Failed to destroy general task of whole task graph (%d)\n", pstTaskThread->bIsThreadFinished);
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
		printf("cannot find matching task: %s\n", pstTargetTask->pszTaskName);
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
