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
#include <UKModelController.h>

#define THREAD_DESTROY_TIMEOUT (3000)
#define CHECK_MODE_ARGUMENT

typedef struct _SGeneralTaskThread {
	int nProcId;
	int nTaskFuncId;
	HThread hThread; // modified
	uem_bool bIsThreadFinished; // modified
	HThreadEvent hEvent;
	uem_bool bSuspended; // modified
	uem_bool bFunctionCalled; // modified
	uem_bool bRestarted;
} SGeneralTaskThread;

typedef struct _SGeneralTask {
	STask *pstTask;
	HLinkedList hThreadList; // modified
	HThreadMutex hMutex;
	ECPUTaskState enTaskState; // modified
	uem_bool bCreated;
	uem_bool bIsTaskGraphSourceTask;
	int nProcessorId;
	SGenericMapProcessor *pstMapProcessorAPI;
	HCPUGeneralTaskManager hManager;
	int nCurLoopIndex; // modified
	HThreadMutex hTaskGraphLock;
	STaskGraph *pstTaskGraphLockGraph;
	ECPUTaskState enRequestState;
	uem_bool bResumedByControl;
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
	uem_bool bActivate;
};

struct _SGeneralTaskThreadDataDuringStopping {
	SGeneralTaskThread *pstTaskThread;
	SGeneralTask *pstGeneralTask;
	uem_bool bNeedToStop;
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
	SGeneralTask *pstGeneralTask;
};

struct _STaskThreadDestroyTraverse {
	SGeneralTask *pstGeneralTask;
};

struct _STraverseChangeSubgraphState {
	SGeneralTask *pstGeneralTask;
	ECPUTaskState enTaskState;
	ECPUTaskState enNewState;
};

struct _STraverseChangeMappedCore {
	SGeneralTask *pstGeneralTask;
	int nNewLocalId;
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
	pstGeneralTask->pstTask = pstMappedInfo->pstTask;
	pstGeneralTask->nProcessorId = pstMappedInfo->nProcessorId;
	pstGeneralTask->pstMapProcessorAPI = pstMappedInfo->pstMapProcessorAPI;
	pstGeneralTask->nCurLoopIndex = 0;
	pstGeneralTask->enRequestState = TASK_STATE_NONE;
	pstGeneralTask->bResumedByControl = FALSE;

	result = UKModelController_GetTopLevelGraph(pstMappedInfo->pstTask->pstParentGraph, &(pstGeneralTask->pstTaskGraphLockGraph));
	ERRIFGOTO(result, _EXIT);

	result = UKModelController_GetTopLevelLockHandle(pstMappedInfo->pstTask->pstParentGraph, &(pstGeneralTask->hTaskGraphLock));
	ERRIFGOTO(result, _EXIT);

	pstGeneralTask->bIsTaskGraphSourceTask = UKChannel_IsTaskSourceTask(pstGeneralTask->pstTask->nTaskId);

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
	pstGeneralTaskThread->bFunctionCalled = FALSE;
	pstGeneralTaskThread->bRestarted = FALSE;

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
	uem_time tCurTime = 0;

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
			result = UCTime_GetCurTickInMilliSeconds(&tCurTime);
			ERRIFGOTO(result, _EXIT);

			result = UKTime_GetNextTimeByPeriod(tCurTime, pstCurrentTask->nPeriod, pstCurrentTask->enPeriodMetric,
																pllNextTime, pnNextMaxRunCount);
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

static uem_result traverseAndCallChangeThreadState(STaskGraph *pstCurrentTaskGraph, ETaskControllerType enControllerType,
											SModelControllerFunctionSet *pstFunctionSet, void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _STraverseChangeSubgraphState *pstUserData = NULL;
	ECPUTaskState enNewTaskState = TASK_STATE_STOP;

	pstUserData = (struct _STraverseChangeSubgraphState *) pUserData;

	switch(enControllerType)
	{
	case CONTROLLER_TYPE_VOID:
	case CONTROLLER_TYPE_CONTROL_TASK_INCLUDED:
		// skip
		break;
	case CONTROLLER_TYPE_STATIC_MODE_TRANSITION:
	case CONTROLLER_TYPE_STATIC_CONVERGENT_LOOP:
	case CONTROLLER_TYPE_STATIC_DATA_LOOP:
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_CONTROL, _EXIT);
		break;
	case CONTROLLER_TYPE_DYNAMIC_MODE_TRANSITION:
	case CONTROLLER_TYPE_DYNAMIC_CONVERGENT_LOOP:
	case CONTROLLER_TYPE_DYNAMIC_DATA_LOOP:
		if(pstFunctionSet->fnChangeThreadState != NULL)
		{
			// general task does not need thread info for changing the state
			result = pstFunctionSet->fnChangeThreadState(pstCurrentTaskGraph, (void *) pstUserData->pstGeneralTask,
												NULL, pstUserData->enTaskState,
												&enNewTaskState);
			ERRIFGOTO(result, _EXIT);

			pstUserData->enNewState = enNewTaskState;
		}
		break;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result handleControllerChangeSubgraphTaskState(SGeneralTask *pstTask, ECPUTaskState enTargetState)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _STraverseChangeSubgraphState stUserData;

	stUserData.pstGeneralTask = pstTask;
	stUserData.enTaskState = enTargetState;
	stUserData.enNewState = enTargetState;

	result = UKModelController_TraverseAndCallFunctions(pstTask->pstTask->pstParentGraph, NULL, traverseAndCallChangeThreadState, &stUserData);
	ERRIFGOTO(result, _EXIT);

	if(stUserData.enNewState == TASK_STATE_SUSPEND)
	{
		result = UCDynamicLinkedList_Traverse(pstTask->hThreadList, traverseAndSetIsSuspended, NULL);
		ERRIFGOTO(result, _EXIT);
	}

	pstTask->enTaskState = stUserData.enNewState;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result changeTaskStateInLock(SGeneralTask *pstGeneralTask, ECPUTaskState enTaskState)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = UKCPUTaskCommon_CheckTaskState(pstGeneralTask->enTaskState, enTaskState);
	ERRIFGOTO(result, _EXIT);

	if(result != ERR_UEM_NOERROR)
	{
		UEMASSIGNGOTO(result, ERR_UEM_NOERROR, _EXIT);
	}

	result = handleControllerChangeSubgraphTaskState(pstGeneralTask, enTaskState);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static uem_result changeTaskStateWithTaskGraphLock(SGeneralTask *pstGeneralTask, ECPUTaskState enTaskState)
{
	uem_result result = ERR_UEM_UNKNOWN;

	if(pstGeneralTask->hTaskGraphLock != NULL)
	{
		result = UCThreadMutex_Lock(pstGeneralTask->hTaskGraphLock);
		ERRIFGOTO(result, _EXIT);
	}

	result = changeTaskStateInLock(pstGeneralTask, enTaskState);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	if(pstGeneralTask->hTaskGraphLock != NULL)
	{
		UCThreadMutex_Unlock(pstGeneralTask->hTaskGraphLock);
	}
	return result;
}



static uem_result changeTaskState(SGeneralTask *pstGeneralTask, ECPUTaskState enTaskState)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = UCThreadMutex_Lock(pstGeneralTask->hMutex);
	ERRIFGOTO(result, _EXIT);

	result = changeTaskStateWithTaskGraphLock(pstGeneralTask, enTaskState);
	ERRIFGOTO(result, _EXIT_LOCK);

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
	UCThreadMutex_Unlock(pstGeneralTask->hMutex);
_EXIT:
	return result;
}


static uem_result clearGeneralTaskData(SGeneralTask *pstGeneralTask)
{
	uem_result result = ERR_UEM_UNKNOWN;

	pstGeneralTask->nCurLoopIndex = 0;
	pstGeneralTask->enRequestState = TASK_STATE_NONE;

	result = ERR_UEM_NOERROR;

	return result;
}


static uem_result setTaskThreadIteration(SGeneralTask *pstGeneralTask, SGeneralTaskThread *pstTaskThread, uem_bool *pbSuspended)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstCurrentTask = NULL;
	int nIndex = 0;
	int nCurIteration;
	int nTaskIterationNumber;

	nIndex = pstTaskThread->nTaskFuncId;
	pstCurrentTask = pstGeneralTask->pstTask;

	result = UCThreadMutex_Lock(pstCurrentTask->hMutex);
	ERRIFGOTO(result, _EXIT);

	nCurIteration = pstCurrentTask->nCurIteration;

	result = UKTask_GetTaskIteration(pstCurrentTask, &nTaskIterationNumber);
	ERRIFGOTO(result, _EXIT_LOCK);

	if(pstGeneralTask->nCurLoopIndex < nCurIteration * nTaskIterationNumber)
	{
		pstGeneralTask->nCurLoopIndex = nCurIteration * nTaskIterationNumber;
	}

    if(pstCurrentTask->nTargetIteration > 0 && pstGeneralTask->nCurLoopIndex  >= pstCurrentTask->nTargetIteration * nTaskIterationNumber)
    {
        *pbSuspended = TRUE;
    }
    else
    {
        *pbSuspended = FALSE;
        pstCurrentTask->astThreadContext[nIndex].nCurRunIndex = pstGeneralTask->nCurLoopIndex / nTaskIterationNumber;
        pstGeneralTask->nCurLoopIndex++;
    }

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
	UCThreadMutex_Unlock(pstCurrentTask->hMutex);
_EXIT:
	return result;
}

static uem_result activateRequest(STask *pstTask, void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SGeneralTaskThreadData *pstUserData = NULL;

	pstUserData = (struct _SGeneralTaskThreadData *) pUserData;

	if(pstTask->nTaskId != pstUserData->pstGeneralTask->pstTask->nTaskId)
	{
		result = UCThreadMutex_Unlock(pstUserData->pstGeneralTask->hTaskGraphLock);
		ERRIFGOTO(result, _EXIT);

		result = UKCPUGeneralTaskManager_ActivateThread(pstUserData->pstGeneralTask->hManager, pstTask);
		UCThreadMutex_Lock(pstUserData->pstGeneralTask->hTaskGraphLock);
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result traverseAndCallHandleModel(STaskGraph *pstCurrentTaskGraph, ETaskControllerType enControllerType,
											SModelControllerFunctionSet *pstFunctionSet, void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SGeneralTaskThreadData *pstUserData = NULL;

	pstUserData = (struct _SGeneralTaskThreadData *) pUserData;

	switch(enControllerType)
	{
	case CONTROLLER_TYPE_VOID:
	case CONTROLLER_TYPE_CONTROL_TASK_INCLUDED:
		// skip
		break;
	case CONTROLLER_TYPE_STATIC_MODE_TRANSITION:
	case CONTROLLER_TYPE_STATIC_CONVERGENT_LOOP:
	case CONTROLLER_TYPE_STATIC_DATA_LOOP:
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_CONTROL, _EXIT);
		break;
	case CONTROLLER_TYPE_DYNAMIC_MODE_TRANSITION:
	case CONTROLLER_TYPE_DYNAMIC_CONVERGENT_LOOP:
	case CONTROLLER_TYPE_DYNAMIC_DATA_LOOP:
		if(pstFunctionSet->fnHandleModel != NULL)
		{
			result = pstFunctionSet->fnHandleModel(pstCurrentTaskGraph, (void *) pstUserData->pstGeneralTask,
												(void *) pstUserData->pstTaskThread);
			ERRIFGOTO(result, _EXIT);
			if(result == ERR_UEM_FOUND_DATA)
			{
				pstUserData->bActivate = TRUE;
			}
		}
		break;
	}

	if( pstCurrentTaskGraph == pstUserData->pstGeneralTask->pstTaskGraphLockGraph && pstUserData->bActivate == TRUE)
	{
		result = UKCPUTaskCommon_TraverseSubGraphTasks(pstUserData->pstGeneralTask->pstTask->pstParentGraph->pstParentTask, activateRequest, pstUserData);
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result handleTaskGraphController(SGeneralTask *pstTask, SGeneralTaskThread *pstTaskThread)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SGeneralTaskThreadData stUserData;

	if(pstTask->hTaskGraphLock != NULL)
	{
		result = UCThreadMutex_Lock(pstTask->hTaskGraphLock);
		ERRIFGOTO(result, _EXIT);
	}
	
	if(pstTask->enRequestState != TASK_STATE_NONE)
	{
		result = changeTaskStateInLock(pstTask, pstTask->enRequestState);
		ERRIFGOTO(result, _EXIT);
		pstTask->enRequestState = TASK_STATE_NONE;
	}

	stUserData.pstGeneralTask = pstTask;
	stUserData.pstTaskThread = pstTaskThread;
	stUserData.bActivate = FALSE;
	if(pstTask->enTaskState == TASK_STATE_RUNNING || pstTask->enTaskState == TASK_STATE_SUSPEND)
	{
		result = UKModelController_TraverseAndCallFunctions(pstTask->pstTask->pstParentGraph, NULL, traverseAndCallHandleModel, &stUserData);
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	if(pstTask->hTaskGraphLock != NULL)
	{
		UCThreadMutex_Unlock(pstTask->hTaskGraphLock);
	}
	return result;
}


static uem_result traverseAndCallHandleModelDuringStopping(STaskGraph *pstCurrentTaskGraph, ETaskControllerType enControllerType,
											SModelControllerFunctionSet *pstFunctionSet, void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SGeneralTaskThreadDataDuringStopping *pstUserData = NULL;

	pstUserData = (struct _SGeneralTaskThreadDataDuringStopping *) pUserData;

	switch(enControllerType)
	{
	case CONTROLLER_TYPE_VOID:
	case CONTROLLER_TYPE_CONTROL_TASK_INCLUDED:
		// skip
		break;
	case CONTROLLER_TYPE_STATIC_MODE_TRANSITION:
	case CONTROLLER_TYPE_STATIC_CONVERGENT_LOOP:
	case CONTROLLER_TYPE_STATIC_DATA_LOOP:
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_CONTROL, _EXIT);
		break;
	case CONTROLLER_TYPE_DYNAMIC_MODE_TRANSITION:
	case CONTROLLER_TYPE_DYNAMIC_CONVERGENT_LOOP:
	case CONTROLLER_TYPE_DYNAMIC_DATA_LOOP:
		if(pstFunctionSet->fnHandleStopping != NULL)
		{
			result = pstFunctionSet->fnHandleStopping(pstCurrentTaskGraph, (void *) pstUserData->pstGeneralTask,
												(void *) pstUserData->pstTaskThread);
			ERRIFGOTO(result, _EXIT);
			if(result == ERR_UEM_ALREADY_DONE)
			{
				pstUserData->bNeedToStop = TRUE;
			}
		}
		break;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result handleTaskGraphControllerDuringStopping(SGeneralTask *pstTask, SGeneralTaskThread *pstTaskThread)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SGeneralTaskThreadDataDuringStopping stUserData;

	stUserData.pstGeneralTask = pstTask;
	stUserData.pstTaskThread = pstTaskThread;
	stUserData.bNeedToStop = FALSE;

	result = UKModelController_TraverseAndCallFunctions(pstTask->pstTask->pstParentGraph, pstTask->hTaskGraphLock, traverseAndCallHandleModelDuringStopping, &stUserData);
	ERRIFGOTO(result, _EXIT);

	if(stUserData.bNeedToStop == TRUE)
	{
		result = ERR_UEM_ALREADY_DONE;
	}
	else
	{
		result = ERR_UEM_NOERROR;
	}
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
	int nExecutionCount = 0;
	uem_bool bTargetIterationReached = FALSE;
	uem_bool bNeedSuspended = FALSE;

	pstCurrentTask = pstGeneralTask->pstTask;

	enRunCondition = pstCurrentTask->enRunCondition;
	pstTaskThread->bFunctionCalled = FALSE;

	result = waitRunSignal(pstGeneralTask, pstTaskThread, TRUE, &llNextTime, &nMaxRunCount);
	ERRIFGOTO(result, _EXIT);

	result = UCThreadMutex_Lock(pstGeneralTask->hMutex);
	ERRIFGOTO(result, _EXIT);

	if(enRunCondition == RUN_CONDITION_TIME_DRIVEN)
	{
		result = setTaskThreadIteration(pstGeneralTask, pstTaskThread, &bNeedSuspended);
		ERRIFGOTO(result, _EXIT_ERROR_LOCK);
		if(bNeedSuspended == TRUE)
		{
			result = changeTaskStateWithTaskGraphLock(pstGeneralTask, TASK_STATE_SUSPEND);
			ERRIFGOTO(result, _EXIT_ERROR_LOCK);
		}
	}

	// if nSeqId is changed, it means this thread is detached or stopped from the CPU task manager.
	// So, end this thread
	while(pstGeneralTask->enTaskState != TASK_STATE_STOP)
	{
		if(pstGeneralTask->enTaskState == TASK_STATE_RUNNING || pstGeneralTask->enTaskState == TASK_STATE_SUSPEND)
		{
			result = handleTaskGraphController(pstGeneralTask, pstTaskThread);
			ERRIFGOTO(result, _EXIT);

			if(pstGeneralTask->enTaskState == TASK_STATE_RUNNING &&
				(enRunCondition != RUN_CONDITION_TIME_DRIVEN ||
				(enRunCondition == RUN_CONDITION_TIME_DRIVEN && (pstTaskThread->bFunctionCalled == TRUE || pstTaskThread->bRestarted == TRUE))))
			{
				result = setTaskThreadIteration(pstGeneralTask, pstTaskThread, &bNeedSuspended);
				ERRIFGOTO(result, _EXIT_ERROR_LOCK);

				if(bNeedSuspended == TRUE)
				{
					result = changeTaskStateWithTaskGraphLock(pstGeneralTask, TASK_STATE_SUSPEND);
					ERRIFGOTO(result, _EXIT_ERROR_LOCK);
				}
			}
		}

		pstTaskThread->bRestarted = FALSE;

		switch(pstGeneralTask->enTaskState)
		{
		case TASK_STATE_RUNNING:
			switch(enRunCondition)
			{
			case RUN_CONDITION_TIME_DRIVEN:
				result = UCThreadMutex_Unlock(pstGeneralTask->hMutex);
				ERRIFGOTO(result, _EXIT);
				//use bFunctionCall To check TimeDrivenTask run condition.
				result = UKCPUTaskCommon_HandleTimeDrivenTask(pstCurrentTask, fnGo, &llNextTime, &nRunCount, &nMaxRunCount, &(pstTaskThread->bFunctionCalled));
				ERRIFGOTO(result, _EXIT);
				break;
			case RUN_CONDITION_DATA_DRIVEN:
			case RUN_CONDITION_CONTROL_DRIVEN:
				result = UCThreadMutex_Unlock(pstGeneralTask->hMutex);
				ERRIFGOTO(result, _EXIT);
				fnGo(pstCurrentTask->nTaskId);
				pstTaskThread->bFunctionCalled = TRUE;
				break;
			default:
				ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT_ERROR_LOCK);
				break;
			}

			result = UCThreadMutex_Lock(pstGeneralTask->hMutex);
			ERRIFGOTO(result, _EXIT);
			if(pstTaskThread->bFunctionCalled == TRUE)
			{
				pstGeneralTask->bResumedByControl = FALSE;
				nExecutionCount++;

				result = UKTask_IncreaseRunCount(pstCurrentTask, pstTaskThread->nTaskFuncId, &bTargetIterationReached);
				if(result != ERR_UEM_NOERROR)
					UEM_DEBUG_PRINT("%s (Proc: %d, func_id: %d, current iteration: %d, reached: %d)\n", pstCurrentTask->pszTaskName, pstTaskThread->nProcId, pstTaskThread->nTaskFuncId, pstCurrentTask->nCurIteration, bTargetIterationReached);
				ERRIFGOTO(result, _EXIT_ERROR_LOCK);

				if(bTargetIterationReached == TRUE)
				{
					result = changeTaskStateWithTaskGraphLock(pstGeneralTask, TASK_STATE_STOPPING);
					ERRIFGOTO(result, _EXIT_ERROR_LOCK);
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
			ERRIFGOTO(result, _EXIT_ERROR_LOCK);

			result = handleTaskGraphControllerDuringStopping(pstGeneralTask, pstTaskThread);
			ERRIFGOTO(result, _EXIT_ERROR_LOCK);
			// run until iteration count;
			while(bTargetIterationReached == FALSE)
			{
				if(result == ERR_UEM_ALREADY_DONE)
				{
					break;
				}

				// skip bNeedSuspended here
				result = setTaskThreadIteration(pstGeneralTask, pstTaskThread, &bNeedSuspended);
				ERRIFGOTO(result, _EXIT_ERROR_LOCK);

				result = UCThreadMutex_Unlock(pstGeneralTask->hMutex);
				ERRIFGOTO(result, _EXIT);
				fnGo(pstCurrentTask->nTaskId);
				result = UCThreadMutex_Lock(pstGeneralTask->hMutex);
				ERRIFGOTO(result, _EXIT);
				//UEM_DEBUG_PRINT("%s (stopping-driven, Proc: %d, func_id: %d, current iteration: %d)\n", pstCurrentTask->pszTaskName, pstTaskThread->nProcId, pstTaskThread->nTaskFuncId, pstCurrentTask->nCurIteration);
				nExecutionCount++;
				result = UKTask_IncreaseRunCount(pstCurrentTask, pstTaskThread->nTaskFuncId, &bTargetIterationReached);
				ERRIFGOTO(result, _EXIT_ERROR_LOCK);

				result = handleTaskGraphControllerDuringStopping(pstGeneralTask, pstTaskThread);
				ERRIFGOTO(result, _EXIT_ERROR_LOCK);
			}
			UEMASSIGNGOTO(result, ERR_UEM_NOERROR, _EXIT);
			break;
		case TASK_STATE_STOP:
			// do nothing

			break;
		case TASK_STATE_SUSPEND:
			pstTaskThread->bSuspended = TRUE;

			result = UCThreadMutex_Unlock(pstGeneralTask->hMutex);
			ERRIFGOTO(result, _EXIT);

			result = waitRunSignal(pstGeneralTask, pstTaskThread, FALSE, &llNextTime, &nMaxRunCount);
			ERRIFGOTO(result, _EXIT);

			pstTaskThread->bFunctionCalled = FALSE;

			pstTaskThread->bRestarted = TRUE;

			result = UCThreadMutex_Lock(pstGeneralTask->hMutex);
			ERRIFGOTO(result, _EXIT);
			break;
		default:
			ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_CONTROL, _EXIT_ERROR_LOCK);
			break;
		}
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	UCThreadMutex_Unlock(pstGeneralTask->hMutex);
	UEM_DEBUG_PRINT("pstCurrentTask out : %s (count: %d)\n", pstCurrentTask->pszTaskName, nExecutionCount);
//	{
//		int nLoop = 0;
//		for(nLoop = 0; nLoop < g_nChannelNum; nLoop++)
//		{
//			UEM_DEBUG_PRINT("g_astChannels[%d]: size: %d, dataLen: %d\n", nLoop, g_astChannels[nLoop].nBufSize, g_astChannels[nLoop].nDataLen);
//		}
//	}
	return result;
_EXIT_ERROR_LOCK:
	UEM_DEBUG_PRINT("pstCurrentTask error out : %s (count: %d)\n", pstCurrentTask->pszTaskName, nExecutionCount);
	UCThreadMutex_Unlock(pstGeneralTask->hMutex);

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

	result = UCThreadEvent_Create(&(pstTaskThread->hEvent));
	ERRIFGOTO(result, _EXIT);

	pstTaskThread->bIsThreadFinished = FALSE;
	pstTaskThread->bSuspended = TRUE;

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
	if(result == ERR_UEM_NOERROR)
	{
		if(pstGeneralTask->enTaskState == TASK_STATE_STOPPING || pstGeneralTask->pstTask->enRunCondition == RUN_CONDITION_CONTROL_DRIVEN)
		{
			result = checkAndStopStoppingThread(pstTaskManager, pstGeneralTask);
		}
	}
	UCThreadMutex_Unlock(pstTaskManager->hMutex);
	ERRIFGOTO(result, _EXIT);

	stCreateData.pstGeneralTask = pstGeneralTask;

	result = UCThreadMutex_Lock(pstGeneralTask->hMutex);
	ERRIFGOTO(result, _EXIT);

	result = clearGeneralTaskData(pstGeneralTask);
	ERRIFGOTO(result, _EXIT_LOCK);

	if(pstGeneralTask->bCreated == FALSE)
	{
		result = UCDynamicLinkedList_Traverse(pstGeneralTask->hThreadList, createGeneralTaskThread, &stCreateData);
		ERRIFGOTO(result, _EXIT_LOCK);

		pstGeneralTask->bCreated = TRUE;
	}
	else
	{
		pstGeneralTask->bResumedByControl = TRUE;
	}

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
	UCThreadMutex_Unlock(pstGeneralTask->hMutex);
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
	UCThreadMutex_Unlock(pstTaskManager->hMutex);
	ERRIFGOTO(result, _EXIT);

	result = changeTaskState(pstGeneralTask, enTaskState);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKCPUGeneralTaskManager_RequestTaskState(HCPUGeneralTaskManager hManager, STask *pstTargetTask, ECPUTaskState enTaskState)
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
	UCThreadMutex_Unlock(pstTaskManager->hMutex);
	ERRIFGOTO(result, _EXIT);

	pstGeneralTask->enRequestState = enTaskState;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}



static uem_result traverseAndSetEventToTaskThread(IN int nOffset, IN void *pData, IN void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SGeneralTaskThread *pstTaskThread = NULL;

	pstTaskThread = (SGeneralTaskThread *) pData;

	if(pstTaskThread->bSuspended == TRUE && pstTaskThread->hEvent != NULL)
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
	SGeneralTask *pstGeneralTask = NULL;
	struct _SWaitTaskTraverse *pstUserData = NULL;
	long long llCurTime;
	long long llNextTime;

	pstTaskThread = (SGeneralTaskThread *) pData;
	pstUserData = (struct _SWaitTaskTraverse *) pUserData;
	pstGeneralTask = (SGeneralTask *) pstUserData->pstGeneralTask;

	result = UCTime_GetCurTickInMilliSeconds(&llCurTime);
	ERRIFGOTO(result, _EXIT);

	llNextTime = llCurTime + pstUserData->llLeftTime + 1; // extra 1 ms for minimum time check

	result = UCThreadMutex_Unlock(pstGeneralTask->hMutex);
	ERRIFGOTO(result, _EXIT);

	while(pstTaskThread->bSuspended == TRUE && llCurTime < llNextTime)
	{
		UCThread_Yield();
		UCTime_GetCurTickInMilliSeconds(&llCurTime);
	}

	result = UCThreadMutex_Lock(pstGeneralTask->hMutex);
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
	UCThreadMutex_Unlock(pstTaskManager->hMutex);
	ERRIFGOTO(result, _EXIT);

	result = UCThreadMutex_Lock(pstGeneralTask->hMutex);
	ERRIFGOTO(result, _EXIT);

	result = UCDynamicLinkedList_Traverse(pstGeneralTask->hThreadList, traverseAndSetEventToTaskThread, NULL);
	ERRIFGOTO(result, _EXIT_LOCK);

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
	UCThreadMutex_Unlock(pstGeneralTask->hMutex);
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
	UCThreadMutex_Unlock(pstTaskManager->hMutex);
	ERRIFGOTO(result, _EXIT);

	stUserData.llLeftTime = nTimeoutInMilliSec;
	stUserData.pstGeneralTask = pstGeneralTask;

	result = UCThreadMutex_Lock(pstGeneralTask->hMutex);
	ERRIFGOTO(result, _EXIT);

	result = UCDynamicLinkedList_Traverse(pstGeneralTask->hThreadList, traverseAndWaitActivation, &stUserData);
	ERRIFGOTO(result, _EXIT_LOCK);

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
	UCThreadMutex_Unlock(pstGeneralTask->hMutex);
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
	if(result == ERR_UEM_NOERROR)
	{
		if(pstGeneralTask->enTaskState == TASK_STATE_STOPPING)
		{
			result = checkAndStopStoppingThread(pstTaskManager, pstGeneralTask);
		}
	}
	UCThreadMutex_Unlock(pstTaskManager->hMutex);
	ERRIFGOTO(result, _EXIT);

	*penTaskState = pstGeneralTask->enTaskState;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result traverseAndDestroyThread(IN int nOffset, IN void *pData, IN void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SGeneralTaskThread *pstTaskThread = (SGeneralTaskThread *) pData;
	struct _STaskThreadDestroyTraverse *pstDestroyData = NULL;
	SGeneralTask *pstGeneralTask = NULL;
	HThread hThread = NULL;

	pstDestroyData = (struct _STaskThreadDestroyTraverse *) pUserData;
	pstGeneralTask = pstDestroyData->pstGeneralTask;

	if(pstTaskThread->hThread != NULL)
	{
		hThread = pstTaskThread->hThread;
		pstTaskThread->hThread = NULL;
		result = UCThreadMutex_Unlock(pstGeneralTask->hMutex);
		ERRIFGOTO(result, _EXIT);

		result = UCThread_Destroy(&hThread, FALSE, THREAD_DESTROY_TIMEOUT);
		UCThreadMutex_Lock(pstGeneralTask->hMutex);
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


static uem_result destroyGeneralTaskThread(SGeneralTask *pstGeneralTask)
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
	UCThreadMutex_Unlock(pstTaskManager->hMutex);
	ERRIFGOTO(result, _EXIT);

	result = UCThreadMutex_Lock(pstGeneralTask->hMutex);
	ERRIFGOTO(result, _EXIT);

	result = destroyGeneralTaskThread(pstGeneralTask);
	ERRIFGOTO(result, _EXIT_LOCK);

	pstGeneralTask->bCreated = FALSE;

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
	UCThreadMutex_Unlock(pstGeneralTask->hMutex);
_EXIT:
	return result;
}


static uem_result traverseAndDestroyGeneralTask(IN int nOffset, IN void *pData, IN void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SGeneralTask *pstGeneralTask = NULL;

	pstGeneralTask = (SGeneralTask *) pData;

	result = destroyGeneralTaskThread(pstGeneralTask);
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

static uem_result changeMappedCore(IN int nOffset, IN void *pData, IN void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SGeneralTaskThread *pstTaskThread = NULL;
	struct _STraverseChangeMappedCore *pstUserData = NULL;
	SGeneralTask *pstGeneralTask = NULL;

	pstTaskThread = (SGeneralTaskThread *) pData;
	pstUserData = (struct _STraverseChangeMappedCore *) pUserData;
	pstGeneralTask = pstUserData->pstGeneralTask;

	result = UCThreadMutex_Lock(pstGeneralTask->hMutex);
	ERRIFGOTO(result, _EXIT_LOCK);

	pstTaskThread->nProcId = pstUserData->nNewLocalId;

	result = pstGeneralTask->pstMapProcessorAPI->fnMapProcessor(pstTaskThread->hThread, pstGeneralTask->nProcessorId, pstTaskThread->nProcId);
	ERRIFGOTO(result, _EXIT_LOCK);

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
	UCThreadMutex_Unlock(pstGeneralTask->hMutex);
_EXIT:
	return result;
}

uem_result UKCPUGeneralTaskManager_ChangeMappedCore(HCPUGeneralTaskManager hManager, STask *pstTargetTask, int nNewLocalId)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUGeneralTaskManager *pstTaskManager = NULL;
	SGeneralTask *pstGeneralTask = NULL;
	struct _STraverseChangeMappedCore stUserData;
	uem_bool bIsCPU = FALSE;
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

	result = UKProcessor_IsCPUByProcessorId(pstGeneralTask->nProcessorId, &bIsCPU);
	ERRIFGOTO(result, _EXIT_LOCK);

	if(bIsCPU == FALSE)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_CONTROL, _EXIT_LOCK);
	}

	stUserData.pstGeneralTask = pstGeneralTask;
	stUserData.nNewLocalId = nNewLocalId;

	result = UCDynamicLinkedList_Traverse(pstGeneralTask->hThreadList, changeMappedCore, &stUserData);
	ERRIFGOTO(result, _EXIT_LOCK);

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

uem_result UKCPUGeneralTaskManagerCB_IsSourceTask(void *pTaskHandle, OUT uem_bool *pbIsSourceTask)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SGeneralTask *pstGeneralTask = NULL;
#if defined(ARGUMENT_CHECK) && defined(CHECK_MODE_ARGUMENT)
	IFVARERRASSIGNGOTO(pbIsSourceTask, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	IFVARERRASSIGNGOTO(pTaskHandle, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	pstGeneralTask = (SGeneralTask *) pTaskHandle;

	*pbIsSourceTask = pstGeneralTask->bIsTaskGraphSourceTask;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKCPUGeneralTaskManagerCB_GetCurrentTaskStructure(void *pTaskHandle, OUT STask **ppstTask)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SGeneralTask *pstGeneralTask = NULL;
#if defined(ARGUMENT_CHECK) && defined(CHECK_MODE_ARGUMENT)
	IFVARERRASSIGNGOTO(ppstTask, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	IFVARERRASSIGNGOTO(pTaskHandle, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	pstGeneralTask = (SGeneralTask *) pTaskHandle;

	*ppstTask = pstGeneralTask->pstTask;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKCPUGeneralTaskManagerCB_GetTaskGraphLock(void *pTaskHandle, OUT HThreadMutex *phMutex)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SGeneralTask *pstGeneralTask = NULL;
#if defined(ARGUMENT_CHECK) && defined(CHECK_MODE_ARGUMENT)
	IFVARERRASSIGNGOTO(phMutex, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	IFVARERRASSIGNGOTO(pTaskHandle, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	pstGeneralTask = (SGeneralTask *) pTaskHandle;

	*phMutex = pstGeneralTask->hTaskGraphLock;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKCPUGeneralTaskManagerCB_ActivateTask(void *pTaskHandle)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SGeneralTask *pstGeneralTask = NULL;
#if defined(ARGUMENT_CHECK) && defined(CHECK_MODE_ARGUMENT)
	IFVARERRASSIGNGOTO(pTaskHandle, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	pstGeneralTask = (SGeneralTask *) pTaskHandle;

	result = UCDynamicLinkedList_Traverse(pstGeneralTask->hThreadList, traverseAndSetEventToTaskThread, NULL);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}



uem_result UKCPUGeneralTaskManagerCB_GetCurrentTaskState(void *pTaskHandle, OUT ECPUTaskState *penState)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SGeneralTask *pstGeneralTask = NULL;
#if defined(ARGUMENT_CHECK) && defined(CHECK_MODE_ARGUMENT)
	IFVARERRASSIGNGOTO(penState, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	IFVARERRASSIGNGOTO(pTaskHandle, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	pstGeneralTask = (SGeneralTask *) pTaskHandle;

	*penState = pstGeneralTask->enTaskState;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKCPUGeneralTaskManagerCB_ChangeTaskState(void *pTaskHandle, ECPUTaskState enState)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SGeneralTask *pstGeneralTask = NULL;
#if defined(ARGUMENT_CHECK) && defined(CHECK_MODE_ARGUMENT)
	IFVARERRASSIGNGOTO(pTaskHandle, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	pstGeneralTask = (SGeneralTask *) pTaskHandle;

	result = changeTaskStateInLock(pstGeneralTask, enState);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

// temporary function
uem_result UKCPUGeneralTaskManagerCB_GetLoopIndex(void *pTaskHandle, OUT int *pnLoopIndex)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SGeneralTask *pstGeneralTask = NULL;
#if defined(ARGUMENT_CHECK) && defined(CHECK_MODE_ARGUMENT)
	IFVARERRASSIGNGOTO(pTaskHandle, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	IFVARERRASSIGNGOTO(pnLoopIndex, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	pstGeneralTask = (SGeneralTask *) pTaskHandle;

	*pnLoopIndex = pstGeneralTask->nCurLoopIndex;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

// temporary function
uem_result UKCPUGeneralTaskManagerCB_SetLoopIndex(void *pTaskHandle, OUT int nLoopIndex)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SGeneralTask *pstGeneralTask = NULL;
#if defined(ARGUMENT_CHECK) && defined(CHECK_MODE_ARGUMENT)
	IFVARERRASSIGNGOTO(pTaskHandle, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	pstGeneralTask = (SGeneralTask *) pTaskHandle;

	pstGeneralTask->nCurLoopIndex = nLoopIndex;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKCPUGeneralTaskManagerCB_GetManagerHandle(void *pTaskHandle, OUT HCPUGeneralTaskManager *phManager)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SGeneralTask *pstGeneralTask = NULL;
#if defined(ARGUMENT_CHECK) && defined(CHECK_MODE_ARGUMENT)
	IFVARERRASSIGNGOTO(pTaskHandle, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	IFVARERRASSIGNGOTO(phManager, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	pstGeneralTask = (SGeneralTask *) pTaskHandle;

	*phManager = pstGeneralTask->hManager;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKCPUGeneralTaskManagerCB_GetFunctionCalled(void *pThreadHandle, OUT uem_bool *pbFunctionCalled)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SGeneralTaskThread *pstTaskThread = NULL;
#if defined(ARGUMENT_CHECK) && defined(CHECK_MODE_ARGUMENT)
	IFVARERRASSIGNGOTO(pThreadHandle, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	IFVARERRASSIGNGOTO(pbFunctionCalled, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	pstTaskThread = (SGeneralTaskThread *) pThreadHandle;

	*pbFunctionCalled = pstTaskThread->bFunctionCalled;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKCPUGeneralTaskManagerCB_GetRestarted(void *pThreadHandle, OUT uem_bool *pbRestarted)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SGeneralTaskThread *pstTaskThread = NULL;
#if defined(ARGUMENT_CHECK) && defined(CHECK_MODE_ARGUMENT)
	IFVARERRASSIGNGOTO(pThreadHandle, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	IFVARERRASSIGNGOTO(pbRestarted, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	pstTaskThread = (SGeneralTaskThread *) pThreadHandle;

	*pbRestarted = pstTaskThread->bRestarted;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKCPUGeneralTaskManagerCB_IsResumedByControl(void *pTaskHandle, OUT uem_bool *pbResumedByControl)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SGeneralTask *pstGeneralTask = NULL;
#if defined(ARGUMENT_CHECK) && defined(CHECK_MODE_ARGUMENT)
	IFVARERRASSIGNGOTO(pTaskHandle, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	IFVARERRASSIGNGOTO(pbResumedByControl, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	pstGeneralTask = (SGeneralTask *) pTaskHandle;

	*pbResumedByControl = pstGeneralTask->bResumedByControl;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


