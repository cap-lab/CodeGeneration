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

#include <UCThreadMutex.h>
#include <UCDynamicLinkedList.h>

#include <uem_data.h>

#define MIN_SLEEP_DURATION (10)
#define MAX_SLEEP_DURATION (100)

#define THREAD_DESTROY_TIMEOUT (3000)

typedef struct _SCPUGeneralTaskManager *HCPUGeneralTaskManager;


typedef struct _SGeneralTaskThread {
	int nProcId;
	int nTaskFuncId;
	ECPUTaskState enTaskState; // modified
	HThread hThread; // modified
	uem_bool bIsThreadFinished; // modified
	HThreadEvent hEvent;
	uem_bool bSuspended; // modified
} SGeneralTaskThread;

typedef struct _SGeneralTask {
	STask *pstTask;
	HLinkedList hThreadList; // modified
	HThreadEvent hEvent;
	HThreadMutex hMutex;
	int nFinishedThreadNum; // modified
	ECPUTaskState enTaskState; // modified
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



static uem_result getCachedCompositeTask(SCPUGeneralTaskManager *pstTaskManager, int nTaskId, OUT SGeneralTask **ppstGeneralTask)
{
	SGeneralTask *pstGeneralTask = NULL;
	uem_result result = ERR_UEM_UNKNOWN;

	result = UCThreadMutex_Lock(pstTaskManager->hMutex);
	ERRIFGOTO(result, _EXIT);

	pstGeneralTask = pstTaskManager->pstCachedTask;

	result = UCThreadMutex_Unlock(pstTaskManager->hMutex);
	ERRIFGOTO(result, _EXIT);

	if(pstGeneralTask == NULL)
	{
		*ppstGeneralTask = NULL;
		result = ERR_UEM_NOT_FOUND;
	}
	else if(pstGeneralTask->pstTask == NULL && nTaskId == INVALID_TASK_ID) // whole task graph
	{
		*ppstGeneralTask = pstGeneralTask;
	}
	else if(pstGeneralTask->pstTask->nTaskId == nTaskId)
	{
		*ppstGeneralTask = pstGeneralTask;
	}
	else
	{
		*ppstGeneralTask = NULL;
		result = ERR_UEM_NOT_FOUND;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
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

static uem_result setCachedTask(SCPUGeneralTaskManager *pstTaskManager, SGeneralTask *pstGeneralTask)
{
	uem_result result = ERR_UEM_UNKNOWN;
	result = UCThreadMutex_Lock(pstTaskManager->hMutex);
	ERRIFGOTO(result, _EXIT);

	pstTaskManager->pstCachedTask = pstGeneralTask;

	result = UCThreadMutex_Unlock(pstTaskManager->hMutex);
	ERRIFGOTO(result, _EXIT);


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
	result = getCachedCompositeTask(pstTaskManager, nTaskId, &pstGeneralTask);
	if(result == ERR_UEM_NOT_FOUND)
	{
		stSearchData.nTargetTaskId = nTaskId;
		stSearchData.pstMatchingTask = NULL;

		result = UCDynamicLinkedList_Traverse(pstTaskManager->hTaskList, findTask, &stSearchData);
		ERRIFGOTO(result, _EXIT);

		if(result == ERR_UEM_FOUND_DATA)
		{
			pstGeneralTask = stSearchData.pstMatchingTask;
			setCachedTask(pstTaskManager, pstGeneralTask);
		}
		else
		{
			UEMASSIGNGOTO(result, ERR_UEM_NOT_FOUND, _EXIT);
		}
	}

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

	result = UCThreadEvent_Destroy(&(pstGeneralTaskThread->hEvent));
	ERRIFGOTO(result, _EXIT);

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

	if(pstGeneralTask->hEvent != NULL)
	{
		UCThreadEvent_Destroy(&(pstGeneralTask->hEvent));
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

	pstGeneralTask = UC_malloc(sizeof(SGeneralTask));
	ERRMEMGOTO(pstGeneralTask, result, _EXIT);

	pstGeneralTask->hEvent = NULL;
	pstGeneralTask->hMutex = NULL;
	pstGeneralTask->hThreadList = NULL;
	pstGeneralTask->enTaskState = TASK_STATE_STOP;
	pstGeneralTask->nFinishedThreadNum = 0;
	pstGeneralTask->pstTask = pstMappedInfo->pstTask;

	result = UCThreadEvent_Create(&(pstGeneralTask->hEvent));
	ERRIFGOTO(result, _EXIT);

	result = UCThreadMutex_Create(&(pstGeneralTask->hMutex));
	ERRIFGOTO(result, _EXIT);

	result = UCDynamicLinkedList_Create(&(pstGeneralTask->hThreadList));
	ERRIFGOTO(result, _EXIT);

	//pstGeneralTask->hManager = hCPUTaskManager;

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

	pstGeneralTaskThread->enTaskState = TASK_STATE_STOP;
	pstGeneralTaskThread->nProcId = pstMappedInfo->nProcessorId;
	pstGeneralTaskThread->hThread = NULL;
	pstGeneralTaskThread->bIsThreadFinished = TRUE;
	pstGeneralTaskThread->hEvent = NULL;
	pstGeneralTaskThread->bSuspended = FALSE;
	pstGeneralTaskThread->nTaskFuncId = 0;

	result = UCThreadEvent_Create(&(pstGeneralTaskThread->hEvent));
	ERRIFGOTO(result, _EXIT);

	*ppstGeneralTaskThread = pstGeneralTaskThread;

	result = ERR_UEM_NOERROR;
_EXIT:
	if(result != ERR_UEM_NOERROR && pstGeneralTaskThread != NULL)
	{
		SAFEMEMFREE(pstGeneralTaskThread);
	}
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

	if (IS_VALID_HANDLE(hManager, ID_UEM_CPU_COMPOSITE_TASK_MANAGER) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}
#endif
	pstTaskManager = hManager;

	result = findMatchingGeneralTask(pstTaskManager, pstMappedTask->pstTask->nTaskId, &pstGeneralTask);
	if(result == ERR_UEM_NOT_FOUND)
	{
		result = createCompositeTaskStruct(hManager, pstMappedTask, &pstGeneralTask);
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


static uem_result waitRunSignal(SGeneralTask *pstTask, SGeneralTaskThread *pstTaskThread, uem_bool bStartWait, OUT long long *pllNextTime, OUT int *pnNextMaxRunCount)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstCurrentTask = NULL;
	long long llCurTime = 0;

	pstCurrentTask = pstTask->pstTask;

	result = UCThreadEvent_WaitEvent(pstTaskThread->hEvent);
	ERRIFGOTO(result, _EXIT);

	if(pstTask->enTaskState == TASK_STATE_RUNNING || bStartWait == TRUE)
	{
		result = UCThreadMutex_Lock(pstTask->hMutex);
		ERRIFGOTO(result, _EXIT);

		pstTaskThread->bSuspended = FALSE;

		result = UCThreadMutex_Unlock(pstTask->hMutex);
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


static uem_result handleTimeDrivenTask(STask *pstCurrentTask, FnUemTaskGo fnGo, IN OUT long long *pllNextTime,
										IN OUT int *pnRunCount, IN OUT int *pnNextMaxRunCount)
{
	uem_result result = ERR_UEM_UNKNOWN;
	long long llCurTime = 0;
	long long llNextTime = 0;
	int nMaxRunCount = 0;
	int nRunCount = 0;

	llNextTime = *pllNextTime;
	nRunCount = *pnRunCount;
	nMaxRunCount = *pnNextMaxRunCount;

	result = UCTime_GetCurTickInMilliSeconds(&llCurTime);
	ERRIFGOTO(result, _EXIT);
	if(llCurTime <= llNextTime) // time is not passed
	{
		if(nRunCount < nMaxRunCount) // run count is available
		{
			nRunCount++;
			fnGo(pstCurrentTask->nTaskId);
		}
		else // all run count is used and time is not passed yet
		{
			// if period is long, sleep by time passing
			if(llNextTime - llCurTime > MAX_SLEEP_DURATION)
			{
				UCTime_Sleep(MAX_SLEEP_DURATION);
			}
			else if(llNextTime - llCurTime > MIN_SLEEP_DURATION) // left time is more than SLEEP_DURATION ms
			{
				UCTime_Sleep(MIN_SLEEP_DURATION);
			}
			else
			{
				// otherwise, busy wait
			}
		}
	}
	else // time is passed, reset
	{
		result = UKTime_GetNextTimeByPeriod(llNextTime, pstCurrentTask->nPeriod, pstCurrentTask->enPeriodMetric,
										&llNextTime, &nMaxRunCount);
		ERRIFGOTO(result, _EXIT);
		nRunCount = 0;
	}

	*pllNextTime = llNextTime;
	*pnRunCount = nRunCount;
	*pnNextMaxRunCount = nMaxRunCount;

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
	uem_bool bIsTaskGraphSourceTask = FALSE;

	pstCurrentTask = pstGeneralTask->pstTask;
	enRunCondition = pstCurrentTask->enRunCondition;
	bIsTaskGraphSourceTask = UKChannel_IsTaskSourceTask(pstCurrentTask->nTaskId);

	result = waitRunSignal(pstTaskThread, TRUE, &llNextTime, &nMaxRunCount);
	ERRIFGOTO(result, _EXIT);

	// if nSeqId is changed, it means this thread is detached or stopped from the CPU task manager.
	// So, end this thread
	while(pstTaskThread->enTaskState != TASK_STATE_STOP)
	{
		switch(pstTaskThread->enTaskState)
		{
		case TASK_STATE_RUNNING:
			switch(enRunCondition)
			{
			case RUN_CONDITION_TIME_DRIVEN:
				result = handleTimeDrivenTask(pstCurrentTask, fnGo, &llNextTime, &nRunCount, &nMaxRunCount);
				ERRIFGOTO(result, _EXIT);
				break;
			case RUN_CONDITION_DATA_DRIVEN:
				fnGo(pstCurrentTask->nTaskId);
				break;
			case RUN_CONDITION_CONTROL_DRIVEN: // run once for control-driven leaf task
				fnGo(pstCurrentTask->nTaskId);
				break;
			default:
				ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
				break;
			}
			break;
		case TASK_STATE_STOPPING:
			//fnGo(pstCurrentTask->nTaskId);
			UEMASSIGNGOTO(result, ERR_UEM_NOERROR, _EXIT);
			break;
		case TASK_STATE_STOP:
			// do nothing
			break;
		case TASK_STATE_SUSPEND:
			result = waitRunSignal(pstTaskThread, FALSE, &llNextTime, &nMaxRunCount);
			ERRIFGOTO(result, _EXIT);
			break;
		default:
			ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_CONTROL, _EXIT);
			break;
		}
	}

	result = ERR_UEM_NOERROR;
_EXIT:
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

	result = handleTaskMainRoutine(pstTaskThread, pstCurrentTask->astTaskFunctions[nIndex].fnGo);
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

	pstTaskThread->bIsThreadFinished = FALSE;

	result = UCThreadMutex_Unlock(pstGeneralTask->hMutex);
	ERRIFGOTO(result, _EXIT);

	pstTaskThreadData = UC_malloc(sizeof(struct _SGeneralTaskThreadData));
	ERRMEMGOTO(pstTaskThreadData, result, _EXIT);

	pstTaskThreadData->pstGeneralTask = pstCreateData->pstGeneralTask;
	pstTaskThreadData->pstTaskThread = pstTaskThread;

	result = UCThread_Create(taskThreadRoutine, pstTaskThreadData, &(pstTaskThread->hThread));
	ERRIFGOTO(result, _EXIT);

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
    return result;
}


uem_result UKCPUGeneralTaskManager_CreateThread(HCPUGeneralTaskManager hManager, STask *pstTargetTask)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUGeneralTaskManager *pstTaskManager = NULL;
	SGeneralTask *pstGeneralTask = NULL;
	SGeneralTaskThread *pstGeneralTaskThread = NULL;
	struct _SGeneralTaskCreateData stCreateData = NULL;
	int nTaskId = INVALID_TASK_ID;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(pstTargetTask, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	if (IS_VALID_HANDLE(hManager, ID_UEM_CPU_COMPOSITE_TASK_MANAGER) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}
#endif
	pstTaskManager = hManager;

	result = findMatchingGeneralTask(pstTaskManager, pstTargetTask->nTaskId, &pstGeneralTask);
	ERRIFGOTO(result, _EXIT);

	stCreateData.pstGeneralTask = pstGeneralTask;

	result = UCDynamicLinkedList_Traverse(pstGeneralTask->hThreadList, createGeneralTaskThread, &stCreateData);
	ERRIFGOTO(result, _EXIT);


	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKCPUGeneralTaskManager_ChangeState(HCPUGeneralTaskManager hManager, STask *pstTargetTask, EInternalTaskState enTaskState)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUGeneralTaskManager *pstTaskManager = NULL;
	SGeneralTask *pstGeneralTask = NULL;
	SGeneralTaskThread *pstGeneralTaskThread = NULL;
	int nTaskId = INVALID_TASK_ID;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(pstTargetTask, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	if (IS_VALID_HANDLE(hManager, ID_UEM_CPU_COMPOSITE_TASK_MANAGER) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}
#endif
	pstTaskManager = hManager;

	result = findMatchingGeneralTask(pstTaskManager, pstTargetTask->nTaskId, &pstGeneralTask);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKCPUGeneralTaskManager_ActivateThread(HCPUGeneralTaskManager hManager, STask *pstTargetTask)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUGeneralTaskManager *pstTaskManager = NULL;
	SGeneralTask *pstGeneralTask = NULL;
	SGeneralTaskThread *pstGeneralTaskThread = NULL;
	int nTaskId = INVALID_TASK_ID;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(pstTargetTask, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	if (IS_VALID_HANDLE(hManager, ID_UEM_CPU_COMPOSITE_TASK_MANAGER) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}
#endif
	pstTaskManager = hManager;

	result = findMatchingGeneralTask(pstTaskManager, pstTargetTask->nTaskId, &pstGeneralTask);
	ERRIFGOTO(result, _EXIT);


	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKCPUGeneralTaskManager_GetTaskState(HCPUGeneralTaskManager hManager, STask *pstTargetTask, EInternalTaskState *penTaskState)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUGeneralTaskManager *pstTaskManager = NULL;
	SGeneralTask *pstGeneralTask = NULL;
	SGeneralTaskThread *pstGeneralTaskThread = NULL;
	int nTaskId = INVALID_TASK_ID;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(pstTargetTask, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	if (IS_VALID_HANDLE(hManager, ID_UEM_CPU_COMPOSITE_TASK_MANAGER) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}
#endif
	pstTaskManager = hManager;

	result = findMatchingGeneralTask(pstTaskManager, pstTargetTask->nTaskId, &pstGeneralTask);
	ERRIFGOTO(result, _EXIT);


	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKCPUGeneralTaskManager_DestroyThread(HCPUGeneralTaskManager hManager, STask *pstTargetTask)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUGeneralTaskManager *pstTaskManager = NULL;
	SGeneralTask *pstGeneralTask = NULL;
	SGeneralTaskThread *pstGeneralTaskThread = NULL;
	int nTaskId = INVALID_TASK_ID;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(pstTargetTask, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	if (IS_VALID_HANDLE(hManager, ID_UEM_CPU_COMPOSITE_TASK_MANAGER) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}
#endif
	pstTaskManager = hManager;

	result = findMatchingGeneralTask(pstTaskManager, pstTargetTask->nTaskId, &pstGeneralTask);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}



uem_result UKCPUGeneralTaskManager_Destroy(IN OUT HCPUGeneralTaskManager *phManager)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}
