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

#include <UCBasic.h>
#include <UCTime.h>
#include <UCThread.h>
#include <UCDynamicLinkedList.h>
#include <UCDynamicStack.h>
#include <UCThreadMutex.h>
#include <UCThreadEvent.h>

#include <UKCPUTaskManager.h>
#include <UKTime.h>
#include <UKChannel.h>

#define THREAD_DESTROY_TIMEOUT (5000)

#define MIN_SLEEP_DURATION (10)
#define MAX_SLEEP_DURATION (100)

typedef struct _SCPUCompositeTaskManager *HCPUCompositeTaskManager;

typedef struct _SCompositeTaskThread {
	int nModeId;
	int nThroughputConstraint;
	int nProcId;
	FnUemTaskGo fnCompositeGo;
	ECPUTaskState enTaskState; // modified
	HThread hThread; // modified
	uem_bool bIsThreadFinished; // modified
	HThreadEvent hEvent;
	uem_bool bSuspended; // modified
	HCPUCompositeTaskManager hManager; // handle for accessing higher data structures
} SCompositeTaskThread;


typedef struct _SCompositeTask {
	STask *pstParentTask;
	HLinkedList hThreadList; // modified
	HThreadEvent hEvent;
	HThreadMutex hMutex;
	int nFinishedThreadNum; // modified
	SScheduledTasks *pstScheduledTasks;
	ECPUTaskState enTaskState; // modified
	int nCurrentThroughputConstraint; // modified
} SCompositeTask;


typedef struct _SCPUCompositeTaskManager {
	EUemModuleId enId;
	HLinkedList hTaskList;
	uem_bool bListStatic;
	HThreadMutex hMutex;
	SCompositeTask *pstCachedCompositeTask;
} SCPUCompositeTaskManager;

struct _SCompositeTaskSearchData {
	int nTargetParentTaskId;
	SCompositeTask *pstMatchingCompositeTask;
};


struct _SCompositeTaskCreateData {
	SCompositeTask *pstCompositeTask;
};

struct _SCompositeTaskStateChangeData {
	SCompositeTask *pstCompositeTask;
	ECPUTaskState enTaskState;
};

struct _SCompositeTaskThreadData {
	SCompositeTaskThread *pstTaskThread;
	SCompositeTask *pstCompositeTask;
};


struct _SCompositeTaskStopCheck {
	uem_bool bAllStop;
	SCompositeTask *pstCompositeTask;
};


typedef uem_result (*FnCbHandleGeneralTask)(STask *pstTask);


static uem_result destroyCompositeTaskThreadStruct(IN int nOffset, IN void *pData, IN void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCompositeTaskThread *pstCompositeTaskThread = NULL;

	pstCompositeTaskThread = (SCompositeTaskThread *) pData;

	result = UCThreadEvent_Destroy(&(pstCompositeTaskThread->hEvent));
	ERRIFGOTO(result, _EXIT);

	SAFEMEMFREE(pstCompositeTaskThread);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result destroyCompositeTaskStruct(IN OUT SCompositeTask **ppstCompositeTask)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCompositeTask *pstCompositeTask = NULL;

	pstCompositeTask = *ppstCompositeTask;

	if(pstCompositeTask->hThreadList != NULL)
	{
		// Traverse all thread to be destroyed
		UCDynamicLinkedList_Traverse(pstCompositeTask->hThreadList, destroyCompositeTaskThreadStruct, NULL);

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
	pstCompositeTaskThread->hThread = NULL;
	pstCompositeTaskThread->bIsThreadFinished = TRUE;
	pstCompositeTaskThread->hEvent = NULL;
	pstCompositeTaskThread->bSuspended = FALSE;

	result = UCThreadEvent_Create(&(pstCompositeTaskThread->hEvent));
	ERRIFGOTO(result, _EXIT);

	*ppstCompositeTaskThread = pstCompositeTaskThread;

	result = ERR_UEM_NOERROR;
_EXIT:
	if(result != ERR_UEM_NOERROR && pstCompositeTaskThread != NULL)
	{
		SAFEMEMFREE(pstCompositeTaskThread);
	}
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
static uem_result createCompositeTaskStruct(HCPUCompositeTaskManager hCPUTaskManager, SMappedCompositeTaskInfo *pstMappedInfo, OUT SCompositeTask **ppstCompositeTask)
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
	pstCompositeTask->pstParentTask = pstMappedInfo->pstScheduledTasks->pstParentTask;
	pstCompositeTask->nCurrentThroughputConstraint = pstMappedInfo->pstScheduledTasks->astScheduleList[0].nThroughputConstraint;

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
	pstTaskManager->pstCachedCompositeTask = NULL;

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


static uem_result waitRunSignal(SCompositeTask *pstTask, SCompositeTaskThread *pstTaskThread, uem_bool bStartWait, OUT long long *pllNextTime, OUT int *pnNextMaxRunCount)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstCurrentTask = NULL;
	long long llCurTime = 0;

	pstCurrentTask = pstTask->pstParentTask;

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

static uem_bool isModeTransitionTask(SCompositeTask *pstTask)
{
	STask *pstCurrentTask = NULL;
	uem_bool bIsModeTransition = FALSE;
	int nLen = 0;

	pstCurrentTask = pstTask->pstParentTask;

	if(pstCurrentTask != NULL && pstCurrentTask->pstMTMInfo != NULL)
	{
		nLen = pstCurrentTask->pstMTMInfo->nNumOfModes;

		if(nLen > 1)
		{
			bIsModeTransition = TRUE;
		}
	}

	return bIsModeTransition;
}


static uem_result checkTaskThreadState(ECPUTaskState enOldState, ECPUTaskState enNewState)
{
	uem_result result = ERR_UEM_UNKNOWN;

	if(enNewState == enOldState)
	{
		UEMASSIGNGOTO(result, ERR_UEM_ALREADY_DONE, _EXIT);
	}

	switch(enOldState)
	{
	case TASK_STATE_RUNNING:
		// do nothing
		break;
	case TASK_STATE_SUSPEND:
		if(enNewState == TASK_STATE_STOPPING)
		{
			ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_CONTROL, _EXIT);
		}
		break;
	case TASK_STATE_STOP:
		if(enNewState != TASK_STATE_RUNNING && enNewState != TASK_STATE_STOPPING)
		{
			ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_CONTROL, _EXIT);
		}
		break;
	case TASK_STATE_STOPPING:
		if(enNewState != TASK_STATE_STOP)
		{
			ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_CONTROL, _EXIT);
		}
		break;
	default:
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_CONTROL, _EXIT);
		break;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static int getCurrentTaskModeId(STask *pstTask)
{
	int nCurModeIndex = INVALID_MODE_ID;
	int nModeId = 0;
	if(pstTask != NULL && pstTask->pstMTMInfo != NULL)
	{
		UCThreadMutex_Lock(pstTask->hMutex);
		nCurModeIndex = pstTask->pstMTMInfo->nCurModeIndex;
		UCThreadMutex_Unlock(pstTask->hMutex);
		nModeId = pstTask->pstMTMInfo->astModeMap[nCurModeIndex].nModeId;
	}
	else
	{
		nModeId = 0;
	}

	return nModeId;
}


static uem_result traverseAndChangeTaskState(IN int nOffset, IN void *pData, IN void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SCompositeTaskStateChangeData *pstUserData = NULL;
	int nCurModeId = 0;
	SCompositeTaskThread *pstTaskThread = (SCompositeTaskThread *) pData;
	SCompositeTask *pstTask = NULL;
	pstUserData = (struct _SCompositeTaskStateChangeData *) pUserData;
	pstTask = pstUserData->pstCompositeTask;

	result = checkTaskThreadState(pstTaskThread->enTaskState, pstUserData->enTaskState);
	ERRIFGOTO(result, _EXIT);

	// For MTM task
	nCurModeId = getCurrentTaskModeId(pstTask->pstParentTask);

	result = UCThreadMutex_Lock(pstTask->hMutex);
	ERRIFGOTO(result, _EXIT);

	if(pstUserData->enTaskState == TASK_STATE_STOP)
	{
		if(pstTaskThread->enTaskState == TASK_STATE_RUNNING ||
			pstTaskThread->enTaskState == TASK_STATE_STOPPING ||
			pstTaskThread->enTaskState == TASK_STATE_SUSPEND)
		{
			// release task if task is suspended
			if(pstTaskThread->enTaskState == TASK_STATE_SUSPEND)
			{
				pstTaskThread->enTaskState = TASK_STATE_STOP;

				// exit from WaitEvent
				result = UCThreadEvent_SetEvent(pstTask->hEvent);
				ERRIFGOTO(result, _EXIT);
			}
			else
			{
				pstTaskThread->enTaskState = TASK_STATE_STOP;
			}
		}
		else
		{
			UEMASSIGNGOTO(result, ERR_UEM_ALREADY_DONE, _EXIT);
		}
	}
	else if(pstTask->nCurrentThroughputConstraint != pstTaskThread->nThroughputConstraint)
	{
		pstTaskThread->enTaskState = TASK_STATE_STOP;
	}
	else if(pstTaskThread->nModeId == nCurModeId)
	{
		if(pstTaskThread->enTaskState != TASK_STATE_STOP)
		{
			// state is changed to suspended
			if( pstUserData->enTaskState == TASK_STATE_SUSPEND)
			{
				pstTaskThread->bSuspended = TRUE;
			}

			pstTaskThread->enTaskState = pstUserData->enTaskState;
		}
	}
	else
	{
		// Different modes needs to be suspended when specific mode becomes running
		if(pstUserData->enTaskState == TASK_STATE_RUNNING)
		{
			pstTaskThread->enTaskState = TASK_STATE_SUSPEND;
			pstTaskThread->bSuspended = TRUE;
		}
	}

	result = UCThreadMutex_Unlock(pstTask->hMutex);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result activateTaskThread(SCompositeTaskThread *pstTaskThread)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = UCThreadEvent_SetEvent(pstTaskThread->hEvent);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result traverseAndSetEventToTaskThread(IN int nOffset, IN void *pData, IN void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCompositeTaskThread *pstTaskThread = NULL;

	pstTaskThread = (SCompositeTaskThread *) pData;

	result = activateTaskThread(pstTaskThread);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result activateTask(SCompositeTask *pstTask)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = UCDynamicLinkedList_Traverse(pstTask->hThreadList, traverseAndSetEventToTaskThread, NULL);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result checkAndPopStack(HStack hStack, IN OUT STaskGraph **ppstTaskGraph, IN OUT int *pnIndex)
{
	uem_result result = ERR_UEM_UNKNOWN;

	STaskGraph *pstTaskGraph = NULL;
	int nIndex = 0;
	int nStackNum = 0;
	void *pIndex = NULL;

	pstTaskGraph = *ppstTaskGraph;
	nIndex = *pnIndex;

	result = UCDynamicStack_Length(hStack, &nStackNum);
	ERRIFGOTO(result, _EXIT);

	if(nIndex >= pstTaskGraph->nNumOfTasks && nStackNum > 0)
	{
		result = UCDynamicStack_Pop(hStack, &pIndex);
		ERRIFGOTO(result, _EXIT);

#if SIZEOF_VOID_P == 8
		nIndex = (int) ((long long) pIndex);
#else
		nIndex = (int) pIndex;
#endif

		result = UCDynamicStack_Pop(hStack, (void **) &pstTaskGraph);
		ERRIFGOTO(result, _EXIT);

		*ppstTaskGraph = pstTaskGraph;
		*pnIndex = nIndex;
	}
	else
	{
		// do nothing
	}
	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result callHierarchicalTaskGraphInitWrapupFunctions(STask *pstParentTask, FnCbHandleGeneralTask fnCallback, HStack hStack)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nCurrentIndex = 0;
	int nStackNum = 0;
	STaskGraph *pstCurrentTaskGraph = NULL;
	STaskGraph *pstNextTaskGraph = NULL;
	STask *pstCurTask = NULL;
	int nNumOfTasks = 0;

	nNumOfTasks = pstParentTask->pstSubGraph->nNumOfTasks;
	pstCurrentTaskGraph = pstParentTask->pstSubGraph;

	while(nCurrentIndex < nNumOfTasks || nStackNum > 0)
	{
		if(pstCurrentTaskGraph->astTasks[nCurrentIndex].pstSubGraph != NULL)
		{
			pstNextTaskGraph = pstCurrentTaskGraph->astTasks[nCurrentIndex].pstSubGraph;
			// the current task has subgraph, skip current task index
			nCurrentIndex++;

			if(nCurrentIndex < pstCurrentTaskGraph->nNumOfTasks)
			{
				result = UCDynamicStack_Push(hStack, pstCurrentTaskGraph);
				ERRIFGOTO(result, _EXIT);
#if SIZEOF_VOID_P == 8
				result = UCDynamicStack_Push(hStack, (void *) (long long) nCurrentIndex);
#else
				result = UCDynamicStack_Push(hStack, (void *) nCurrentIndex);
#endif
				ERRIFGOTO(result, _EXIT);
			}

			// reset values
			pstCurrentTaskGraph = pstNextTaskGraph;
			nCurrentIndex = 0;
			nNumOfTasks = pstCurrentTaskGraph->nNumOfTasks;
		}
		else // does not have internal task
		{
			// call current index's proper callback function
			pstCurTask = &(pstCurrentTaskGraph->astTasks[nCurrentIndex]);

			result = fnCallback(pstCurTask);
			ERRIFGOTO(result, _EXIT);

			// proceed index, if all index is proceeded, pop the task graph from stack
			nCurrentIndex++;

			result = checkAndPopStack(hStack, &pstCurrentTaskGraph, &nCurrentIndex);
			ERRIFGOTO(result, _EXIT);

			nNumOfTasks = pstCurrentTaskGraph->nNumOfTasks;
		}

		result = UCDynamicStack_Length(hStack, &nStackNum);
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result traverseAndHandleAllTasks(STask *pstTask, void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	FnCbHandleGeneralTask fnCallback = NULL;

	fnCallback = (FnCbHandleGeneralTask) pUserData;

	result = fnCallback(pstTask);
	ERRIFGOTO(result, _EXIT);
_EXIT:
	return result;
}


static uem_result handleSubgraphTasks(STask *pstParentTask, FnCbHandleGeneralTask fnCallback)
{
	uem_result result = ERR_UEM_UNKNOWN;
	HStack hStack = NULL;

	if(pstParentTask != NULL)
	{
		result = UCDynamicStack_Create(&hStack);
		ERRIFGOTO(result, _EXIT);

		result = callHierarchicalTaskGraphInitWrapupFunctions(pstParentTask, fnCallback, hStack);
		ERRIFGOTO(result, _EXIT);
	}
	else
	{
		result = UKTask_TraverseAllTasks(traverseAndHandleAllTasks, fnCallback);
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	if(hStack != NULL)
	{
		UCDynamicStack_Destroy(&hStack, NULL, NULL);
	}
	return result;
}


static uem_result setChannelExitFlags(STask *pstLeafTask)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = UKChannel_SetExitByTaskId(pstLeafTask->nTaskId);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result changeCompositeTaskState(SCompositeTask *pstTask, ECPUTaskState enTargetState, HLinkedList hTaskList)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SCompositeTaskStateChangeData stCompositeTaskData;

	stCompositeTaskData.enTaskState = enTargetState;
	stCompositeTaskData.pstCompositeTask = pstTask;

	// Change all composite task state
	result = UCDynamicLinkedList_Traverse(hTaskList, traverseAndChangeTaskState, &stCompositeTaskData);
	ERRIFGOTO(result, _EXIT);

	result = UCThreadMutex_Lock(pstTask->hMutex);
	ERRIFGOTO(result, _EXIT);

	pstTask->enTaskState = enTargetState;

	result = UCThreadMutex_Unlock(pstTask->hMutex);
	ERRIFGOTO(result, _EXIT);

	if(enTargetState == TASK_STATE_STOPPING && pstTask->pstParentTask != NULL && pstTask->pstParentTask->pstSubGraph != NULL)
	{
		// release channel block related to the task to be stopped
		result = handleSubgraphTasks(pstTask->pstParentTask, setChannelExitFlags);
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result handleCompositeTaskModeTransition(SCompositeTaskThread *pstTaskThread, SCompositeTask *pstTask)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstCurrentTask = NULL;

	pstCurrentTask = pstTask->pstParentTask;

	result = UCThreadMutex_Lock(pstCurrentTask->hMutex);
	ERRIFGOTO(result, _EXIT);

	// Mode transition is happened (suspend current thread and wait for other threads to be suspended)
	if(pstCurrentTask->pstMTMInfo->nCurModeIndex != pstCurrentTask->pstMTMInfo->nNextModeIndex)
	{
		pstCurrentTask->pstMTMInfo->nCurModeIndex = pstCurrentTask->pstMTMInfo->nNextModeIndex;

		// rerun composite task with new mode id
		result = changeCompositeTaskState(pstTask, TASK_STATE_RUNNING, pstTask->hThreadList);
		ERRIFGOTO(result, _EXIT_LOCK);
	}

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
	UCThreadMutex_Unlock(pstCurrentTask->hMutex);
_EXIT:
	return result;
}


static uem_result handleTaskMainRoutine(SCompositeTask *pstTask, SCompositeTaskThread *pstTaskThread, FnUemTaskGo fnGo)
{
	uem_result result = ERR_UEM_UNKNOWN;
	long long llNextTime = 0;
	int nMaxRunCount = 0;
	int nRunCount = 0;
	ERunCondition enRunCondition;

	if(pstTask->pstParentTask == NULL)
	{
		enRunCondition = RUN_CONDITION_DATA_DRIVEN;
	}
	else
	{
		enRunCondition = pstTask->pstParentTask->enRunCondition;
	}

	result = waitRunSignal(pstTask, pstTaskThread, TRUE, &llNextTime, &nMaxRunCount);
	ERRIFGOTO(result, _EXIT);

	while(pstTaskThread->enTaskState != TASK_STATE_STOP)
	{
		switch(pstTaskThread->enTaskState)
		{
		case TASK_STATE_RUNNING:
			switch(enRunCondition)
			{
			case RUN_CONDITION_TIME_DRIVEN:
				// pstCurrentTask is not NULL because whole task graph is a data-driven task graph
				result = handleTimeDrivenTask(pstTask->pstParentTask, fnGo, &llNextTime, &nRunCount, &nMaxRunCount);
				ERRIFGOTO(result, _EXIT);
				break;
			case RUN_CONDITION_DATA_DRIVEN:
				if(pstTask->pstParentTask != NULL)
				{
					fnGo(pstTask->pstParentTask->nTaskId);
				}
				else // if pstCurrentTask is NULL, the whole task graph is a composite task
				{
					fnGo(INVALID_TASK_ID);
				}
				break;
			case RUN_CONDITION_CONTROL_DRIVEN: // run once for control-driven leaf task
				fnGo(pstTask->pstParentTask->nTaskId);
				break;
			default:
				ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
				break;
			}
			if(isModeTransitionTask(pstTask) == TRUE)
			{
				result = handleCompositeTaskModeTransition(pstTaskThread, pstTask);
				ERRIFGOTO(result, _EXIT);
			}
			break;
		case TASK_STATE_STOPPING:
			UEMASSIGNGOTO(result, ERR_UEM_NOERROR, _EXIT);
			break;
		case TASK_STATE_STOP:
			// do nothing
			break;
		case TASK_STATE_SUSPEND:
			result = UCThreadMutex_Lock(pstTask->hMutex);
			ERRIFGOTO(result, _EXIT);

			pstTaskThread->bSuspended = TRUE;

			result = UCThreadMutex_Unlock(pstTask->hMutex);
			ERRIFGOTO(result, _EXIT);

			result = waitRunSignal(pstTask, pstTaskThread, FALSE, &llNextTime, &nMaxRunCount);
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


static void *compositeTaskThreadRoutine(void *pData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SCompositeTaskThreadData *pstThreadData = NULL;
	SCompositeTaskThread *pstTaskThread = NULL;
	SCompositeTask *pstTask = NULL;

	pstThreadData = (struct _SCompositeTaskThreadData *) pData;

	pstTask = pstThreadData->pstCompositeTask;
	pstTaskThread = pstThreadData->pstTaskThread;
	// pstThreadData->nCurSeqId

	result = handleTaskMainRoutine(pstTask, pstTaskThread, pstTaskThread->fnCompositeGo);
	ERRIFGOTO(result, _EXIT);

_EXIT:
	if(pstTaskThread != NULL && pstTask != NULL)
	{
		UCThreadMutex_Lock(pstTask->hMutex);
		pstTaskThread->bIsThreadFinished = TRUE;
		UCThreadMutex_Unlock(pstTask->hMutex);
	}
	SAFEMEMFREE(pstThreadData);
	return NULL;
}


static uem_result createCompositeTaskThread(IN int nOffset, IN void *pData, IN void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCompositeTaskThread *pstTaskThread = NULL;
	SCompositeTask *pstCompositeTask = NULL;
	struct _SCompositeTaskCreateData *pstCreateData = NULL;
	struct _SCompositeTaskThreadData *pstTaskThreadData = NULL;
	// int nModeId = INVALID_MODE_ID;

	pstTaskThread = (SCompositeTaskThread *) pData;
	pstCreateData = (struct _SCompositeTaskCreateData *) pUserData;
	pstCompositeTask = pstCreateData->pstCompositeTask;

	result = UCThreadMutex_Lock(pstCompositeTask->hMutex);
	ERRIFGOTO(result, _EXIT);

	if(pstCompositeTask->nCurrentThroughputConstraint == pstTaskThread->nThroughputConstraint)
	{
		pstTaskThread->bIsThreadFinished = FALSE;

		result = UCThreadMutex_Unlock(pstCompositeTask->hMutex);
		ERRIFGOTO(result, _EXIT);

		pstTaskThreadData = UC_malloc(sizeof(struct _SCompositeTaskThreadData));
		ERRMEMGOTO(pstTaskThreadData, result, _EXIT);

		pstTaskThreadData->pstCompositeTask = pstCreateData->pstCompositeTask;
		pstTaskThreadData->pstTaskThread = pstTaskThread;

		result = UCThread_Create(compositeTaskThreadRoutine, pstTaskThreadData, &(pstTaskThread->hThread));
		ERRIFGOTO(result, _EXIT);

		pstTaskThreadData = NULL;

		if(pstTaskThread->nProcId != MAPPING_NOT_SPECIFIED)
		{
			result = UCThread_SetMappedCPU(pstTaskThread->hThread, pstTaskThread->nProcId);
			ERRIFGOTO(result, _EXIT);
		}
	}
	else
	{
		result = UCThreadMutex_Unlock(pstCompositeTask->hMutex);
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

static uem_result findCompositeTask(IN int nOffset, IN void *pData, IN void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCompositeTask *pstTaskStruct = NULL;
	struct _SCompositeTaskSearchData *pstSearchData = NULL;

	pstTaskStruct = (SCompositeTask *) pData;
	pstSearchData = (struct _SCompositeTaskSearchData *) pUserData;

	if(pstTaskStruct->pstParentTask->nTaskId == pstSearchData->nTargetParentTaskId)
	{
		pstSearchData->pstMatchingCompositeTask = pstTaskStruct;
		UEMASSIGNGOTO(result, ERR_UEM_FOUND_DATA, _EXIT);
	}

    result = ERR_UEM_NOERROR;
_EXIT:
    return result;
}

static uem_result setCachedCompositeTask(SCPUCompositeTaskManager *pstTaskManager, SCompositeTask *pstCompositeTask)
{
	uem_result result = ERR_UEM_UNKNOWN;
	result = UCThreadMutex_Lock(pstTaskManager->hMutex);
	ERRIFGOTO(result, _EXIT);

	pstTaskManager->pstCachedCompositeTask = pstCompositeTask;

	result = UCThreadMutex_Unlock(pstTaskManager->hMutex);
	ERRIFGOTO(result, _EXIT);


    result = ERR_UEM_NOERROR;
_EXIT:
    return result;
}

static uem_result getCachedCompositeTask(SCPUCompositeTaskManager *pstTaskManager, int nTaskId, OUT SCompositeTask **ppstCompositeTask)
{
	SCompositeTask *pstTask = NULL;
	uem_result result = ERR_UEM_UNKNOWN;

	result = UCThreadMutex_Lock(pstTaskManager->hMutex);
	ERRIFGOTO(result, _EXIT);

	pstTask = pstTaskManager->pstCachedCompositeTask;

	result = UCThreadMutex_Unlock(pstTaskManager->hMutex);
	ERRIFGOTO(result, _EXIT);

	if(pstTask == NULL)
	{
		*ppstCompositeTask = NULL;
		result = ERR_UEM_NOT_FOUND;
	}
	else if(pstTask->pstParentTask == NULL && nTaskId == INVALID_TASK_ID) // whole task graph
	{
		*ppstCompositeTask = pstTask;
	}
	else if(pstTask->pstParentTask->nTaskId == nTaskId)
	{
		*ppstCompositeTask = pstTask;
	}
	else
	{
		*ppstCompositeTask = NULL;
		result = ERR_UEM_NOT_FOUND;
	}


_EXIT:
	return result;
}

static uem_result findMatchingCompositeTask(SCPUCompositeTaskManager *pstTaskManager, int nTaskId, OUT SCompositeTask **ppstCompositeTask)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCompositeTask *pstCompositeTask = NULL;
	struct _SCompositeTaskSearchData stSearchData;

	// check cache
	result = getCachedCompositeTask(pstTaskManager, nTaskId, &pstCompositeTask);
	if(result == ERR_UEM_NOT_FOUND)
	{
		stSearchData.nTargetParentTaskId = nTaskId;
		stSearchData.pstMatchingCompositeTask = NULL;

		result = UCDynamicLinkedList_Traverse(pstTaskManager->hTaskList, findCompositeTask, &stSearchData);
		ERRIFGOTO(result, _EXIT);

		if(result == ERR_UEM_FOUND_DATA)
		{
			pstCompositeTask = stSearchData.pstMatchingCompositeTask;
			setCachedCompositeTask(pstTaskManager, pstCompositeTask);
		}
		else
		{
			UEMASSIGNGOTO(result, ERR_UEM_NOT_FOUND, _EXIT);
		}
	}

	*ppstCompositeTask = pstCompositeTask;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKCPUCompositeTaskManager_RegisterTask(HCPUCompositeTaskManager hManager, SMappedCompositeTaskInfo *pstMappedTask)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUCompositeTaskManager *pstTaskManager = NULL;
	SCompositeTask *pstCompositeTask = NULL;
	int nTaskId = INVALID_TASK_ID;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(pstMappedTask, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	if (IS_VALID_HANDLE(hManager, ID_UEM_CPU_COMPOSITE_TASK_MANAGER) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}
#endif
	pstTaskManager = hManager;

	if(pstMappedTask->pstScheduledTasks->pstParentTask != NULL)
	{
		nTaskId = pstMappedTask->pstScheduledTasks->pstParentTask->nTaskId;
	}

	result = findMatchingCompositeTask(pstTaskManager, nTaskId, &pstCompositeTask);
	if(result == ERR_UEM_NOT_FOUND)
	{
		result = createCompositeTaskStruct(hManager, pstMappedTask, &pstCompositeTask);
		ERRIFGOTO(result, _EXIT);

		result = UCDynamicLinkedList_Add(pstTaskManager->hTaskList, LINKED_LIST_OFFSET_FIRST, 0, pstCompositeTask);
	}
	ERRIFGOTO(result, _EXIT);

	result = createCompositeTaskThreadStructs(pstMappedTask, pstCompositeTask->hThreadList);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result callInitFunction(STask *pstLeafTask)
{
	int nLoop = 0;

	for(nLoop = 0; nLoop < pstLeafTask->nTaskFunctionSetNum ; nLoop++)
	{
		pstLeafTask->astTaskFunctions[nLoop].fnInit(pstLeafTask->nTaskId);
	}

	return ERR_UEM_NOERROR;
}

static uem_result callWrapupFunction(STask *pstLeafTask)
{
	int nLoop = 0;

	for(nLoop = 0; nLoop < pstLeafTask->nTaskFunctionSetNum ; nLoop++)
	{
		pstLeafTask->astTaskFunctions[nLoop].fnWrapup();
	}

	return ERR_UEM_NOERROR;
}


static uem_result clearChannelExitFlags(STask *pstLeafTask)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = UKChannel_ClearExitByTaskId(pstLeafTask->nTaskId);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKCPUCompositeTaskManager_CreateThread(HCPUCompositeTaskManager hManager, STask *pstTargetTask)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUCompositeTaskManager *pstTaskManager = NULL;
	SCompositeTask *pstCompositeTask = NULL;
	struct _SCompositeTaskCreateData stCreateData;
	int nTaskId = INVALID_TASK_ID;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(pstTargetTask, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	if (IS_VALID_HANDLE(hManager, ID_UEM_CPU_COMPOSITE_TASK_MANAGER) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}
#endif
	pstTaskManager = hManager;

	if(pstTargetTask != NULL)
	{
		nTaskId = pstTargetTask->nTaskId;
	}

	result = findMatchingCompositeTask(pstTaskManager, nTaskId, &pstCompositeTask);
	ERRIFGOTO(result, _EXIT);

	// call init functions
	result = handleSubgraphTasks(pstCompositeTask->pstParentTask, callInitFunction);
	ERRIFGOTO(result, _EXIT);

	stCreateData.pstCompositeTask = pstCompositeTask;

	result = UCDynamicLinkedList_Traverse(pstCompositeTask->hThreadList, createCompositeTaskThread, &stCreateData);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKCPUCompositeTaskManager_ChangeState(HCPUCompositeTaskManager hManager, STask *pstTargetTask, ECPUTaskState enTaskState)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUCompositeTaskManager *pstTaskManager = NULL;
	SCompositeTask *pstCompositeTask = NULL;
	int nTaskId = INVALID_TASK_ID;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(pstTargetTask, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	if (IS_VALID_HANDLE(hManager, ID_UEM_CPU_COMPOSITE_TASK_MANAGER) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}
#endif
	pstTaskManager = hManager;

	if(pstTargetTask != NULL)
	{
		nTaskId = pstTargetTask->nTaskId;
	}

	result = findMatchingCompositeTask(pstTaskManager, nTaskId, &pstCompositeTask);
	ERRIFGOTO(result, _EXIT);

	result = changeCompositeTaskState(pstCompositeTask, enTaskState, pstCompositeTask->hThreadList);
	ERRIFGOTO(result, _EXIT);


	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKCPUCompositeTaskManager_ActivateThread(HCPUCompositeTaskManager hManager, STask *pstTargetTask)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUCompositeTaskManager *pstTaskManager = NULL;
	SCompositeTask *pstCompositeTask = NULL;
	int nTaskId = INVALID_TASK_ID;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(pstTargetTask, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	if (IS_VALID_HANDLE(hManager, ID_UEM_CPU_COMPOSITE_TASK_MANAGER) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}
#endif
	pstTaskManager = hManager;

	if(pstTargetTask != NULL)
	{
		nTaskId = pstTargetTask->nTaskId;
	}

	result = findMatchingCompositeTask(pstTaskManager, nTaskId, &pstCompositeTask);
	ERRIFGOTO(result, _EXIT);

	result = activateTask(pstCompositeTask);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result traverseAndCheckStoppingThread(IN int nOffset, IN void *pData, IN void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCompositeTaskThread *pstTaskThread = NULL;
	SCompositeTask *pstCompositeTask = NULL;
	struct _SCompositeTaskStopCheck *pstStopCheck = NULL;

	pstTaskThread = (SCompositeTaskThread *) pData;
	pstStopCheck = (struct _SCompositeTaskStopCheck *) pUserData;

	pstCompositeTask = pstStopCheck->pstCompositeTask;

	if(pstTaskThread->bIsThreadFinished == TRUE)
	{
		result = UCThread_Destroy(&(pstTaskThread->hThread), FALSE, THREAD_DESTROY_TIMEOUT);
		ERRIFGOTO(result, _EXIT);

		result = UCThreadMutex_Lock(pstCompositeTask->hMutex);
		ERRIFGOTO(result, _EXIT);

		pstTaskThread->enTaskState = TASK_STATE_STOP;

		result = UCThreadMutex_Unlock(pstCompositeTask->hMutex);
		ERRIFGOTO(result, _EXIT);
	}

	if(pstTaskThread->enTaskState != TASK_STATE_STOP)
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

uem_result UKCPUCompositeTaskManager_GetTaskState(HCPUCompositeTaskManager hManager, STask *pstTargetTask, ECPUTaskState *penTaskState)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUCompositeTaskManager *pstTaskManager = NULL;
	SCompositeTask *pstCompositeTask = NULL;
	struct _SCompositeTaskStopCheck stStopCheck;
	int nTaskId = INVALID_TASK_ID;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(pstTargetTask, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	IFVARERRASSIGNGOTO(penTaskState, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	if (IS_VALID_HANDLE(hManager, ID_UEM_CPU_COMPOSITE_TASK_MANAGER) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}
#endif

	pstTaskManager = hManager;

	if(pstTargetTask != NULL)
	{
		nTaskId = pstTargetTask->nTaskId;
	}

	result = findMatchingCompositeTask(pstTaskManager, nTaskId, &pstCompositeTask);
	ERRIFGOTO(result, _EXIT);

	if(pstCompositeTask->enTaskState == TASK_STATE_STOPPING)
	{
		stStopCheck.bAllStop = TRUE;
		stStopCheck.pstCompositeTask = pstCompositeTask;

		result = UCDynamicLinkedList_Traverse(pstCompositeTask->hThreadList, traverseAndCheckStoppingThread, &stStopCheck);
		ERRIFGOTO(result, _EXIT);

		if(stStopCheck.bAllStop == TRUE)
		{
			result = UCThreadMutex_Lock(pstCompositeTask->hMutex);
			ERRIFGOTO(result, _EXIT);

			pstCompositeTask->enTaskState = TASK_STATE_STOP;

			result = UCThreadMutex_Lock(pstCompositeTask->hMutex);
			ERRIFGOTO(result, _EXIT);
		}
	}

	*penTaskState = pstCompositeTask->enTaskState;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}



static uem_result traverseAndDestroyThread(IN int nOffset, IN void *pData, IN void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCompositeTaskThread *pstTaskThread = (SCompositeTaskThread *) pData;
	SCompositeTask *pstCompositeTask = (SCompositeTask *) pUserData;

	if(pstTaskThread->hThread != NULL)
	{
		result = UCThread_Destroy(&(pstTaskThread->hThread), FALSE, THREAD_DESTROY_TIMEOUT);
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	if(result != ERR_UEM_NOERROR)
	{
		if(pstCompositeTask->pstParentTask != NULL)
		{
			printf("Failed to destroy composite task of %s (%d)\n", pstCompositeTask->pstParentTask->pszTaskName, pstTaskThread->bIsThreadFinished);
		}
		else
		{
			printf("Failed to destroy composite task of whole task graph (%d)\n", pstTaskThread->bIsThreadFinished);
		}
	}
	return result;
}


uem_result UKCPUCompositeTaskManager_DestroyThread(HCPUCompositeTaskManager hManager, STask *pstTargetTask)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUCompositeTaskManager *pstTaskManager = NULL;
	SCompositeTask *pstCompositeTask = NULL;
	struct _SCompositeTaskStateChangeData stUserData;
	int nTaskId = INVALID_TASK_ID;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(pstTargetTask, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	if (IS_VALID_HANDLE(hManager, ID_UEM_CPU_COMPOSITE_TASK_MANAGER) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}
#endif
	pstTaskManager = hManager;

	if(pstTargetTask != NULL)
	{
		nTaskId = pstTargetTask->nTaskId;
	}

	result = findMatchingCompositeTask(pstTaskManager, nTaskId, &pstCompositeTask);
	ERRIFGOTO(result, _EXIT);

	stUserData.enTaskState = TASK_STATE_STOP;
	stUserData.pstCompositeTask = pstCompositeTask;

	result = UCDynamicLinkedList_Traverse(pstCompositeTask->hThreadList, traverseAndChangeTaskState, &stUserData);
	ERRIFGOTO(result, _EXIT);

	// set channel block free
	result = handleSubgraphTasks(pstCompositeTask->pstParentTask, setChannelExitFlags);
	ERRIFGOTO(result, _EXIT);

	result = UCDynamicLinkedList_Traverse(pstCompositeTask->hThreadList, traverseAndDestroyThread, pstCompositeTask);
	ERRIFGOTO(result, _EXIT);

	// call wrapup functions
	result = handleSubgraphTasks(pstCompositeTask->pstParentTask, callWrapupFunction);
	ERRIFGOTO(result, _EXIT);

	// clear channel exit flags
	result = handleSubgraphTasks(pstCompositeTask->pstParentTask, clearChannelExitFlags);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result traverseAndDestroyCompositeTask(IN int nOffset, IN void *pData, IN void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCompositeTask *pstCompositeTask = NULL;
	struct _SCompositeTaskStateChangeData stUserData;

	pstCompositeTask = (SCompositeTask *) pUserData;

	stUserData.enTaskState = TASK_STATE_STOP;
	stUserData.pstCompositeTask = pstCompositeTask;

	result = UCDynamicLinkedList_Traverse(pstCompositeTask->hThreadList, traverseAndChangeTaskState, &stUserData);
	ERRIFGOTO(result, _EXIT);

	// set channel block free
	result = handleSubgraphTasks(pstCompositeTask->pstParentTask, setChannelExitFlags);
	ERRIFGOTO(result, _EXIT);

	result = UCDynamicLinkedList_Traverse(pstCompositeTask->hThreadList, traverseAndDestroyThread, pstCompositeTask);
	ERRIFGOTO(result, _EXIT);

	// call wrapup functions
	result = handleSubgraphTasks(pstCompositeTask->pstParentTask, callWrapupFunction);
	ERRIFGOTO(result, _EXIT);

	// clear channel exit flags
	result = handleSubgraphTasks(pstCompositeTask->pstParentTask, clearChannelExitFlags);
	ERRIFGOTO(result, _EXIT);

	result = destroyCompositeTaskStruct(&pstCompositeTask);
	ERRIFGOTO(result, _EXIT);

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

	result = UCDynamicLinkedList_Traverse(pstTaskManager->hTaskList, traverseAndDestroyCompositeTask, NULL);
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
