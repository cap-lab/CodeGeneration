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
	int nSeqId;
	int nModeId;
	int nThroughputConstraint;
	int nProcId;
	FnUemTaskGo fnCompositeGo;
	ECPUTaskState enTaskState;
	HThread hThread;
	uem_bool bIsThreadFinished;
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
	int nCurrentThroughputConstraint;
} SCompositeTask;


typedef struct _SCPUCompositeTaskManager {
	EUemModuleId enId;
	HLinkedList hTaskList;
	uem_bool bListStatic;
	HThreadMutex hMutex;
} SCPUCompositeTaskManager;

struct _SCompositeTaskSearchData {
	int nTargetParentTaskId;
	SMappedCompositeTaskInfo *pstMappedInfo;
	SCompositeTask *pstMatchingCompositeTask;
};


struct _SCompositeTaskCreateData {
	SCompositeTask *pstCompositeTask;
};

struct _SCompositeTaskStateChangeData {
	SCompositeTask *pstCompositeTask;
	ECPUTaskState enTaskState;
};

typedef struct _SCompositeTaskThreadData {
	SCompositeTaskThread *pstTaskThread;
	SCompositeTask *pstCompositeTask;
} SCompositeTaskThreadData;

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
	pstCompositeTaskThread->hThread = NULL;
	pstCompositeTaskThread->bIsThreadFinished = TRUE;

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
	pstCompositeTask->nWaitingThreadNum = 0;
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


static uem_result waitRunSignal(SCompositeTask *pstTask, uem_bool bStartWait, OUT long long *pllNextTime, OUT int *pnNextMaxRunCount)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstCurrentTask = NULL;
	long long llCurTime = 0;

	pstCurrentTask = pstTask->pstParentTask;

	result = UCThreadEvent_WaitEvent(pstTask->hEvent);
	ERRIFGOTO(result, _EXIT);

	if(pstTask->enTaskState == TASK_STATE_RUNNING || bStartWait == TRUE)
	{
		result = UCThreadMutex_Lock(pstTask->hMutex);
		ERRIFGOTO(result, _EXIT);

		pstTask->nWaitingThreadNum--;

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


static uem_result suspendCurrentTaskThread(SCompositeTaskThread *pstTaskThread)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUCompositeTaskManager *pstManager = NULL;

	pstManager = pstTaskThread->hManager;

	result = UCThreadMutex_Lock(pstManager->hMutex);
	ERRIFGOTO(result, _EXIT);

	pstTaskThread->enTaskState = TASK_STATE_SUSPEND;

	result = ERR_UEM_NOERROR;

	UCThreadMutex_Unlock(pstManager->hMutex);
_EXIT:
	return result;
}

static uem_result checkTaskIsSuspended(SCompositeTaskThread *pstTaskThread, uem_bool *pbIsSuspended)
{
	uem_bool bIsSuspended = *pbIsSuspended;

	if(bIsSuspended == TRUE && pstTaskThread->enTaskState == TASK_STATE_RUNNING)
	{
		bIsSuspended = FALSE;
	}

	*pbIsSuspended = bIsSuspended;

	return ERR_UEM_NOERROR;
}


static uem_result traverseAndCheckIsSuspended(IN int nOffset, IN void *pData, IN void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;

	SCompositeTaskThread *pstTaskThread = (SCompositeTaskThread *) pData;
	uem_bool *pbIsSuspended = (uem_bool *) pUserData;

	result = checkTaskIsSuspended(pstTaskThread, pbIsSuspended);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
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


static uem_result traverseAndChangeTaskState(IN int nOffset, IN void *pData, IN void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SCompositeTaskStateChangeData *pstUserData = NULL;
	int nCurModeIndex = 0;
	int nCurModeId = 0;

	SCompositeTaskThread *pstTaskThread = (SCompositeTaskThread *) pData;
	pstUserData = (struct _SCompositeTaskStateChangeData *) pUserData;

	result = checkTaskThreadState(pstTaskThread->enTaskState, pstUserData->enTaskState);
	ERRIFGOTO(result, _EXIT);

	// For MTM task
	if(pstUserData->pstCompositeTask->pstParentTask != NULL && pstUserData->pstCompositeTask->pstParentTask->pstMTMInfo != NULL)
	{
		nCurModeIndex = pstUserData->pstCompositeTask->pstParentTask->pstMTMInfo->nCurModeIndex;
		nCurModeId = pstUserData->pstCompositeTask->pstParentTask->pstMTMInfo->astModeMap[nCurModeIndex].nModeId;
	}
	else
	{
		nCurModeId = 0;
	}

	if(pstTaskThread->nModeId == nCurModeId)
	{
		if(pstTaskThread->enTaskState != TASK_STATE_STOP)
		{
			pstTaskThread->enTaskState = pstUserData->enTaskState;
		}
	}
	else
	{
		if(pstUserData->enTaskState == TASK_STATE_RUNNING) // Different modes needs to be suspended when specific mode becomes running
		{
			pstTaskThread->enTaskState = TASK_STATE_SUSPEND;
		}
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result activateTask(SCompositeTask *pstTask)
{
	uem_result result = ERR_UEM_UNKNOWN;

	while(pstTask->nWaitingThreadNum > 0)
	{
		result = UCThreadEvent_SetEvent(pstTask->hEvent);
		ERRIFGOTO(result, _EXIT);

		UCThread_Yield();
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static uem_result traverseAndActivateTaskThread(IN int nOffset, IN void *pData, IN void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SCompositeTaskStateChangeData *pstUserData = NULL;
	int nCurModeIndex = 0;
	int nCurModeId = 0;

	SCompositeTaskThread *pstTaskThread = (SCompositeTaskThread *) pData;
	pstUserData = (struct _SCompositeTaskStateChangeData *) pUserData;

	result = checkTaskThreadState(pstTaskThread->enTaskState, pstUserData->enTaskState);
	ERRIFGOTO(result, _EXIT);

	// For MTM task
	if(pstUserData->pstCompositeTask->pstParentTask != NULL && pstUserData->pstCompositeTask->pstParentTask->pstMTMInfo != NULL)
	{
		nCurModeIndex = pstUserData->pstCompositeTask->pstParentTask->pstMTMInfo->nCurModeIndex;
		nCurModeId = pstUserData->pstCompositeTask->pstParentTask->pstMTMInfo->astModeMap[nCurModeIndex].nModeId;
	}
	else
	{
		nCurModeId = 0;
	}

	if(pstUserData->pstCompositeTask->nCurrentThroughputConstraint != pstTaskThread->nThroughputConstraint)
	{
		pstTaskThread->enTaskState = TASK_STATE_STOP;
	}
	else if(pstTaskThread->nModeId == nCurModeId)
	{
		pstTaskThread->enTaskState = TASK_STATE_RUNNING;
	}
	else
	{
		pstTaskThread->enTaskState = TASK_STATE_SUSPEND;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result changeCompositeTaskState(SCompositeTask *pstTask, ECPUTaskState enTargetState, HLinkedList hTaskList)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SCompositeTaskStateChangeData stCompositeTaskData;
	int nLen = 0;
	int nLoop = 0;

	stCompositeTaskData.enTaskState = enTargetState;
	stCompositeTaskData.pstCompositeTask = pstTask;

	// Change all composite task state
	result = UCDynamicLinkedList_Traverse(hTaskList, traverseAndChangeTaskState, &stCompositeTaskData);
	ERRIFGOTO(result, _EXIT);

	if(enTargetState == TASK_STATE_RUNNING)
	{
		result = UCDynamicLinkedList_Traverse(hTaskList, traverseAndActivateTaskThread, &stCompositeTaskData);
		ERRIFGOTO(result, _EXIT);

		result = activateTask(pstTask);
		ERRIFGOTO(result, _EXIT);
	}
	else if(enTargetState == TASK_STATE_STOPPING && pstTask->pstParentTask != NULL && pstTask->pstParentTask->pstSubGraph != NULL)
	{
		// release channel block related to the task to be stopped
		nLen = pstTask->pstParentTask->pstSubGraph->nNumOfTasks;

		for(nLoop = 0 ; nLoop < nLen ; nLoop++)
		{
			result = UKChannel_SetExitByTaskId(pstTask->pstParentTask->pstSubGraph->astTasks[nLoop].nTaskId);
			ERRIFGOTO(result, _EXIT);
		}
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result checkAndSwitchToNextModeInTaskLock(SCompositeTaskThread *pstTaskThread, SCompositeTask *pstTask)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUCompositeTaskManager *pstManager = NULL;
	uem_bool bIsSuspended = TRUE;
	STask *pstParentTask = NULL;

	pstManager = pstTaskThread->hManager;

	result = UCThreadMutex_Lock(pstManager->hMutex);
	ERRIFGOTO(result, _EXIT);

	result = UCDynamicLinkedList_Traverse(pstTask->hThreadList,
			traverseAndCheckIsSuspended, &bIsSuspended);
	ERRIFGOTO(result, _EXIT_LOCK);

	pstParentTask->pstMTMInfo->nCurModeIndex = pstParentTask->pstMTMInfo->nNextModeIndex;

	result = changeCompositeTaskState(pstTask, TASK_STATE_RUNNING, pstTask->hThreadList);
	ERRIFGOTO(result, _EXIT_LOCK);

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
	UCThreadMutex_Unlock(pstManager->hMutex);
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
		// change task state to suspend
		result = suspendCurrentTaskThread(pstTaskThread);
		ERRIFGOTO(result, _EXIT_LOCK);

		// check all composite task threads are suspended, and resume tasks with new mode
		result = checkAndSwitchToNextModeInTaskLock(pstTaskThread, pstTask);
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

	result = waitRunSignal(pstTask, TRUE, &llNextTime, &nMaxRunCount);
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
			result = waitRunSignal(pstTask, FALSE, &llNextTime, &nMaxRunCount);
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
	SCompositeTaskThreadData *pstThreadData = NULL;
	SCompositeTaskThread *pstTaskThread = NULL;
	SCompositeTask *pstTask = NULL;

	pstThreadData = (SCompositeTaskThreadData *) pData;

	pstTask = pstThreadData->pstCompositeTask;
	pstTaskThread = pstThreadData->pstTaskThread;
	// pstThreadData->nCurSeqId

	result = handleTaskMainRoutine(pstTask, pstTaskThread, pstTaskThread->fnCompositeGo);
	ERRIFGOTO(result, _EXIT);

_EXIT:
	if(pstTaskThread != NULL)
	{
		UCThreadMutex_Lock(pstThreadData->pstCompositeTask->hMutex);
		pstTask->nFinishedThreadNum++;
		UCThreadMutex_Unlock(pstThreadData->pstCompositeTask->hMutex);
	}
	SAFEMEMFREE(pstThreadData);
	return NULL;
}


static int getCurrentTaskModeId(STask *pstTask)
{
	int nCurModeIndex = INVALID_MODE_ID;
	int nModeId = INVALID_MODE_ID;
	if(pstTask->pstMTMInfo != NULL)
	{
		nCurModeIndex = pstTask->pstMTMInfo->nCurModeIndex;
		nModeId = pstTask->pstMTMInfo->astModeMap[nCurModeIndex].nModeId;
	}

	return nModeId;
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

	// nModeId = getCurrentTaskModeId(pstCompositeTask->pstParentTask);
	// nModeId == pstTaskThread->nModeId &&
	if(pstCompositeTask->nCurrentThroughputConstraint == pstTaskThread->nThroughputConstraint)
	{
		pstTaskThreadData = UC_malloc(sizeof(struct _SCompositeTaskThreadData));
		ERRMEMGOTO(pstTaskThreadData, result, _EXIT);

		pstTaskThreadData->pstCompositeTask = pstCreateData->pstCompositeTask;
		pstTaskThreadData->pstTaskThread = pstTaskThread;

		pstTaskThreadData->pstCompositeTask->nWaitingThreadNum++;

		result = UCThread_Create(compositeTaskThreadRoutine, pstTaskThreadData, &(pstTaskThread->hThread));
		ERRIFGOTO(result, _EXIT);

		if(pstTaskThread->nProcId != MAPPING_NOT_SPECIFIED)
		{
			result = UCThread_SetMappedCPU(pstTaskThread->hThread, pstTaskThread->nProcId);
			ERRIFGOTO(result, _EXIT);
		}
	}

    result = ERR_UEM_NOERROR;
_EXIT:
	if(result != ERR_UEM_NOERROR && pstTaskThread->hThread != NULL)
	{
		UCThread_Destroy(&(pstTaskThread->hThread), FALSE, THREAD_DESTROY_TIMEOUT);
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

	if(pstTaskStruct->pstParentTask->nTaskId == pstSearchData->nTargetParentTaskId)
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

	stSearchData.nTargetParentTaskId = pstMappedTask->pstScheduledTasks->pstParentTask->nTaskId;
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

	result = createCompositeTaskThreadStructs(pstMappedTask, pstCompositeTask->hThreadList);
	ERRIFGOTO(result, _EXIT);

	//result = UCDynamicLinkedList_Add(pstTaskManager->hTaskList, LINKED_LIST_OFFSET_FIRST, 0, pstTaskThread);g
	//ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static uem_result checkAndPopStack(HStack hStack, IN OUT SModeMap **ppstModeMap, IN OUT int *pnIndex)
{
	uem_result result = ERR_UEM_UNKNOWN;

	SModeMap *pstModeMap = NULL;
	int nIndex = 0;
	int nStackNum = 0;
	void *pIndex = NULL;

	pstModeMap = *ppstModeMap;
	nIndex = *pnIndex;

	result = UCDynamicStack_Length(hStack, &nStackNum);
	ERRIFGOTO(result, _EXIT);

	if(nIndex >= pstModeMap->nRelatedChildTaskNum && nStackNum > 0)
	{
		result = UCDynamicStack_Pop(hStack, &pIndex);
		ERRIFGOTO(result, _EXIT);

#if SIZEOF_VOID_P == 8
		nIndex = (int) ((long long) pIndex);
#else
		nIndex = (int) pIndex;
#endif

		result = UCDynamicStack_Pop(hStack, (void **) &pstModeMap);
		ERRIFGOTO(result, _EXIT);

		*ppstModeMap = pstModeMap;
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


static uem_result callCompositeTaskInitOrWrapupFunctions(STask *pstParentTask, uem_bool bCallInit, HStack hStack)
{
	STask *pstCurInitTask = NULL;
	uem_result result = ERR_UEM_UNKNOWN;
	SModeMap *pstCurrentModeMap = NULL;
	int nCurModeIndex = 0;
	int nCurrentIndex = 0;
	SModeMap *pstNextModeMap = NULL;
	int nNextModeIndex = 0;
	int nStackNum = 0;
	int nNumOfTasks = 0;
	int nLoop = 0;

	nCurModeIndex = pstParentTask->pstMTMInfo->nCurModeIndex;
	pstCurrentModeMap = &(pstParentTask->pstMTMInfo->astModeMap[nCurModeIndex]);
	nNumOfTasks = pstCurrentModeMap->nRelatedChildTaskNum;

	while(nCurrentIndex < nNumOfTasks || nStackNum > 0)
	{
		if(pstCurrentModeMap->pastRelatedChildTasks[nCurrentIndex]->pstSubGraph != NULL)
		{
			if(pstCurrentModeMap->pastRelatedChildTasks[nCurrentIndex]->pstMTMInfo != NULL) // MTM Graph
			{
				nNextModeIndex = pstCurrentModeMap->pastRelatedChildTasks[nCurrentIndex]->pstMTMInfo->nCurModeIndex;
				pstNextModeMap = &(pstCurrentModeMap->pastRelatedChildTasks[nCurrentIndex]->pstMTMInfo->astModeMap[nNextModeIndex]);

				// the current task has subgraph, skip current task index
				nCurrentIndex++;
				//result = checkAndPopStack(hStack, &pstCurrentModeMap, &nCurrentIndex);
				//ERRIFGOTO(result, _EXIT);

				if(nCurrentIndex < pstCurrentModeMap->nRelatedChildTaskNum)
				{
					result = UCDynamicStack_Push(hStack, pstCurrentModeMap);
					ERRIFGOTO(result, _EXIT);
	#if SIZEOF_VOID_P == 8
					result = UCDynamicStack_Push(hStack, (void *) (long long) nCurrentIndex);
	#else
					result = UCDynamicStack_Push(hStack, (void *) nCurrentIndex);
	#endif
					ERRIFGOTO(result, _EXIT);
				}

				// reset values
				pstCurrentModeMap = pstNextModeMap;
				nCurrentIndex = 0;
				nNumOfTasks = pstCurrentModeMap->nRelatedChildTaskNum;
			}
			else // Normal data flow graph
			{
				// impossible case (composite task contains a schedule result with the most deepest inner-tasks)
				ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
			}
		}
		else // does not have internal task
		{
			// call current index's task init function
			pstCurInitTask = pstCurrentModeMap->pastRelatedChildTasks[nCurrentIndex];
			for(nLoop = 0; nLoop < pstCurInitTask->nTaskFunctionSetNum ; nLoop++)
			{
				if(bCallInit == TRUE)
				{
					pstCurInitTask->astTaskFunctions[nLoop].fnInit(pstCurInitTask->nTaskId);
				}
				else // call wrapup function
				{
					pstCurInitTask->astTaskFunctions[nLoop].fnWrapup();
				}
			}

			// proceed index, if all index is proceeded, pop the mode map from stack
			nCurrentIndex++;

			result = checkAndPopStack(hStack, &pstCurrentModeMap, &nCurrentIndex);
			ERRIFGOTO(result, _EXIT);

			nNumOfTasks = pstCurrentModeMap->nRelatedChildTaskNum;
		}

		result = UCDynamicStack_Length(hStack, &nStackNum);
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKCPUCompositeTaskManager_CreateThread(HCPUCompositeTaskManager hManager, STask *pstTargetTask)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUCompositeTaskManager *pstTaskManager = NULL;
	SCompositeTask *pstCompositeTask = NULL;
	struct _SCompositeTaskSearchData stSearchData;
	HStack hStack = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(pstTargetTask, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	if (IS_VALID_HANDLE(hManager, ID_UEM_CPU_COMPOSITE_TASK_MANAGER) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}
#endif
	pstTaskManager = hManager;

	stSearchData.nTargetParentTaskId = pstTargetTask->nTaskId;
	stSearchData.pstMatchingCompositeTask = NULL;

	result = UCDynamicLinkedList_Traverse(pstTaskManager->hTaskList, findCompositeTask, &stSearchData);
	ERRIFGOTO(result, _EXIT);
	if(result == ERR_UEM_NOERROR)
	{
		ERRASSIGNGOTO(result, ERR_UEM_NOT_FOUND, _EXIT);
	}

	pstCompositeTask = stSearchData.pstMatchingCompositeTask;

	result = UCDynamicStack_Create(&hStack);
	ERRIFGOTO(result, _EXIT);

	// call init functions
	result = callCompositeTaskInitOrWrapupFunctions(pstCompositeTask->pstParentTask, TRUE, hStack);
	ERRIFGOTO(result, _EXIT);

	result = UCDynamicLinkedList_Traverse(pstCompositeTask->hThreadList, createCompositeTaskThread, &stSearchData);
	ERRIFGOTO(result, _EXIT);

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
