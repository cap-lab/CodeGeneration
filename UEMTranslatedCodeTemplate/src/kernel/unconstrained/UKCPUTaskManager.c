/*
 * UKCPUTaskManager.c
 *
 *  Created on: 2017. 9. 19.
 *      Author: jej
 */
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <UCBasic.h>
#include <UCDynamicLinkedList.h>
#include <UCThreadMutex.h>
#include <UCThreadEvent.h>
#include <UCThread.h>
#include <UCDynamicStack.h>
#include <UCTime.h>

#include <uem_data.h>

#include <UKCPUTaskManager.h>
#include <UKChannel.h>
#include <UKTask.h>


#define MIN_SLEEP_DURATION (10)
#define MAX_SLEEP_DURATION (100)

#define THREAD_DESTROY_TIMEOUT (5000)

typedef enum _EMappedTaskType {
	MAPPED_TYPE_COMPOSITE_TASK,
	MAPPED_TYPE_GENERAL_TASK,
} EMappedTaskType;

typedef enum _ECPUTaskState {
	TASK_STATE_STOP,
	TASK_STATE_RUNNING,
	TASK_STATE_SUSPEND,
	TASK_STATE_STOPPING,
} ECPUTaskState;


typedef union _UMappedCPUList {
	int *anCPUId;
	HLinkedList hMappedCPUList;
} UMappedCPUList;

typedef struct _STaskThread {
	HLinkedList hThreadList;
	int nSeqId;
	HThreadEvent hEvent;
	HThreadMutex hMutex;
	int nWaitingThreadNum;
	int nFinishedThreadNum;
	UMappingTarget uTargetTask;
	ECPUTaskState enTaskState;
	EMappedTaskType enMappedTaskType;
	UMappedCPUList uMappedCPUList;
	HCPUTaskManager hManager; // handle for accessing higher data structures
} STaskThread;

typedef union _UTaskList {
	STaskThread *astTaskThread;
	HLinkedList hTaskList;
} UTaskList;

typedef struct _SCPUTaskManager {
	EUemModuleId enId;
	UTaskList uDataAndTimeDrivenTaskList;
	UTaskList uControlDrivenTaskList;
	uem_bool bListStatic;
	HThreadMutex hMutex;
} SCPUTaskManager;

struct _STaskSearchUserData {
	int nTaskId;
	STaskThread *pstTargetThread;
};

struct _SParentTaskSearchUserData {
	int nTaskId;
	STask *pstTargetParentTask;
};

typedef uem_result (*FnCbHandleTaskThread)(STaskThread *pstTaskThread, void *pUserData);

struct _SCompositeTaskUserData {
	int nTaskId;
	FnCbHandleTaskThread fnCallback;
	void *pUserData;
};

struct _SChildTaskAccessUserData {
	int nParentTaskId;
	int nMatchedTaskNum;
	FnCbHandleTaskThread fnCallback;
	void *pUserData;
};



typedef struct _STaskThreadData {
	STaskThread *pstTaskThread;
	int nCurSeqId;
	int nTaskFunctionIndex;
} STaskThreadData;


uem_result UKCPUTaskManager_Create(OUT HCPUTaskManager *phCPUTaskManager)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUTaskManager *pstManager = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(phCPUTaskManager, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	pstManager = UC_malloc(sizeof(SCPUTaskManager));
	ERRMEMGOTO(pstManager, result, _EXIT);

	pstManager->enId = ID_UEM_CPU_TASK_MANAGER;
	pstManager->bListStatic = FALSE;
	pstManager->hMutex = NULL;
	pstManager->uDataAndTimeDrivenTaskList.hTaskList = NULL;
	pstManager->uControlDrivenTaskList.hTaskList = NULL;

	result = UCThreadMutex_Create(&(pstManager->hMutex));
	ERRIFGOTO(result, _EXIT);

	result = UCDynamicLinkedList_Create(&(pstManager->uDataAndTimeDrivenTaskList.hTaskList));
	ERRIFGOTO(result, _EXIT);

	result = UCDynamicLinkedList_Create(&(pstManager->uControlDrivenTaskList.hTaskList));
	ERRIFGOTO(result, _EXIT);

	*phCPUTaskManager = pstManager;

	result = ERR_UEM_NOERROR;
_EXIT:
	if(pstManager != NULL && result != ERR_UEM_NOERROR)
	{
		UCDynamicLinkedList_Destroy(&(pstManager->uDataAndTimeDrivenTaskList.hTaskList));
		UCThreadMutex_Destroy(&(pstManager->hMutex));
		SAFEMEMFREE(pstManager);
	}
	return result;
}

struct _STaskTraverseUserData {
	STask *pstTask;
	int nCPUId;
};


struct _SCompositeTaskTraverseUserData {
	SScheduledTasks *pstScheduledTasks;
	int nCPUId;
};

static uem_result traverseCPUList(IN int nOffset, IN void *pData, IN void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;

	int nData = 0;
	int nUserData = 0;

#if SIZEOF_VOID_P == 8
	nData = (int) ((long long) pData);
	nUserData = (int) ((long long) pUserData);
#else
	nData = (int) pData;
	nUserData = (int) pUserData;
#endif

	if(nData == nUserData)
	{
		UEMASSIGNGOTO(result, ERR_UEM_FOUND_DATA, _EXIT);
	}

    result = ERR_UEM_NOERROR;
_EXIT:
    return result;
}

static uem_result traverseTaskList(IN int nOffset, IN void *pData, IN void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STaskThread *pstTaskThread = NULL;
	struct _STaskTraverseUserData *pstUserData = NULL;
	void *pCPUId = 0;

	pstTaskThread = (STaskThread *) pData;
	pstUserData = (struct _STaskTraverseUserData *) pUserData;
	if(pstTaskThread->enMappedTaskType == MAPPED_TYPE_GENERAL_TASK &&
		pstTaskThread->uTargetTask.pstTask == pstUserData->pstTask)
	{
#if SIZEOF_VOID_P == 8
		pCPUId = (void *) (long long) pstUserData->nCPUId;
#else
		pCPUId = (void *) pstUserData->nCPUId;
#endif
		result = UCDynamicLinkedList_Traverse(pstTaskThread->uMappedCPUList.hMappedCPUList, traverseCPUList, pCPUId);
		ERRIFGOTO(result, _EXIT);
		if(result == ERR_UEM_FOUND_DATA)
		{
			ERRASSIGNGOTO(result, ERR_UEM_DATA_DUPLICATED, _EXIT);
		}

		result = UCDynamicLinkedList_Add(pstTaskThread->uMappedCPUList.hMappedCPUList, LINKED_LIST_OFFSET_LAST, 0, pCPUId);
		ERRIFGOTO(result, _EXIT);

		UEMASSIGNGOTO(result, ERR_UEM_FOUND_DATA, _EXIT);
	}

    result = ERR_UEM_NOERROR;
_EXIT:
    return result;
}


static uem_result traverseCompositeTaskList(IN int nOffset, IN void *pData, IN void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STaskThread *pstTaskThread = NULL;
	struct _SCompositeTaskTraverseUserData *pstUserData = NULL;

	pstTaskThread = (STaskThread *) pData;
	pstUserData = (struct _SCompositeTaskTraverseUserData *) pUserData;
	if(pstTaskThread->enMappedTaskType == MAPPED_TYPE_COMPOSITE_TASK &&
		pstTaskThread->uTargetTask.pstScheduledTasks == pstUserData->pstScheduledTasks)
	{
		UEMASSIGNGOTO(result, ERR_UEM_FOUND_DATA, _EXIT);
	}

    result = ERR_UEM_NOERROR;
_EXIT:
    return result;
}


static uem_result destroyTaskThreadStruct(IN OUT STaskThread **ppstTaskThread)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STaskThread *pstTaskThread = NULL;

	pstTaskThread = *ppstTaskThread;

	// TODO: Traverse all thread to be destroyed

	if(pstTaskThread->hThreadList != NULL)
	{
		UCDynamicLinkedList_Destroy(&(pstTaskThread->hThreadList));
	}

	if(pstTaskThread->uMappedCPUList.hMappedCPUList != NULL)
	{
		UCDynamicLinkedList_Destroy(&(pstTaskThread->uMappedCPUList.hMappedCPUList));
	}

	if(pstTaskThread->hEvent != NULL)
	{
		UCThreadEvent_Destroy(&(pstTaskThread->hEvent));
	}

	if(pstTaskThread->hMutex != NULL)
	{
		UCThreadMutex_Destroy(&(pstTaskThread->hMutex));
	}

	SAFEMEMFREE(pstTaskThread);

	*ppstTaskThread = NULL;

	result = ERR_UEM_NOERROR;

	return result;
}

static uem_result createTaskThreadStruct(uem_bool bIsCompositeTask, HCPUTaskManager hCPUTaskManager, UMappingTarget uTargetTask, OUT STaskThread **ppstTaskThread)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STaskThread *pstTaskThread = NULL;

	pstTaskThread = UC_malloc(sizeof(STaskThread));
	ERRMEMGOTO(pstTaskThread, result, _EXIT);

	pstTaskThread->enTaskState = TASK_STATE_STOP;
	pstTaskThread->hEvent = NULL;
	pstTaskThread->hThreadList = NULL;
	pstTaskThread->uMappedCPUList.hMappedCPUList = NULL;
	pstTaskThread->nSeqId = 0;
	pstTaskThread->nWaitingThreadNum = 0;
	pstTaskThread->nFinishedThreadNum = 0;
	pstTaskThread->hMutex = NULL;
	pstTaskThread->hManager = hCPUTaskManager;
	if(bIsCompositeTask == TRUE)
	{
		pstTaskThread->uTargetTask.pstScheduledTasks = uTargetTask.pstScheduledTasks;
		pstTaskThread->enMappedTaskType = MAPPED_TYPE_COMPOSITE_TASK;
	}
	else // bIsCompositeTask == FALSE
	{
		pstTaskThread->uTargetTask.pstTask = uTargetTask.pstTask;
		pstTaskThread->enMappedTaskType = MAPPED_TYPE_GENERAL_TASK;
	}

	result = UCThreadEvent_Create(&(pstTaskThread->hEvent));
	ERRIFGOTO(result, _EXIT);

	result = UCThreadMutex_Create(&(pstTaskThread->hMutex));
	ERRIFGOTO(result, _EXIT);

	result = UCDynamicLinkedList_Create(&(pstTaskThread->uMappedCPUList.hMappedCPUList));
	ERRIFGOTO(result, _EXIT);

	result = UCDynamicLinkedList_Create(&(pstTaskThread->hThreadList));
	ERRIFGOTO(result, _EXIT);

	*ppstTaskThread = pstTaskThread;

	result = ERR_UEM_NOERROR;
_EXIT:
	if(result != ERR_UEM_NOERROR && pstTaskThread != NULL)
	{
		destroyTaskThreadStruct(&pstTaskThread);
	}
	return result;
}

static uem_bool checkIsChildTask(int nParentTaskId, STask *pstChildTaskCandidate)
{
	STask *pstCurrentTask = NULL;
	uem_bool bIsChildTask = FALSE;

	pstCurrentTask = pstChildTaskCandidate;

	// if pstTask == NULL, it means the task is top task graph
	while(pstCurrentTask != NULL)
	{
		if (pstCurrentTask->pstParentGraph->pstParentTask != NULL &&
				pstCurrentTask->pstParentGraph->pstParentTask->nTaskId == nParentTaskId)
		{
			bIsChildTask = TRUE;
			break;
		}
		else
		{
			pstCurrentTask = pstCurrentTask->pstParentGraph->pstParentTask;
		}

	}

	return bIsChildTask;
}

static uem_bool checkIsControlDrivenTask(STask *pstTask)
{
	STask *pstCurrentTask = NULL;
	uem_bool bIsControlDriven = FALSE;

	pstCurrentTask = pstTask;

	// if pstTask == NULL, it means the task is top task graph
	while(pstCurrentTask != NULL)
	{
		if (pstCurrentTask->enRunCondition == RUN_CONDITION_CONTROL_DRIVEN)
		{
			bIsControlDriven = TRUE;
			break;
		}
		else
		{
			pstCurrentTask = pstCurrentTask->pstParentGraph->pstParentTask;
		}

	}

	return bIsControlDriven;
}

uem_result UKCPUTaskManager_RegisterTask(HCPUTaskManager hCPUTaskManager, STask *pstTask, int nCPUId)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUTaskManager *pstManager = NULL;
	STaskThread *pstTaskThread = NULL;
	UMappingTarget uTargetTask;
	struct _STaskTraverseUserData stUserData;
	void *pCPUId = 0;
	HLinkedList hTargetList = NULL;

#ifdef ARGUMENT_CHECK
	if (IS_VALID_HANDLE(hCPUTaskManager, ID_UEM_CPU_TASK_MANAGER) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}

	IFVARERRASSIGNGOTO(pstTask, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	if(nCPUId < 0 && nCPUId != MAPPING_NOT_SPECIFIED) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
#endif
	pstManager = hCPUTaskManager;

	stUserData.nCPUId = nCPUId;
	stUserData.pstTask = pstTask;

	if(checkIsControlDrivenTask(pstTask) == TRUE)
	{
		hTargetList = pstManager->uControlDrivenTaskList.hTaskList;

	}
	else
	{
		hTargetList = pstManager->uDataAndTimeDrivenTaskList.hTaskList;
	}

	// Find TaskThread is already created
	result = UCDynamicLinkedList_Traverse(hTargetList, traverseTaskList, &stUserData);
	ERRIFGOTO(result, _EXIT);
	if(result == ERR_UEM_NOERROR) // if there is no task thread structure in the list, create the new one
	{
		uTargetTask.pstTask = pstTask;

		result = createTaskThreadStruct(FALSE, hCPUTaskManager, uTargetTask, &pstTaskThread);
		ERRIFGOTO(result, _EXIT);

#if SIZEOF_VOID_P == 8
		pCPUId = (void *) (long long) nCPUId;
#else
		pCPUId = (void *) pstUserData->nCPUId;
#endif

		result = UCDynamicLinkedList_Add(pstTaskThread->uMappedCPUList.hMappedCPUList, LINKED_LIST_OFFSET_LAST, 0, pCPUId);
		ERRIFGOTO(result, _EXIT);

		result = UCDynamicLinkedList_Add(hTargetList, LINKED_LIST_OFFSET_FIRST, 0, pstTaskThread);
		ERRIFGOTO(result, _EXIT);
	}
	else // ERR_UEM_FOUND_DATA
	{
		// do nothing, already done
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	if(result != ERR_UEM_NOERROR && pstTaskThread != NULL)
	{
		destroyTaskThreadStruct(&pstTaskThread);
	}
	return result;
}


uem_result UKCPUTaskManager_RegisterCompositeTask(HCPUTaskManager hCPUTaskManager, SScheduledTasks *pstScheduledTasks, int nCPUId)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUTaskManager *pstManager = NULL;
	STaskThread *pstTaskThread = NULL;
	UMappingTarget uTargetTask;
	struct _SCompositeTaskTraverseUserData stUserData;
	void *pCPUId = 0;
	HLinkedList hTargetList = NULL;

#ifdef ARGUMENT_CHECK
	if (IS_VALID_HANDLE(hCPUTaskManager, ID_UEM_CPU_TASK_MANAGER) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}

	IFVARERRASSIGNGOTO(pstScheduledTasks, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	if(nCPUId < 0 && nCPUId != MAPPING_NOT_SPECIFIED) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
#endif
	pstManager = hCPUTaskManager;

	stUserData.nCPUId = nCPUId;
	stUserData.pstScheduledTasks = pstScheduledTasks;

	if(checkIsControlDrivenTask(pstScheduledTasks->pstParentTask) == TRUE)
	{
		hTargetList = pstManager->uControlDrivenTaskList.hTaskList;
	}
	else
	{
		hTargetList = pstManager->uDataAndTimeDrivenTaskList.hTaskList;
	}

	result = UCDynamicLinkedList_Traverse(hTargetList, traverseCompositeTaskList, &stUserData);
	ERRIFGOTO(result, _EXIT);
	if(result == ERR_UEM_NOERROR)
	{
		uTargetTask.pstScheduledTasks = pstScheduledTasks;

		result = createTaskThreadStruct(TRUE, hCPUTaskManager, uTargetTask, &pstTaskThread);
		ERRIFGOTO(result, _EXIT);
#if SIZEOF_VOID_P == 8
		pCPUId = (void *) (long long) nCPUId;
#else
		pCPUId = (void *) pstUserData->nCPUId;
#endif

		result = UCDynamicLinkedList_Add(pstTaskThread->uMappedCPUList.hMappedCPUList, LINKED_LIST_OFFSET_LAST, 0, pCPUId);
		ERRIFGOTO(result, _EXIT);

		result = UCDynamicLinkedList_Add(hTargetList, LINKED_LIST_OFFSET_FIRST, 0, pstTaskThread);
		ERRIFGOTO(result, _EXIT);
	}
	else if(result == ERR_UEM_FOUND_DATA)
	{
		// Composite task only mapped to a single core
		ERRASSIGNGOTO(result, ERR_UEM_DATA_DUPLICATED, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	if(result != ERR_UEM_NOERROR && pstTaskThread != NULL)
	{
		destroyTaskThreadStruct(&pstTaskThread);
	}
	return result;
}

#define SEC_UNIT (1000)
#define MINUTE_UNIT (60)
#define HOUR_UNIT (60)

static uem_result getNextTimeByPeriodAndMetric(long long llPrevTime, int nPeriod, ETimeMetric enPeriodMetric,
											OUT long long *pllNextTime, OUT int *pnNextMaxRunCount)
{
	uem_result result = ERR_UEM_UNKNOWN;

	switch(enPeriodMetric)
	{
	case TIME_METRIC_COUNT: // currently, same to 1 ms
		*pllNextTime = llPrevTime + 1 * nPeriod;
		*pnNextMaxRunCount = 1;
		break;
	case TIME_METRIC_CYCLE: // currently, same to 1 ms
		*pllNextTime = llPrevTime + 1 * nPeriod;
		*pnNextMaxRunCount = 1;
		break;
	case TIME_METRIC_MICROSEC: // TODO: micro-second time tick is even not correct
		if(nPeriod > 0 && nPeriod < SEC_UNIT)
		{
			*pnNextMaxRunCount = SEC_UNIT/nPeriod;
		}
		else
		{
			*pnNextMaxRunCount = 1;
		}

		if(nPeriod/1000 <= 0)
		{
			nPeriod = 1;
		}
		*pllNextTime = llPrevTime + 1 * nPeriod/1000;
		break;
	case TIME_METRIC_MILLISEC:
		*pllNextTime = llPrevTime + 1 * nPeriod;
		*pnNextMaxRunCount = 1;
		break;
	case TIME_METRIC_SEC:
		*pllNextTime = llPrevTime + SEC_UNIT * nPeriod;
		*pnNextMaxRunCount = 1;
		break;
	case TIME_METRIC_MINUTE:
		*pllNextTime = llPrevTime + SEC_UNIT * MINUTE_UNIT * nPeriod;
		*pnNextMaxRunCount = 1;
		break;
	case TIME_METRIC_HOUR:
		*pllNextTime = llPrevTime + SEC_UNIT * MINUTE_UNIT * HOUR_UNIT * nPeriod;
		*pnNextMaxRunCount = 1;
		break;
	default:
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
		break;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result waitRunSignal(STaskThread *pstTaskThread, uem_bool bStartWait, OUT long long *pllNextTime, OUT int *pnNextMaxRunCount)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstCurrentTask = NULL;
	long long llCurTime = 0;

	pstCurrentTask = pstTaskThread->uTargetTask.pstTask;

	result = UCThreadEvent_WaitEvent(pstTaskThread->hEvent);
	ERRIFGOTO(result, _EXIT);

	if(pstTaskThread->enTaskState == TASK_STATE_RUNNING || bStartWait == TRUE)
	{
		result = UCThreadMutex_Lock(pstTaskThread->hMutex);
		ERRIFGOTO(result, _EXIT);

		pstTaskThread->nWaitingThreadNum--;

		result = UCThreadMutex_Unlock(pstTaskThread->hMutex);
		ERRIFGOTO(result, _EXIT);

		if(pstCurrentTask->enRunCondition == RUN_CONDITION_TIME_DRIVEN)
		{
			result = UCTime_GetCurTickInMilliSeconds(&llCurTime);
			ERRIFGOTO(result, _EXIT);

			result = getNextTimeByPeriodAndMetric(llCurTime, pstCurrentTask->nPeriod, pstCurrentTask->enPeriodMetric,
																pllNextTime, pnNextMaxRunCount);
			ERRIFGOTO(result, _EXIT);
		}
	}

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

static uem_result traverseTaskAndCallInitFunctions(STask *pstTask, void *pUserData)
{
	int nLoop = 0;

	if(pstTask->enRunCondition == RUN_CONDITION_TIME_DRIVEN ||pstTask->enRunCondition == RUN_CONDITION_DATA_DRIVEN)
	{
		for(nLoop = 0 ; nLoop < pstTask->nTaskFunctionSetNum ; nLoop++)
		{
			pstTask->astTaskFunctions[nLoop].fnInit(pstTask->nTaskId);
		}
	}

	return ERR_UEM_NOERROR;
}

static uem_result traverseTaskAndCallWrapupFunctions(STask *pstTask, void *pUserData)
{
	int nLoop = 0;

	if(pstTask->enRunCondition == RUN_CONDITION_TIME_DRIVEN ||pstTask->enRunCondition == RUN_CONDITION_DATA_DRIVEN)
	{
		for(nLoop = 0 ; nLoop < pstTask->nTaskFunctionSetNum ; nLoop++)
		{
			pstTask->astTaskFunctions[nLoop].fnWrapup(pstTask->nTaskId);
		}
	}

	return ERR_UEM_NOERROR;
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

static uem_result handleTimeDrivenTask(STaskThread *pstTaskThread, STask *pstCurrentTask, FnUemTaskGo fnGo, IN OUT long long *pllNextTime,
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
		result = getNextTimeByPeriodAndMetric(llNextTime, pstCurrentTask->nPeriod, pstCurrentTask->enPeriodMetric,
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



static uem_result traverseCompositeTaskThreads(IN int nOffset, IN void *pData, IN void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SCompositeTaskUserData *pstUserData = pUserData;
	STaskThread *pstTaskThread = (STaskThread *) pData;

	if(pstTaskThread->enMappedTaskType == MAPPED_TYPE_COMPOSITE_TASK &&
		pstTaskThread->uTargetTask.pstScheduledTasks->pstParentTask->nTaskId == pstUserData->nTaskId)
	{
		result = pstUserData->fnCallback(pstTaskThread, pstUserData->pUserData);
		ERRIFGOTO(result, _EXIT);
	}

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


static uem_result checkAndChangeTaskState(STaskThread * pstTaskThread, void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	ECPUTaskState enTaskState = *((ECPUTaskState *)pUserData);

	result = checkTaskThreadState(pstTaskThread->enTaskState, enTaskState);
	ERRIFGOTO(result, _EXIT);
	if(result == ERR_UEM_ALREADY_DONE)
	{
		// TODO: do something?
	}

	if(pstTaskThread->enTaskState != TASK_STATE_STOP)
	{
		pstTaskThread->enTaskState = enTaskState;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_bool isModeTransitionTask(STaskThread *pstTaskThread)
{
	STask *pstCurrentTask = NULL;
	uem_bool bIsModeTransition = FALSE;
	int nLen = 0;

	if(pstTaskThread->enMappedTaskType == MAPPED_TYPE_COMPOSITE_TASK)
	{
		pstCurrentTask = pstTaskThread->uTargetTask.pstScheduledTasks->pstParentTask;
	}
	else // MAPPED_TYPE_GENERAL_TASK
	{
		pstCurrentTask = pstTaskThread->uTargetTask.pstTask;
	}

	if(pstTaskThread->enMappedTaskType == MAPPED_TYPE_COMPOSITE_TASK && pstCurrentTask != NULL && pstCurrentTask->pstMTMInfo != NULL)
	{
		bIsModeTransition = TRUE;

		nLen = pstCurrentTask->pstMTMInfo->nNumOfModes;

		if(nLen > 1)
		{
			bIsModeTransition = TRUE;
		}
	}

	return bIsModeTransition;
}

static uem_result suspendCurrentTaskThread(STaskThread *pstTaskThread)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUTaskManager *pstManager = NULL;

	pstManager = pstTaskThread->hManager;

	result = UCThreadMutex_Lock(pstManager->hMutex);
	ERRIFGOTO(result, _EXIT);

	pstTaskThread->enTaskState = TASK_STATE_SUSPEND;

	result = ERR_UEM_NOERROR;

	UCThreadMutex_Unlock(pstManager->hMutex);
_EXIT:
	return result;
}


static uem_result activateTaskThread(STaskThread *pstTaskThread)
{
	uem_result result = ERR_UEM_UNKNOWN;

	while(pstTaskThread->nWaitingThreadNum > 0)
	{
		result = UCThreadEvent_SetEvent(pstTaskThread->hEvent);
		ERRIFGOTO(result, _EXIT);

		UCThread_Yield();
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}



static uem_result activateSingleTaskThread(STaskThread *pstTaskThread, void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nTargetModeId = 0;
	STask *pstParentTask = NULL;
	int nCurModeIndex = 0;

	if(pstTaskThread->enMappedTaskType == MAPPED_TYPE_COMPOSITE_TASK && pstTaskThread->uTargetTask.pstScheduledTasks->nScheduledIndex  == INVALID_SCHEDULE_ID)
	{
		pstTaskThread->enTaskState = TASK_STATE_STOP;
	}
	else if(isModeTransitionTask(pstTaskThread) == TRUE)
	{
		if(pUserData == NULL)
		{
			pstParentTask = pstTaskThread->uTargetTask.pstScheduledTasks->pstParentTask;

			nCurModeIndex = pstParentTask->pstMTMInfo->nCurModeIndex;
			nTargetModeId = pstParentTask->pstMTMInfo->astModeMap[nCurModeIndex].nModeId;
		}
		else
		{
			nTargetModeId = *((int *)pUserData);
		}

		if(pstTaskThread->uTargetTask.pstScheduledTasks->nModeId != nTargetModeId)
		{
			pstTaskThread->enTaskState = TASK_STATE_SUSPEND;
		}
		else
		{
			pstTaskThread->enTaskState = TASK_STATE_RUNNING;
		}
	}
	else
	{
		pstTaskThread->enTaskState = TASK_STATE_RUNNING;
	}

	result = activateTaskThread(pstTaskThread);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result changeCompositeTaskState(STask *pstParentTask, ECPUTaskState enTargetState, HLinkedList hTaskList)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SCompositeTaskUserData stCompositeTaskUserData;
	int nCurModeIndex = 0;
	int nTargetModeId = 0;
	int nLen = 0;
	int nLoop = 0;

	if(pstParentTask->pstMTMInfo != NULL)
	{
		nCurModeIndex = pstParentTask->pstMTMInfo->nCurModeIndex;
		nTargetModeId = pstParentTask->pstMTMInfo->astModeMap[nCurModeIndex].nModeId;
	}
	else
	{
		nTargetModeId = 0;
	}

	stCompositeTaskUserData.nTaskId = pstParentTask->nTaskId;
	stCompositeTaskUserData.fnCallback = checkAndChangeTaskState;
	stCompositeTaskUserData.pUserData = &enTargetState;

	// Change all composite task state
	result = UCDynamicLinkedList_Traverse(hTaskList, traverseCompositeTaskThreads, &stCompositeTaskUserData);
	ERRIFGOTO(result, _EXIT);

	if(enTargetState == TASK_STATE_RUNNING)
	{
		stCompositeTaskUserData.fnCallback = activateSingleTaskThread;
		stCompositeTaskUserData.pUserData = &nTargetModeId;

		result = UCDynamicLinkedList_Traverse(hTaskList, traverseCompositeTaskThreads, &stCompositeTaskUserData);
		ERRIFGOTO(result, _EXIT);
	}
	else if(enTargetState == TASK_STATE_STOPPING)
	{
		// release channel block related to the task to be stopped
		nLen = pstParentTask->pstSubGraph->nNumOfTasks;

		for(nLoop = 0 ; nLoop < nLen ; nLoop++)
		{
			result = UKChannel_SetExitByTaskId(pstParentTask->pstSubGraph->astTasks[nLoop].nTaskId);
			ERRIFGOTO(result, _EXIT);
		}
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result setTaskToStop(STaskThread *pstTaskThread)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nTaskInstanceNumber = 0;

	result = UCDynamicLinkedList_GetLength(pstTaskThread->hThreadList, &nTaskInstanceNumber);
	ERRIFGOTO(result, _EXIT);

	if(nTaskInstanceNumber > 0)
	{
		pstTaskThread->enTaskState = TASK_STATE_STOP;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result traverseAndDestroyThread(IN int nOffset, IN void *pData, IN void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	HThread hThread = NULL;

	hThread = (HThread) pData;

	result = UCThread_Destroy(&(hThread), FALSE, THREAD_DESTROY_TIMEOUT);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result callCompositeTaskWrapupFunctions(STask *pstParentTask)
{
	uem_result result = ERR_UEM_UNKNOWN;
	HStack hStack = NULL;

	result = UCDynamicStack_Create(&hStack);
	ERRIFGOTO(result, _EXIT);
	// Stack with SModeMap *, current index astRelatedChildTasks

	if(pstParentTask != NULL)
	{
		IFVARERRASSIGNGOTO(pstParentTask->pstMTMInfo, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

		result = callCompositeTaskInitOrWrapupFunctions(pstParentTask, FALSE, hStack);
		ERRIFGOTO(result, _EXIT);
	}
	else
	{
		result = UKTask_TraverseAllTasks(traverseTaskAndCallWrapupFunctions, NULL);
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


static uem_result destroyTaskThreads(STaskThread *pstTaskThread)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nTaskInstanceNumber = 0;
	int nLoop = 0;

	result = UCDynamicLinkedList_GetLength(pstTaskThread->hThreadList, &nTaskInstanceNumber);
	ERRIFGOTO(result, _EXIT);

	if(nTaskInstanceNumber > 0)
	{
		result = UCDynamicLinkedList_Traverse(pstTaskThread->hThreadList, traverseAndDestroyThread, NULL);
		ERRIFGOTO(result, _EXIT);

		for(nLoop = 0 ; nLoop < nTaskInstanceNumber ; nLoop++)
		{
			result = UCDynamicLinkedList_Remove(pstTaskThread->hThreadList, LINKED_LIST_OFFSET_FIRST, 0);
			ERRIFGOTO(result, _EXIT);
		}
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}





static uem_result stopSingleTaskThread(STaskThread *pstTaskThread, void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nTaskId = *((int *) pUserData);

	if(pstTaskThread->enTaskState == TASK_STATE_RUNNING ||
		pstTaskThread->enTaskState == TASK_STATE_STOPPING ||
		pstTaskThread->enTaskState == TASK_STATE_SUSPEND)
	{
		result = setTaskToStop(pstTaskThread);
		ERRIFGOTO(result, _EXIT);

		// release task if task is suspended
		if(pstTaskThread->enTaskState == TASK_STATE_SUSPEND)
		{
			result = activateTaskThread(pstTaskThread);
			ERRIFGOTO(result, _EXIT);
		}
		else // pstTaskThread->enTaskState == TASK_STATE_RUNNING, TASK_STATE_STOPPING
		{
			// release channel block related to the task to be stopped
			result = UKChannel_SetExitByTaskId(nTaskId);
			ERRIFGOTO(result, _EXIT);
		}

		result = destroyTaskThreads(pstTaskThread);
		ERRIFGOTO(result, _EXIT);

		// all tasks are terminated, so clear the exit flag of adjacent channels
		result = UKChannel_ClearExitByTaskId(nTaskId);
		ERRIFGOTO(result, _EXIT);

		pstTaskThread->nFinishedThreadNum = 0;

		pstTaskThread->nSeqId++;
	}
	else
	{
		UEMASSIGNGOTO(result, ERR_UEM_ALREADY_DONE, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result stopTaskThread(STaskThread *pstTaskThread, void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;

	if(pstTaskThread->enTaskState == TASK_STATE_RUNNING ||
		pstTaskThread->enTaskState == TASK_STATE_STOPPING ||
		pstTaskThread->enTaskState == TASK_STATE_SUSPEND)
	{
		result = setTaskToStop(pstTaskThread);
		ERRIFGOTO(result, _EXIT);

		// release task if task is suspended
		if(pstTaskThread->enTaskState == TASK_STATE_SUSPEND)
		{
			result = activateTaskThread(pstTaskThread);
			ERRIFGOTO(result, _EXIT);
		}

		pstTaskThread->nFinishedThreadNum = 0;
		pstTaskThread->nSeqId++;
	}
	else
	{
		UEMASSIGNGOTO(result, ERR_UEM_ALREADY_DONE, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static uem_result joinTaskThread(STaskThread *pstTaskThread, void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = destroyTaskThreads(pstTaskThread);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}



static uem_result createThread(STaskThread *pstTaskThread, int nMappedCPU, FnNativeThread fnThreadRoutine, int nTaskFunctionIndex)
{
	uem_result result = ERR_UEM_UNKNOWN;
	HThread hThread = NULL;
	STaskThreadData *pstThreadData = NULL;

	pstThreadData = UC_malloc(sizeof(STaskThreadData));
	ERRMEMGOTO(pstThreadData, result, _EXIT);

	pstThreadData->pstTaskThread = pstTaskThread;
	pstThreadData->nCurSeqId = pstTaskThread->nSeqId;
	pstThreadData->nTaskFunctionIndex = nTaskFunctionIndex;

	result = UCThread_Create(fnThreadRoutine, pstThreadData, &hThread);
	ERRIFGOTO(result, _EXIT);

	// pstThreadData is already passed to a new thread
	pstThreadData = NULL;

	if(nMappedCPU != MAPPING_NOT_SPECIFIED)
	{
		result = UCThread_SetMappedCPU(hThread, nMappedCPU);
		ERRIFGOTO(result, _EXIT);
	}

	result = UCDynamicLinkedList_Add(pstTaskThread->hThreadList, LINKED_LIST_OFFSET_LAST, 0, (void *) hThread);
	ERRIFGOTO(result, _EXIT);

	hThread = NULL;

	result = ERR_UEM_NOERROR;
_EXIT:
	if(hThread != NULL)
	{
		UCThread_Destroy(&hThread, FALSE, THREAD_DESTROY_TIMEOUT);
	}
	SAFEMEMFREE(pstThreadData);
	return result;
}

static uem_result checkTaskIsSuspended(STaskThread *pstTaskThread, void *pUserData)
{
	uem_bool bIsSuspended = *((uem_bool *)pUserData);
	uem_bool *pbIsSuspended = (uem_bool *)pUserData;

	if(bIsSuspended == TRUE && pstTaskThread->enTaskState == TASK_STATE_RUNNING)
	{
		bIsSuspended = FALSE;
	}

	*pbIsSuspended = bIsSuspended;

	return ERR_UEM_NOERROR;
}

static uem_result findTaskFromTaskId(IN int nOffset, IN void *pData, IN void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _STaskSearchUserData *pstUserData = pUserData;
	STaskThread *pstTaskThread = (STaskThread *) pData;

	if(pstTaskThread->enMappedTaskType == MAPPED_TYPE_GENERAL_TASK && pstTaskThread->uTargetTask.pstTask->nTaskId == pstUserData->nTaskId)
	{
		pstUserData->pstTargetThread = pstTaskThread;
		UEMASSIGNGOTO(result, ERR_UEM_FOUND_DATA, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result findParentTaskFromTaskId(IN int nOffset, IN void *pData, IN void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SParentTaskSearchUserData *pstUserData = pUserData;
	STaskThread *pstTaskThread = (STaskThread *) pData;

	if(pstTaskThread->enMappedTaskType == MAPPED_TYPE_COMPOSITE_TASK &&
		pstTaskThread->uTargetTask.pstScheduledTasks->pstParentTask->nTaskId == pstUserData->nTaskId)
	{
		pstUserData->pstTargetParentTask = pstTaskThread->uTargetTask.pstScheduledTasks->pstParentTask;
		UEMASSIGNGOTO(result, ERR_UEM_FOUND_DATA, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result traverseAndFindTaskThread(HLinkedList hTaskList, int nTaskId, OUT uem_bool *pbIsCompositeTask, OUT STask **ppstTask, OUT STaskThread **ppstTaskThread)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _STaskSearchUserData stUserData;
	struct _SParentTaskSearchUserData stParentUserData;
	uem_bool bIsCompositeTask = FALSE;

	stUserData.nTaskId = nTaskId;
	stUserData.pstTargetThread = NULL;

	result = UCDynamicLinkedList_Traverse(hTaskList, findTaskFromTaskId, &stUserData);
	ERRIFGOTO(result, _EXIT);
	if(result == ERR_UEM_NOERROR)
	{
		stParentUserData.nTaskId = nTaskId;
		stParentUserData.pstTargetParentTask = NULL;

		// Not found, it might be composite task
		result = UCDynamicLinkedList_Traverse(hTaskList, findParentTaskFromTaskId, &stParentUserData);
		ERRIFGOTO(result, _EXIT);
		if(result == ERR_UEM_NOERROR) // No task is registered
		{
			UEMASSIGNGOTO(result, ERR_UEM_NO_DATA, _EXIT);
		}
		else // result == ERR_UEM_FOUND_DATA (Static-scheduled composite task)
		{
			bIsCompositeTask = TRUE;
			*ppstTask = stParentUserData.pstTargetParentTask;
			*ppstTaskThread = NULL;
		}
	}
	else // result == ERR_UEM_FOUND_DATA (General task)
	{
		// bIsCompositeTask is FALSE
		*ppstTaskThread = stUserData.pstTargetThread;
		*ppstTask = NULL;
	}

	*pbIsCompositeTask = bIsCompositeTask;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result traverseChildTaskThreads(IN int nOffset, IN void *pData, IN void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SChildTaskAccessUserData *pstUserData = pUserData;
	STaskThread *pstTaskThread = (STaskThread *) pData;
	STask *pstChildTaskCandidate = NULL;
	int nModeIndex = 0;
	int nModeId = 0;

	switch(pstTaskThread->enMappedTaskType)
	{
	case MAPPED_TYPE_COMPOSITE_TASK:
		pstChildTaskCandidate = pstTaskThread->uTargetTask.pstScheduledTasks->pstParentTask;

		if(pstChildTaskCandidate->pstMTMInfo != NULL)
		{
			nModeIndex = pstChildTaskCandidate->pstMTMInfo->nCurModeIndex;
			nModeId = pstChildTaskCandidate->pstMTMInfo->astModeMap[nModeIndex].nModeId;
		}
		else
		{
			nModeId = 0;
		}

		// if mode ID is different, skip it
		if(pstTaskThread->uTargetTask.pstScheduledTasks->nModeId != nModeId)
		{
			UEMASSIGNGOTO(result, ERR_UEM_NOERROR, _EXIT);
		}
		break;
	case MAPPED_TYPE_GENERAL_TASK:
		pstChildTaskCandidate = pstTaskThread->uTargetTask.pstTask;

		// If a child task is a control-driven task, skip the task (control-driven tasks are only controlled by the tasks on the same task-graph)
		if(pstChildTaskCandidate->enRunCondition == RUN_CONDITION_CONTROL_DRIVEN)
		{
			UEMASSIGNGOTO(result, ERR_UEM_NOERROR, _EXIT);
		}
		break;
	default: // error
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
		break;
	}

	if(checkIsChildTask(pstUserData->nParentTaskId, pstChildTaskCandidate) == TRUE)
	{
		pstUserData->nMatchedTaskNum++;

		result = pstUserData->fnCallback(pstTaskThread, pstUserData->pUserData);
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result changeChildTaskState(STask *pstParentTask, ECPUTaskState enTargetState, HLinkedList hTaskList)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SChildTaskAccessUserData stChildTaskAccessUserData;

	stChildTaskAccessUserData.nParentTaskId = pstParentTask->nTaskId;
	stChildTaskAccessUserData.nMatchedTaskNum = 0;
	stChildTaskAccessUserData.fnCallback = checkAndChangeTaskState;
	stChildTaskAccessUserData.pUserData = &enTargetState;
	result = UCDynamicLinkedList_Traverse(hTaskList, traverseChildTaskThreads, &stChildTaskAccessUserData);
	ERRIFGOTO(result, _EXIT);

	if(stChildTaskAccessUserData.nMatchedTaskNum == 0)
	{
		ERRASSIGNGOTO(result, ERR_UEM_NO_DATA, _EXIT);
	}

	if(enTargetState == TASK_STATE_RUNNING)
	{
		stChildTaskAccessUserData.fnCallback = activateSingleTaskThread;
		stChildTaskAccessUserData.pUserData = NULL;

		result = UCDynamicLinkedList_Traverse(hTaskList, traverseCompositeTaskThreads, &stChildTaskAccessUserData);
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result updateTaskState(HLinkedList hTaskList, int nTaskId, ECPUTaskState enState)
{
	uem_result result = ERR_UEM_UNKNOWN;
	uem_bool bIsCompositeTask = FALSE;
	STask *pstTargetParentTask = NULL;
	STaskThread *pstTargetThread = NULL;

	result = traverseAndFindTaskThread(hTaskList, nTaskId,
										&bIsCompositeTask, &pstTargetParentTask, &pstTargetThread);
	if(result != ERR_UEM_NO_DATA) // ERR_UEM_NO_DATA is handled differently
	{
		ERRIFGOTO(result, _EXIT);
	}

	if(bIsCompositeTask == TRUE)
	{
		result = changeCompositeTaskState(pstTargetParentTask, enState, hTaskList);
		ERRIFGOTO(result, _EXIT);
	}
	else if(result == ERR_UEM_NO_DATA) // Task thread is not found, check it has child tasks
	{
		result = UKTask_GetTaskFromTaskId(nTaskId, &pstTargetParentTask);
		ERRIFGOTO(result, _EXIT);

		if(pstTargetParentTask->pstSubGraph != NULL)
		{
			result = changeChildTaskState(pstTargetParentTask, enState, hTaskList);
			ERRIFGOTO(result, _EXIT);
		}
		else
		{
			ERRASSIGNGOTO(result, ERR_UEM_NO_DATA, _EXIT);
		}
	}
	else
	{
		result = checkAndChangeTaskState(pstTargetThread, &enState);
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static uem_result checkAndSwitchToNextModeInTaskLock(STaskThread *pstTaskThread, STask *pstParentTask)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUTaskManager *pstManager = NULL;
	struct _SCompositeTaskUserData stUserData;
	uem_bool bIsSuspended = TRUE;

	pstManager = pstTaskThread->hManager;

	result = UCThreadMutex_Lock(pstManager->hMutex);
	ERRIFGOTO(result, _EXIT);

	stUserData.fnCallback = checkTaskIsSuspended;
	stUserData.pUserData = &bIsSuspended;
	stUserData.nTaskId = pstParentTask->nTaskId;

	result = UCDynamicLinkedList_Traverse(pstManager->uControlDrivenTaskList.hTaskList,
			traverseCompositeTaskThreads, &stUserData);
	ERRIFGOTO(result, _EXIT_LOCK);

	if(bIsSuspended == FALSE)
	{
		result = UCDynamicLinkedList_Traverse(pstManager->uDataAndTimeDrivenTaskList.hTaskList,
				traverseCompositeTaskThreads, &stUserData);
		ERRIFGOTO(result, _EXIT_LOCK);

		if(bIsSuspended == TRUE)
		{
			pstParentTask->pstMTMInfo->nCurModeIndex = pstParentTask->pstMTMInfo->nNextModeIndex;

			result = updateTaskState(pstManager->uDataAndTimeDrivenTaskList.hTaskList, pstParentTask->nTaskId, TASK_STATE_RUNNING);
			ERRIFGOTO(result, _EXIT_LOCK);
		}
	}
	else // task pstParentTask->nTaskId of all composite tasks are suspended;
	{
		pstParentTask->pstMTMInfo->nCurModeIndex = pstParentTask->pstMTMInfo->nNextModeIndex;

		result = updateTaskState(pstManager->uControlDrivenTaskList.hTaskList, pstParentTask->nTaskId, TASK_STATE_RUNNING);
		ERRIFGOTO(result, _EXIT_LOCK);
	}

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
	UCThreadMutex_Unlock(pstManager->hMutex);
_EXIT:
	return result;
}

static uem_result handleCompositeTaskModeTransition(STaskThread *pstTaskThread, STask *pstCurrentTask)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = UCThreadMutex_Lock(pstCurrentTask->hMutex);
	ERRIFGOTO(result, _EXIT);

	// Mode transition is happened (suspend current thread and wait for other threads to be suspended)
	if(pstCurrentTask->pstMTMInfo->nCurModeIndex != pstCurrentTask->pstMTMInfo->nNextModeIndex)
	{
		// change task state to suspend
		result = suspendCurrentTaskThread(pstTaskThread);
		ERRIFGOTO(result, _EXIT_LOCK);

		// check all composite task threads are suspended, and resume tasks with new mode
		result = checkAndSwitchToNextModeInTaskLock(pstTaskThread, pstCurrentTask);
		ERRIFGOTO(result, _EXIT_LOCK);
	}

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
	UCThreadMutex_Unlock(pstCurrentTask->hMutex);
_EXIT:
	return result;
}



static uem_result checkModeTransitionHappened(STaskThread *pstTaskThread, STask *pstCurrentTask)
{
	uem_result result = ERR_UEM_UNKNOWN;

	if(isModeTransitionTask(pstTaskThread) == TRUE)
	{
		result = handleCompositeTaskModeTransition(pstTaskThread, pstCurrentTask);
		ERRIFGOTO(result, _EXIT);
	}
	else if(pstTaskThread->enMappedTaskType == MAPPED_TYPE_GENERAL_TASK)
	{
		// MTM check is only performed on two-level
		if(pstCurrentTask->pstParentGraph->pstParentTask != NULL)
		{
			if(pstCurrentTask->pstParentGraph->pstParentTask->pstMTMInfo != NULL)
			{

			}
		}
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}



static uem_result handleTaskMainRoutine(STaskThread *pstTaskThread, FnUemTaskGo fnGo, int nCurSeqId)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstCurrentTask = NULL;
	long long llNextTime = 0;
	int nMaxRunCount = 0;
	int nRunCount = 0;
	ERunCondition enRunCondition;
	uem_bool bIsTaskGraphSourceTask = FALSE;

	if(pstTaskThread->enMappedTaskType == MAPPED_TYPE_COMPOSITE_TASK)
	{
		pstCurrentTask = pstTaskThread->uTargetTask.pstScheduledTasks->pstParentTask;
		if(pstCurrentTask == NULL)
		{
			enRunCondition = RUN_CONDITION_DATA_DRIVEN;
		}
		else
		{
			enRunCondition = pstCurrentTask->enRunCondition;
		}
	}
	else // MAPPED_TYPE_GENERAL_TASK
	{
		pstCurrentTask = pstTaskThread->uTargetTask.pstTask;
		enRunCondition = pstCurrentTask->enRunCondition;
		bIsTaskGraphSourceTask = UKChannel_IsTaskSourceTask(pstCurrentTask->nTaskId);
	}

	result = waitRunSignal(pstTaskThread, TRUE, &llNextTime, &nMaxRunCount);
	ERRIFGOTO(result, _EXIT);

	// if nSeqId is changed, it means this thread is detached or stopped from the CPU task manager.
	// So, end this thread
	while(nCurSeqId == pstTaskThread->nSeqId && pstTaskThread->enTaskState != TASK_STATE_STOP)
	{
		switch(pstTaskThread->enTaskState)
		{
		case TASK_STATE_RUNNING:
			switch(enRunCondition)
			{
			case RUN_CONDITION_TIME_DRIVEN:
				result = handleTimeDrivenTask(pstTaskThread, pstCurrentTask, fnGo, &llNextTime, &nRunCount, &nMaxRunCount);
				ERRIFGOTO(result, _EXIT);
				break;
			case RUN_CONDITION_DATA_DRIVEN:
				fnGo(pstCurrentTask->nTaskId);
				break;
			case RUN_CONDITION_CONTROL_DRIVEN: // run once for control-driven leaf task
				fnGo(pstCurrentTask->nTaskId);
				if(isModeTransitionTask(pstTaskThread) == FALSE)
				{
					UEMASSIGNGOTO(result, ERR_UEM_NOERROR, _EXIT);
				}
				break;
			default:
				ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
				break;
			}
			result = checkModeTransitionHappened(pstTaskThread, pstCurrentTask);
			ERRIFGOTO(result, _EXIT);
			break;
		case TASK_STATE_STOPPING:
			// Just finish time-driven task first, other tasks are still executing the tasks to finish remaining jobs
			if(bIsTaskGraphSourceTask == TRUE || pstTaskThread->enMappedTaskType == MAPPED_TYPE_COMPOSITE_TASK)
			{
				UEMASSIGNGOTO(result, ERR_UEM_NOERROR, _EXIT);
			}
			else // still execute the tasks for the remaining tasks
			{
				fnGo(pstCurrentTask->nTaskId);
			}
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
	STaskThreadData *pstThreadData = NULL;
	STaskThread *pstTaskThread = NULL;
	STask *pstCurrentTask = NULL;
	int nIndex = 0;

	pstThreadData = (STaskThreadData *) pData;

	pstTaskThread = pstThreadData->pstTaskThread;
	pstCurrentTask = pstTaskThread->uTargetTask.pstTask;
	nIndex = pstThreadData->nTaskFunctionIndex;

	result = handleTaskMainRoutine(pstTaskThread, pstCurrentTask->astTaskFunctions[nIndex].fnGo, pstThreadData->nCurSeqId);
	ERRIFGOTO(result, _EXIT);

_EXIT:
	if(pstThreadData->nCurSeqId == pstTaskThread->nSeqId && pstCurrentTask != NULL)
	{
		pstCurrentTask->astTaskFunctions[nIndex].fnWrapup();
	}

	if(pstTaskThread != NULL)
	{
		UCThreadMutex_Lock(pstTaskThread->hMutex);
		pstTaskThread->nFinishedThreadNum++;
		UCThreadMutex_Unlock(pstTaskThread->hMutex);
	}

	SAFEMEMFREE(pstThreadData);
	return NULL;
}


static void *scheduledTaskThreadRoutine(void *pData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STaskThreadData *pstThreadData = NULL;
	STaskThread *pstTaskThread = NULL;

	int nScheduleIndex = 0;
	SScheduledTasks *pstScheduledTasks = NULL;

	pstThreadData = (STaskThreadData *) pData;

	pstTaskThread = pstThreadData->pstTaskThread;
	// pstThreadData->nCurSeqId

	pstScheduledTasks = pstTaskThread->uTargetTask.pstScheduledTasks;
	nScheduleIndex = pstScheduledTasks->nScheduledIndex;

	if(nScheduleIndex != INVALID_SCHEDULE_ID)
	{
		result = handleTaskMainRoutine(pstTaskThread, pstScheduledTasks->astScheduleList[nScheduleIndex].fnCompositeGo,
										pstThreadData->nCurSeqId);
		ERRIFGOTO(result, _EXIT);
	}
	else
	{
		result = UCThreadMutex_Lock(pstTaskThread->hMutex);
		ERRIFGOTO(result, _EXIT);

		pstTaskThread->nWaitingThreadNum--;

		result = UCThreadMutex_Unlock(pstTaskThread->hMutex);
		ERRIFGOTO(result, _EXIT);
	}

_EXIT:
	if(pstTaskThread != NULL)
	{
		UCThreadMutex_Lock(pstTaskThread->hMutex);
		pstTaskThread->nFinishedThreadNum++;
		UCThreadMutex_Unlock(pstTaskThread->hMutex);
	}
	SAFEMEMFREE(pstThreadData);
	return NULL;
}


static uem_result createCompositeTaskThread(STaskThread *pstTaskThread)
{
	uem_result result = ERR_UEM_UNKNOWN;
	void *pCPUId = 0;
	int nMappedCPUNumber = 0;
	STask *pstParentTask = NULL;
	//SScheduledTasks *pstTasks = NULL;
	HStack hStack = NULL;
	int nMappedCPU = 0;

	result = UCDynamicLinkedList_GetLength(pstTaskThread->uMappedCPUList.hMappedCPUList, &nMappedCPUNumber);
	ERRIFGOTO(result, _EXIT);

	// call TASK_INIT for the nSeqInMode is 0 which is a representative composite task needed to call multiple Task INIT functions
	if(pstTaskThread->uTargetTask.pstScheduledTasks->nSeqInMode == 0)
	{
		result = UCDynamicStack_Create(&hStack);
		ERRIFGOTO(result, _EXIT);
		// Stack with SModeMap *, current index astRelatedChildTasks

		pstParentTask = pstTaskThread->uTargetTask.pstScheduledTasks->pstParentTask;
		if(pstParentTask != NULL)
		{
			IFVARERRASSIGNGOTO(pstParentTask->pstMTMInfo, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

			result = callCompositeTaskInitOrWrapupFunctions(pstParentTask, TRUE, hStack);
			ERRIFGOTO(result, _EXIT);
		}
		else // elements of composite tasks are located at the top graph, so initialize all time-driven/data-driven tasks
		{
			result = UKTask_TraverseAllTasks(traverseTaskAndCallInitFunctions, NULL);
			ERRIFGOTO(result, _EXIT);
		}
	}

	if(nMappedCPUNumber > 0)
	{
		// Composite task only mapped to a single thread
		result = UCDynamicLinkedList_Get(pstTaskThread->uMappedCPUList.hMappedCPUList, LINKED_LIST_OFFSET_FIRST, 0, (void **) &pCPUId);
		ERRIFGOTO(result, _EXIT);

#if SIZEOF_VOID_P == 8
		nMappedCPU = (int) ((long long) pCPUId);
#else
		nMappedCPU = (int) pData;
#endif
		result = createThread(pstTaskThread, nMappedCPU, scheduledTaskThreadRoutine, 0);
		ERRIFGOTO(result, _EXIT);
	}
	else // nMappedCPUNumber == 0
	{
		result = createThread(pstTaskThread, MAPPING_NOT_SPECIFIED, scheduledTaskThreadRoutine, 0);
		ERRIFGOTO(result, _EXIT);
	}

	pstTaskThread->nWaitingThreadNum++;

	result = ERR_UEM_NOERROR;
_EXIT:
	if(hStack != NULL)
	{
		UCDynamicStack_Destroy(&hStack, NULL, NULL);
	}
	return result;
}


static uem_result traverseAndCreateEachThread(IN int nOffset, IN void *pData, IN void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STaskThread *pstTaskThread = NULL;
	int nMappedCPU = 0;

	pstTaskThread = (STaskThread *) pUserData;
#if SIZEOF_VOID_P == 8
	nMappedCPU = (int) ((long long) pData);
#else
	nMappedCPU = (int) pData;
#endif
	result = createThread(pstTaskThread, nMappedCPU, taskThreadRoutine, nOffset);
	ERRIFGOTO(result, _EXIT);

	pstTaskThread->nWaitingThreadNum++;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result createMultipleThreads(HLinkedList hMappedCPUList, STaskThread *pstTaskThread)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nMappedCPUNumber = 0;
	STask *pstTask = NULL;
	int nLoop = 0;

	result = UCDynamicLinkedList_GetLength(hMappedCPUList, &nMappedCPUNumber);
	ERRIFGOTO(result, _EXIT);

	// call TASK_INIT
	pstTask = pstTaskThread->uTargetTask.pstTask;
	for(nLoop = 0; nLoop < pstTask->nTaskFunctionSetNum ; nLoop++)
	{
		pstTask->astTaskFunctions[nLoop].fnInit(pstTask->nTaskId);
	}

	if(nMappedCPUNumber > 0)
	{
		result = UCDynamicLinkedList_Traverse(hMappedCPUList, traverseAndCreateEachThread, pstTaskThread);
		ERRIFGOTO(result, _EXIT);
	}
	else // not mapped to specific task
	{
		result = createThread(pstTaskThread, MAPPING_NOT_SPECIFIED, taskThreadRoutine, 0);
		ERRIFGOTO(result, _EXIT);

		pstTaskThread->nWaitingThreadNum++;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result runSingleTaskThread(STaskThread *pstTaskThread, void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;

	// if task state is stopping state Stop the task and run the task
	if(pstTaskThread->enTaskState == TASK_STATE_STOPPING)
	{
		result = stopSingleTaskThread(pstTaskThread, pUserData);
		ERRIFGOTO(result, _EXIT);
	}

	result = checkTaskThreadState(pstTaskThread->enTaskState, TASK_STATE_RUNNING);
	ERRIFGOTO(result, _EXIT);

	if(pstTaskThread->enMappedTaskType == MAPPED_TYPE_COMPOSITE_TASK)
	{
		int nLoop = 0;
		STask *pstParentTask = NULL;
		pstParentTask = pstTaskThread->uTargetTask.pstScheduledTasks->pstParentTask;

		if(pstParentTask->nThroughputConstraint == 0) // initial setting
		{
			pstParentTask->nThroughputConstraint = pstTaskThread->uTargetTask.pstScheduledTasks->astScheduleList[0].nThroughputConstraint;
		}

		for(nLoop = 0; nLoop < pstTaskThread->uTargetTask.pstScheduledTasks->nScheduleNum ; nLoop++)
		{
			if(pstTaskThread->uTargetTask.pstScheduledTasks->astScheduleList[nLoop].nThroughputConstraint ==
				pstParentTask->nThroughputConstraint)
			{
				pstTaskThread->uTargetTask.pstScheduledTasks->nScheduledIndex = nLoop;
				break;
			}
		}

		// skip if there is no matching throughput constraint
		if(nLoop == pstTaskThread->uTargetTask.pstScheduledTasks->nScheduleNum)
		{
			pstTaskThread->uTargetTask.pstScheduledTasks->nScheduledIndex = INVALID_SCHEDULE_ID;
		}

		result = createCompositeTaskThread(pstTaskThread);
		ERRIFGOTO(result, _EXIT);
	}
	else
	{
		result = createMultipleThreads(pstTaskThread->uMappedCPUList.hMappedCPUList, pstTaskThread);
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKCPUTaskManager_SuspendTask(HCPUTaskManager hCPUTaskManager, int nTaskId)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUTaskManager *pstManager = NULL;
#ifdef ARGUMENT_CHECK
	if (IS_VALID_HANDLE(hCPUTaskManager, ID_UEM_CPU_TASK_MANAGER) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}

	if(nTaskId < 0) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
#endif
	pstManager = hCPUTaskManager;

	result = UCThreadMutex_Lock(pstManager->hMutex);
	ERRIFGOTO(result, _EXIT);

	result = updateTaskState(pstManager->uControlDrivenTaskList.hTaskList, nTaskId, TASK_STATE_SUSPEND);
	ERRIFGOTO(result, _EXIT_LOCK);

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
	UCThreadMutex_Unlock(pstManager->hMutex);
_EXIT:
	return result;
}


static uem_result traverseAndCreateControlTasks(IN int nOffset, IN void *pData, IN void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STaskThread *pstTaskThread = (STaskThread *) pData;
	int nCreatedThreadNum = 0;

	result = UCDynamicLinkedList_GetLength(pstTaskThread->hThreadList, &nCreatedThreadNum);
	ERRIFGOTO(result, _EXIT);

	if(nCreatedThreadNum == 0 && pstTaskThread->enMappedTaskType == MAPPED_TYPE_GENERAL_TASK &&
		pstTaskThread->uTargetTask.pstTask->enType == TASK_TYPE_CONTROL)
	{
		result = createMultipleThreads(pstTaskThread->uMappedCPUList.hMappedCPUList, pstTaskThread);
		ERRIFGOTO(result, _EXIT);

		pstTaskThread->enTaskState = TASK_STATE_RUNNING;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result traverseAndCreateComputationalTasks(IN int nOffset, IN void *pData, IN void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STaskThread *pstTaskThread = (STaskThread *) pData;
	int nCreatedThreadNum = 0;
	ERunCondition enRunCondition;
	STask *pstParentTask = NULL;
	int nCurModeIndex;

	result = UCDynamicLinkedList_GetLength(pstTaskThread->hThreadList, &nCreatedThreadNum);
	ERRIFGOTO(result, _EXIT);

	if(nCreatedThreadNum == 0)
	{
		if(pstTaskThread->enMappedTaskType == MAPPED_TYPE_GENERAL_TASK &&
		(pstTaskThread->uTargetTask.pstTask->enType == TASK_TYPE_COMPUTATIONAL ||
			pstTaskThread->uTargetTask.pstTask->enType == TASK_TYPE_LOOP) &&
		pstTaskThread->uTargetTask.pstTask->enRunCondition != RUN_CONDITION_CONTROL_DRIVEN)
		{
			result = createMultipleThreads(pstTaskThread->uMappedCPUList.hMappedCPUList, pstTaskThread);
			ERRIFGOTO(result, _EXIT);

			pstTaskThread->enTaskState = TASK_STATE_RUNNING;
		}
		else if(pstTaskThread->enMappedTaskType == MAPPED_TYPE_COMPOSITE_TASK)
		{
			int nLoop = 0;
			pstParentTask = pstTaskThread->uTargetTask.pstScheduledTasks->pstParentTask;

			if(pstParentTask->nThroughputConstraint == 0) // initial setting
			{
				pstParentTask->nThroughputConstraint = pstTaskThread->uTargetTask.pstScheduledTasks->astScheduleList[0].nThroughputConstraint;
			}

			for(nLoop = 0; nLoop < pstTaskThread->uTargetTask.pstScheduledTasks->nScheduleNum ; nLoop++)
			{
				if(pstTaskThread->uTargetTask.pstScheduledTasks->astScheduleList[nLoop].nThroughputConstraint ==
					pstParentTask->nThroughputConstraint)
				{
					pstTaskThread->uTargetTask.pstScheduledTasks->nScheduledIndex = nLoop;
					break;
				}
			}

			// skip if there is no matching throughput constraint
			if(nLoop == pstTaskThread->uTargetTask.pstScheduledTasks->nScheduleNum)
			{
				pstTaskThread->uTargetTask.pstScheduledTasks->nScheduledIndex = INVALID_SCHEDULE_ID;
			}

			if(pstParentTask != NULL)
			{
				enRunCondition = pstParentTask->enRunCondition;
			}
			else
			{
				enRunCondition = RUN_CONDITION_DATA_DRIVEN;
			}

			if(enRunCondition != RUN_CONDITION_CONTROL_DRIVEN)
			{
				if(isModeTransitionTask(pstTaskThread) == TRUE)
				{
					nCurModeIndex = pstParentTask->pstMTMInfo->nCurModeIndex;

					result = createCompositeTaskThread(pstTaskThread);
					ERRIFGOTO(result, _EXIT);

					if(pstParentTask->pstMTMInfo->astModeMap[nCurModeIndex].nModeId ==
						pstTaskThread->uTargetTask.pstScheduledTasks->nModeId)
					{
						pstTaskThread->enTaskState = TASK_STATE_RUNNING;
					}
					else
					{
						pstTaskThread->enTaskState = TASK_STATE_SUSPEND;
					}
				}
				else
				{
					result = createCompositeTaskThread(pstTaskThread);
					ERRIFGOTO(result, _EXIT);

					pstTaskThread->enTaskState = TASK_STATE_RUNNING;
				}
			}
		}
		else
		{
			// MAPPED_TYPE_GENERAL_TASK && TASK_TYPE_COMPOSITE => impossible case

			// MAPPED_TYPE_COMPOSITE_TASK && RUN_CONDITION_CONTROL_DRIVEN => no need to be executed at first

			// MAPPED_TYPE_GENERAL_TASK && TASK_TYPE_CONTROL => already created

			// do nothing for RUN_CONDITION_CONTROL_DRIVEN tasks
		}
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result traverseAndActivateTaskThreads(IN int nOffset, IN void *pData, IN void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STaskThread *pstTaskThread = (STaskThread *) pData;

	result = activateTaskThread(pstTaskThread);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKCPUTaskManager_RunRegisteredTasks(HCPUTaskManager hCPUTaskManager)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUTaskManager *pstManager = NULL;

#ifdef ARGUMENT_CHECK
	if (IS_VALID_HANDLE(hCPUTaskManager, ID_UEM_CPU_TASK_MANAGER) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}
#endif
	pstManager = hCPUTaskManager;

	result = UCDynamicLinkedList_Traverse(pstManager->uDataAndTimeDrivenTaskList.hTaskList, traverseAndCreateControlTasks, NULL);
	ERRIFGOTO(result, _EXIT);

	result = UCDynamicLinkedList_Traverse(pstManager->uDataAndTimeDrivenTaskList.hTaskList, traverseAndCreateComputationalTasks, NULL);
	ERRIFGOTO(result, _EXIT);

	// Send event signal to execute
	result = UCDynamicLinkedList_Traverse(pstManager->uDataAndTimeDrivenTaskList.hTaskList, traverseAndActivateTaskThreads, NULL);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result traverseAndDestroyAllThreads(IN int nOffset, IN void *pData, IN void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STaskThread *pstTaskThread = (STaskThread *) pData;

	result = destroyTaskThreads(pstTaskThread);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result traverseAndSetTaskToStop(IN int nOffset, IN void *pData, IN void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STaskThread *pstTaskThread = (STaskThread *) pData;

	result = setTaskToStop(pstTaskThread);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKCPUTaskManager_StopAllTasks(HCPUTaskManager hCPUTaskManager)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUTaskManager *pstManager = NULL;

#ifdef ARGUMENT_CHECK
	if (IS_VALID_HANDLE(hCPUTaskManager, ID_UEM_CPU_TASK_MANAGER) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}
#endif

	pstManager = hCPUTaskManager;

	result = UCDynamicLinkedList_Traverse(pstManager->uDataAndTimeDrivenTaskList.hTaskList, traverseAndSetTaskToStop, NULL);
	ERRIFGOTO(result, _EXIT);

	result = UCDynamicLinkedList_Traverse(pstManager->uControlDrivenTaskList.hTaskList, traverseAndSetTaskToStop, NULL);
	ERRIFGOTO(result, _EXIT);

	result = UKChannel_SetExit();
	ERRIFGOTO(result, _EXIT);

	result = UCDynamicLinkedList_Traverse(pstManager->uDataAndTimeDrivenTaskList.hTaskList, traverseAndDestroyAllThreads, NULL);
	ERRIFGOTO(result, _EXIT);

	result = UCDynamicLinkedList_Traverse(pstManager->uControlDrivenTaskList.hTaskList, traverseAndDestroyAllThreads, NULL);
	ERRIFGOTO(result, _EXIT);


	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}



static uem_result stopCompositeTaskThreads(STask *pstParentTask, HLinkedList hTaskList)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SCompositeTaskUserData stCompositeTaskUserData;
	int nLen = 0;
	int nLoop = 0;

	stCompositeTaskUserData.fnCallback = stopTaskThread;
	stCompositeTaskUserData.pUserData = NULL;

	// Change all composite task state
	result = UCDynamicLinkedList_Traverse(hTaskList,
			traverseCompositeTaskThreads, &stCompositeTaskUserData);
	ERRIFGOTO(result, _EXIT);

	// release channel block related to the task to be stopped
	nLen = pstParentTask->pstSubGraph->nNumOfTasks;

	for(nLoop = 0 ; nLoop < nLen ; nLoop++)
	{
		result = UKChannel_SetExitByTaskId(pstParentTask->pstSubGraph->astTasks[nLoop].nTaskId);
		ERRIFGOTO(result, _EXIT);
	}

	stCompositeTaskUserData.fnCallback = joinTaskThread;
	stCompositeTaskUserData.pUserData = NULL;

	// Change all composite task state
	result = UCDynamicLinkedList_Traverse(hTaskList,
			traverseCompositeTaskThreads, &stCompositeTaskUserData);
	ERRIFGOTO(result, _EXIT);

	result = callCompositeTaskWrapupFunctions(pstParentTask);
	ERRIFGOTO(result, _EXIT);

	// all tasks are terminated, so clear the exit flag of adjacent channels
	result = UKChannel_ClearExitByTaskId(pstParentTask->nTaskId);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result stopChildTaskThreads(int nParentTaskId, HLinkedList hTaskList)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SChildTaskAccessUserData stChildTaskAccessUserData;

	stChildTaskAccessUserData.nParentTaskId = nParentTaskId;
	stChildTaskAccessUserData.nMatchedTaskNum = 0;
	stChildTaskAccessUserData.fnCallback = stopSingleTaskThread;
	stChildTaskAccessUserData.pUserData = &nParentTaskId;
	result = UCDynamicLinkedList_Traverse(hTaskList, traverseChildTaskThreads, &stChildTaskAccessUserData);
	ERRIFGOTO(result, _EXIT);

	if(stChildTaskAccessUserData.nMatchedTaskNum == 0)
	{
		ERRASSIGNGOTO(result, ERR_UEM_NO_DATA, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKCPUTaskManager_StopTask(HCPUTaskManager hCPUTaskManager, int nTaskId)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUTaskManager *pstManager = NULL;
	uem_bool bIsCompositeTask = FALSE;
	STask *pstTargetParentTask = NULL;
	STaskThread *pstTargetThread = NULL;
#ifdef ARGUMENT_CHECK
	if (IS_VALID_HANDLE(hCPUTaskManager, ID_UEM_CPU_TASK_MANAGER) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}

	if(nTaskId < 0) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
#endif
	pstManager = hCPUTaskManager;

	result = UCThreadMutex_Lock(pstManager->hMutex);
	ERRIFGOTO(result, _EXIT);

	result = traverseAndFindTaskThread(pstManager->uControlDrivenTaskList.hTaskList, nTaskId,
										&bIsCompositeTask, &pstTargetParentTask, &pstTargetThread);
	if(result != ERR_UEM_NO_DATA) // ERR_UEM_NO_DATA is handled differently
	{
		ERRIFGOTO(result, _EXIT_LOCK);
	}

	if(bIsCompositeTask == TRUE)
	{
		result = stopCompositeTaskThreads(pstTargetParentTask, pstManager->uControlDrivenTaskList.hTaskList);
		ERRIFGOTO(result, _EXIT_LOCK);
	}
	else if(result == ERR_UEM_NO_DATA) // Task thread is not found, check it has child tasks
	{
		result = UKTask_GetTaskFromTaskId(nTaskId, &pstTargetParentTask);
		ERRIFGOTO(result, _EXIT_LOCK);

		if(pstTargetParentTask->pstSubGraph != NULL)
		{
			result = stopChildTaskThreads(nTaskId, pstManager->uControlDrivenTaskList.hTaskList);
			ERRIFGOTO(result, _EXIT_LOCK);
		}
		else
		{
			ERRASSIGNGOTO(result, ERR_UEM_NO_DATA, _EXIT_LOCK);
		}
	}
	else // single general task is found
	{
		result = stopSingleTaskThread(pstTargetThread, &nTaskId);
		ERRIFGOTO(result, _EXIT_LOCK);
	}

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
	UCThreadMutex_Unlock(pstManager->hMutex);
_EXIT:
	return result;
}


static uem_result checkCompositeTaskIsNotStopped(STaskThread *pstTaskThread, void *pUserData)
{
	uem_bool *pbNotStopped = (uem_bool *) pUserData;

	if(pstTaskThread->enTaskState != TASK_STATE_STOP)
	{
		*pbNotStopped = TRUE;
	}

	return ERR_UEM_NOERROR;
}


static uem_result runCompositeTaskThreads(STask *pstParentTask, HLinkedList hTaskList)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SCompositeTaskUserData stCompositeTaskUserData;
	int nCurModeIndex = 0;
	int nTargetModeId = 0;
	uem_bool bNotStopped = FALSE;

	stCompositeTaskUserData.nTaskId = pstParentTask->nTaskId;
	if(pstParentTask->pstMTMInfo != NULL)
	{
		nCurModeIndex = pstParentTask->pstMTMInfo->nCurModeIndex;
		nTargetModeId = pstParentTask->pstMTMInfo->astModeMap[nCurModeIndex].nModeId;
	}
	else
	{
		nTargetModeId = 0;
	}

	stCompositeTaskUserData.fnCallback = checkCompositeTaskIsNotStopped;
	stCompositeTaskUserData.pUserData = &bNotStopped;

	result = UCDynamicLinkedList_Traverse(hTaskList,
			traverseCompositeTaskThreads, &stCompositeTaskUserData);
	ERRIFGOTO(result, _EXIT);

	if(bNotStopped == TRUE)
	{
		result = stopCompositeTaskThreads(pstParentTask, hTaskList);
		ERRIFGOTO(result, _EXIT);
	}

	stCompositeTaskUserData.fnCallback = runSingleTaskThread;
	stCompositeTaskUserData.pUserData = &stCompositeTaskUserData.nTaskId;

	// Change all composite task state
	result = UCDynamicLinkedList_Traverse(hTaskList,
			traverseCompositeTaskThreads, &stCompositeTaskUserData);
	ERRIFGOTO(result, _EXIT);

	// traverse and activate thread
	stCompositeTaskUserData.fnCallback = activateSingleTaskThread;
	stCompositeTaskUserData.pUserData = &nTargetModeId;

	result = UCDynamicLinkedList_Traverse(hTaskList,
			traverseCompositeTaskThreads, &stCompositeTaskUserData);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static uem_result runChildTaskThreads(int nParentTaskId, HLinkedList hTaskList)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SChildTaskAccessUserData stChildTaskAccessUserData;

	stChildTaskAccessUserData.nParentTaskId = nParentTaskId;
	stChildTaskAccessUserData.nMatchedTaskNum = 0;
	stChildTaskAccessUserData.fnCallback = runSingleTaskThread;
	stChildTaskAccessUserData.pUserData = &nParentTaskId;
	result = UCDynamicLinkedList_Traverse(hTaskList, traverseChildTaskThreads, &stChildTaskAccessUserData);
	ERRIFGOTO(result, _EXIT);

	if(stChildTaskAccessUserData.nMatchedTaskNum == 0)
	{
		ERRASSIGNGOTO(result, ERR_UEM_NO_DATA, _EXIT);
	}

	// traverse and activate thread
	stChildTaskAccessUserData.fnCallback = activateSingleTaskThread;
	stChildTaskAccessUserData.pUserData = NULL;

	result = UCDynamicLinkedList_Traverse(hTaskList,
			traverseChildTaskThreads, &stChildTaskAccessUserData);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}



uem_result UKCPUTaskManager_RunTask(HCPUTaskManager hCPUTaskManager, int nTaskId)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUTaskManager *pstManager = NULL;
	uem_bool bIsCompositeTask = FALSE;
	STask *pstTargetParentTask = NULL;
	STaskThread *pstTargetThread = NULL;
#ifdef ARGUMENT_CHECK
	if (IS_VALID_HANDLE(hCPUTaskManager, ID_UEM_CPU_TASK_MANAGER) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}

	if(nTaskId < 0) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}

#endif
	pstManager = hCPUTaskManager;

	result = UCThreadMutex_Lock(pstManager->hMutex);
	ERRIFGOTO(result, _EXIT);

	result = traverseAndFindTaskThread(pstManager->uControlDrivenTaskList.hTaskList, nTaskId,
										&bIsCompositeTask, &pstTargetParentTask, &pstTargetThread);
	if(result != ERR_UEM_NO_DATA) // ERR_UEM_NO_DATA is handled differently
	{
		ERRIFGOTO(result, _EXIT_LOCK);
	}

	if(bIsCompositeTask == TRUE)
	{
		result = runCompositeTaskThreads(pstTargetParentTask, pstManager->uControlDrivenTaskList.hTaskList);
		ERRIFGOTO(result, _EXIT_LOCK);
	}
	else if(result == ERR_UEM_NO_DATA) // Task thread is not found, check it has child tasks
	{
		result = UKTask_GetTaskFromTaskId(nTaskId, &pstTargetParentTask);
		ERRIFGOTO(result, _EXIT_LOCK);

		if(pstTargetParentTask->pstSubGraph != NULL)
		{
			result = runChildTaskThreads(nTaskId, pstManager->uControlDrivenTaskList.hTaskList);
			ERRIFGOTO(result, _EXIT_LOCK);
		}
		else
		{
			ERRASSIGNGOTO(result, ERR_UEM_NO_DATA, _EXIT_LOCK);
		}
	}
	else // single general task is found
	{
		result = runSingleTaskThread(pstTargetThread, &nTaskId);
		ERRIFGOTO(result, _EXIT_LOCK);

		result = activateSingleTaskThread(pstTargetThread, NULL);
		ERRIFGOTO(result, _EXIT_LOCK);
	}

	result = ERR_UEM_NOERROR;

_EXIT_LOCK:
	UCThreadMutex_Unlock(pstManager->hMutex);
_EXIT:
	return result;
}


uem_result UKCPUTaskManager_ResumeTask(HCPUTaskManager hCPUTaskManager, int nTaskId)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUTaskManager *pstManager = NULL;
#ifdef ARGUMENT_CHECK
	if (IS_VALID_HANDLE(hCPUTaskManager, ID_UEM_CPU_TASK_MANAGER) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}

	if(nTaskId < 0) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
#endif

	pstManager = hCPUTaskManager;

	result = UCThreadMutex_Lock(pstManager->hMutex);
	ERRIFGOTO(result, _EXIT);

	result = updateTaskState(pstManager->uControlDrivenTaskList.hTaskList, nTaskId, TASK_STATE_RUNNING);
	ERRIFGOTO(result, _EXIT_LOCK);

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
	UCThreadMutex_Unlock(pstManager->hMutex);
_EXIT:
	return result;
}

uem_result UKCPUTaskManager_StoppingTask(HCPUTaskManager hCPUTaskManager, int nTaskId)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUTaskManager *pstManager = NULL;
#ifdef ARGUMENT_CHECK
	if (IS_VALID_HANDLE(hCPUTaskManager, ID_UEM_CPU_TASK_MANAGER) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}

	if(nTaskId < 0) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
#endif

	pstManager = hCPUTaskManager;

	result = UCThreadMutex_Lock(pstManager->hMutex);
	ERRIFGOTO(result, _EXIT);

	result = updateTaskState(pstManager->uControlDrivenTaskList.hTaskList, nTaskId, TASK_STATE_STOPPING);
	ERRIFGOTO(result, _EXIT_LOCK);

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
	UCThreadMutex_Unlock(pstManager->hMutex);
_EXIT:
	return result;
}


static uem_result traverseAndDestroyTaskThread(IN int nOffset, IN void *pData, IN void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STaskThread *pstTaskThread = (STaskThread *) pData;

	destroyTaskThreadStruct(&pstTaskThread);

	result = ERR_UEM_NOERROR;

	return result;
}

struct _STraverseCompositeTaskState {
	STask *pstParentTask;
	EInternalTaskState enConfirmedTaskState;
};

static uem_result traverseCompositeTaskStateInfo(IN int nOffset, IN void *pData, IN void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _STraverseCompositeTaskState *pstUserData = (struct _STraverseCompositeTaskState *) pUserData;
	int nLen = 0;

	STaskThread *pstTaskThread = (STaskThread *) pData;

	if(pstTaskThread->enMappedTaskType == MAPPED_TYPE_COMPOSITE_TASK && pstTaskThread->uTargetTask.pstScheduledTasks->pstParentTask == NULL)
	{
		UEMASSIGNGOTO(result, ERR_UEM_NOERROR, _EXIT);
	}

	if(pstTaskThread->enMappedTaskType == MAPPED_TYPE_COMPOSITE_TASK &&
		pstTaskThread->uTargetTask.pstScheduledTasks->pstParentTask->nTaskId == pstUserData->pstParentTask->nTaskId)
	{
		result = UCDynamicLinkedList_GetLength(pstTaskThread->uMappedCPUList.hMappedCPUList, &nLen);
		ERRIFGOTO(result, _EXIT);

		if(pstTaskThread->nFinishedThreadNum == nLen)
		{
			result = stopTaskThread(pstTaskThread, NULL);
			ERRIFGOTO(result, _EXIT);

			result = joinTaskThread(pstTaskThread, NULL);
			ERRIFGOTO(result, _EXIT);
		}

		if(pstTaskThread->enTaskState == TASK_STATE_RUNNING)
		{
			pstUserData->enConfirmedTaskState = INTERNAL_STATE_RUN;
		}
		else if(pstUserData->enConfirmedTaskState != INTERNAL_STATE_RUN && pstTaskThread->enTaskState == TASK_STATE_SUSPEND)
		{
			pstUserData->enConfirmedTaskState = INTERNAL_STATE_WAIT;
		}
		else if(pstUserData->enConfirmedTaskState != INTERNAL_STATE_RUN && pstUserData->enConfirmedTaskState != INTERNAL_STATE_WAIT &&
			pstTaskThread->enTaskState == TASK_STATE_STOPPING)
		{
			pstUserData->enConfirmedTaskState = INTERNAL_STATE_END;
		}
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKCPUTaskManager_GetTaskState(HCPUTaskManager hCPUTaskManager, int nTaskId, EInternalTaskState *penTaskState)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUTaskManager *pstManager = NULL;
	uem_bool bIsCompositeTask = FALSE;
	STask *pstParentTask = NULL;
	STaskThread *pstTaskThread = NULL;
	struct _STraverseCompositeTaskState stUserData;
	HLinkedList hLinkedList = NULL;
	int nLen = 0;
#ifdef ARGUMENT_CHECK
	if (IS_VALID_HANDLE(hCPUTaskManager, ID_UEM_CPU_TASK_MANAGER) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}

	if(nTaskId < 0) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}

	IFVARERRASSIGNGOTO(penTaskState, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif

	pstManager = hCPUTaskManager;

	result = UCThreadMutex_Lock(pstManager->hMutex);
	ERRIFGOTO(result, _EXIT);

	hLinkedList = pstManager->uControlDrivenTaskList.hTaskList;
	result = traverseAndFindTaskThread(hLinkedList, nTaskId, &bIsCompositeTask, &pstParentTask, &pstTaskThread);
	if(result == ERR_UEM_NO_DATA) // ERR_UEM_NO_DATA is handled differently
	{
		hLinkedList = pstManager->uDataAndTimeDrivenTaskList.hTaskList;
		result = traverseAndFindTaskThread(hLinkedList, nTaskId, &bIsCompositeTask, &pstParentTask, &pstTaskThread);
	}
	ERRIFGOTO(result, _EXIT_LOCK);

	if(bIsCompositeTask == TRUE)
	{
		stUserData.enConfirmedTaskState = INTERNAL_STATE_STOP;
		stUserData.pstParentTask = pstParentTask;

		// Change all composite task state
		result = UCDynamicLinkedList_Traverse(hLinkedList, traverseCompositeTaskStateInfo, &stUserData);
		ERRIFGOTO(result, _EXIT_LOCK);

		*penTaskState = stUserData.enConfirmedTaskState;
	}
	else
	{
		result = UCDynamicLinkedList_GetLength(pstTaskThread->uMappedCPUList.hMappedCPUList, &nLen);
		ERRIFGOTO(result, _EXIT);

		if(pstTaskThread->nFinishedThreadNum == nLen)
		{
			result = stopSingleTaskThread(pstTaskThread, &nTaskId);
			ERRIFGOTO(result, _EXIT);
		}

		if(pstTaskThread->enTaskState == TASK_STATE_RUNNING)
		{
			*penTaskState = INTERNAL_STATE_RUN;
		}
		else if(pstTaskThread->enTaskState == TASK_STATE_SUSPEND)
		{
			*penTaskState = INTERNAL_STATE_WAIT;
		}
		else if(pstTaskThread->enTaskState == TASK_STATE_STOPPING)
		{
			*penTaskState = INTERNAL_STATE_END;
		}
		else
		{
			*penTaskState = INTERNAL_STATE_STOP;
		}
	}

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
	UCThreadMutex_Unlock(pstManager->hMutex);
_EXIT:
	return result;
}


uem_result UKCPUTaskManager_Destroy(IN OUT HCPUTaskManager *phCPUTaskManager)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUTaskManager *pstManager = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(phCPUTaskManager, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	if (IS_VALID_HANDLE(*phCPUTaskManager, ID_UEM_CPU_TASK_MANAGER) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}

#endif
	pstManager = *phCPUTaskManager;
#ifdef ARGUMENT_CHECK
	if(pstManager->bListStatic == TRUE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
#endif

	result = UKCPUTaskManager_StopAllTasks(*phCPUTaskManager);
	ERRIFGOTO(result, _EXIT);

	result = UCDynamicLinkedList_Traverse(pstManager->uDataAndTimeDrivenTaskList.hTaskList, traverseAndDestroyTaskThread, NULL);
	ERRIFGOTO(result, _EXIT);

	result = UCDynamicLinkedList_Traverse(pstManager->uControlDrivenTaskList.hTaskList, traverseAndDestroyTaskThread, NULL);
	ERRIFGOTO(result, _EXIT);

	result = UCDynamicLinkedList_Destroy(&(pstManager->uDataAndTimeDrivenTaskList.hTaskList));
	ERRIFGOTO(result, _EXIT);

	result = UCDynamicLinkedList_Destroy(&(pstManager->uControlDrivenTaskList.hTaskList));
	ERRIFGOTO(result, _EXIT);

	result = UCThreadMutex_Destroy(&(pstManager->hMutex));
	ERRIFGOTO(result, _EXIT);

	SAFEMEMFREE(pstManager);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


