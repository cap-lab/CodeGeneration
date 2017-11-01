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

#include <uem_data.h>

#include <UKCPUTaskManager.h>


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
	UMappingTarget uTargetTask;
	ECPUTaskState enTaskState;
	EMappedTaskType enMappedTaskType;
	UMappedCPUList uMappedCPUList;
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


uem_result UKCPUTaskManager_Create(OUT HCPUTaskManager *phCPUThreadPool)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUTaskManager *pstManager = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(phCPUThreadPool, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
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

	*phCPUThreadPool = pstManager;

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

	int nData = *((int *) pData);
	int nUserData = *((int *) pUserData);

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

	SAFEMEMFREE(pstTaskThread);

	*ppstTaskThread = NULL;

	result = ERR_UEM_NOERROR;

	return result;
}

static uem_result createTaskThreadStruct(uem_bool bIsCompositeTask, UMappingTarget uTargetTask, OUT STaskThread **ppstTaskThread)
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

static uem_bool checkIsControlDrivenTask(STask *pstTask)
{
	STask *pstCurrentTask = NULL;
	uem_bool bIsControlDriven = FALSE;

	pstCurrentTask = pstTask;

	do
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

	} while(pstCurrentTask != NULL);

	return bIsControlDriven;
}

uem_result UKCPUTaskManager_RegisterTask(HCPUTaskManager hCPUThreadPool, STask *pstTask, int nCPUId)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUTaskManager *pstManager = NULL;
	STaskThread *pstTaskThread = NULL;
	UMappingTarget uTargetTask;
	struct _STaskTraverseUserData stUserData;
	void *pCPUId = 0;
	HLinkedList hTargetList = NULL;

#ifdef ARGUMENT_CHECK
	if (IS_VALID_HANDLE(hCPUThreadPool, ID_UEM_CPU_TASK_MANAGER) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}

	IFVARERRASSIGNGOTO(pstTask, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	if(nCPUId < 0 && nCPUId != MAPPING_NOT_SPECIFIED) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
#endif
	pstManager = hCPUThreadPool;

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

	result = UCDynamicLinkedList_Traverse(hTargetList, traverseTaskList, &stUserData);
	ERRIFGOTO(result, _EXIT);
	if(result == ERR_UEM_NOERROR)
	{
		uTargetTask.pstTask = pstTask;

		result = createTaskThreadStruct(FALSE, uTargetTask, &pstTaskThread);
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
		// do nothing, Already done
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	if(result != ERR_UEM_NOERROR && pstTaskThread != NULL)
	{
		destroyTaskThreadStruct(&pstTaskThread);
	}
	return result;
}


uem_result UKCPUTaskManager_RegisterCompositeTask(HCPUTaskManager hCPUThreadPool, SScheduledTasks *pstScheduledTasks, int nCPUId)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUTaskManager *pstManager = NULL;
	STaskThread *pstTaskThread = NULL;
	UMappingTarget uTargetTask;
	struct _SCompositeTaskTraverseUserData stUserData;
	void *pCPUId = 0;
	HLinkedList hTargetList = NULL;

#ifdef ARGUMENT_CHECK
	if (IS_VALID_HANDLE(hCPUThreadPool, ID_UEM_CPU_TASK_MANAGER) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}

	IFVARERRASSIGNGOTO(pstScheduledTasks, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	if(nCPUId < 0 && nCPUId != MAPPING_NOT_SPECIFIED) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
#endif
	pstManager = hCPUThreadPool;

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

		result = createTaskThreadStruct(TRUE, uTargetTask, &pstTaskThread);
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

struct _STaskSearchUserData {
	int nTaskId;
	STaskThread *pstTargetThread;
};

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


uem_result UKCPUTaskManager_SuspendTask(HCPUTaskManager hCPUThreadPool, int nTaskId)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUTaskManager *pstManager = NULL;
	struct _STaskSearchUserData stCallbackData;
#ifdef ARGUMENT_CHECK
	if (IS_VALID_HANDLE(hCPUThreadPool, ID_UEM_CPU_TASK_MANAGER) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}

	if(nTaskId < 0) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
#endif
	pstManager = hCPUThreadPool;

	stCallbackData.nTaskId = nTaskId;
	stCallbackData.pstTargetThread = NULL;

	result = UCThreadMutex_Lock(pstManager->hMutex);
	ERRIFGOTO(result, _EXIT);

	result = UCDynamicLinkedList_Traverse(pstManager->uControlDrivenTaskList.hTaskList, findTaskFromTaskId, &stCallbackData);
	ERRIFGOTO(result, _EXIT_LOCK);
	if(result == ERR_UEM_NOERROR)
	{
		ERRASSIGNGOTO(result, ERR_UEM_NO_DATA, _EXIT_LOCK);
	}
	else // result == ERR_UEM_FOUND_DATA
	{
		// TODO: general task or scheduled task?
		if(stCallbackData.pstTargetThread->enTaskState == TASK_STATE_RUNNING)
		{
			stCallbackData.pstTargetThread->enTaskState = TASK_STATE_SUSPEND;
		}
		else
		{
			ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_CONTROL, _EXIT_LOCK);
		}
	}

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
	UCThreadMutex_Unlock(pstManager->hMutex);
_EXIT:
	return result;
}


typedef struct _STaskThreadData {
	STaskThread *pstTaskThread;
	int nCurSeqId;
} STaskThreadData;



static void *taskThreadRoutine(void *pData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STaskThreadData *pstThreadData = NULL;
	STaskThread *pstTaskThread = NULL;
	STask *pstCurrentTask = NULL;

	pstThreadData = (STaskThreadData *) pData;

	pstTaskThread = pstThreadData->pstTaskThread;
	// pstThreadData->nCurSeqId

	pstCurrentTask = pstTaskThread->uTargetTask.pstTask;

	result = UCThreadEvent_WaitEvent(pstTaskThread->hEvent);
	ERRIFGOTO(result, _EXIT);

	// if nSeqId is changed, it means this thread is detached from the CPU task manager.
	// So, end this thread
	while(pstThreadData->nCurSeqId == pstTaskThread->nSeqId)
	{
		pstCurrentTask->fnGo();
	}

	//pstCurrentTask->fnWrapup();

_EXIT:
	SAFEMEMFREE(pstThreadData);
	return NULL;
}


static void *scheduledTaskThreadRoutine(void *pData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STaskThreadData *pstThreadData = NULL;
	STaskThread *pstTaskThread = NULL;

	pstThreadData = (STaskThreadData *) pData;

	pstTaskThread = pstThreadData->pstTaskThread;
	// pstThreadData->nCurSeqId

	result = UCThreadEvent_WaitEvent(pstTaskThread->hEvent);
	ERRIFGOTO(result, _EXIT);

	// if nSeqId is changed, it means this thread is detached from the CPU task manager.
	// So, end this thread
	while(pstThreadData->nCurSeqId == pstTaskThread->nSeqId)
	{

	}
_EXIT:
	SAFEMEMFREE(pstThreadData);
	return NULL;
}




static uem_result createThread(STaskThread *pstTaskThread, int nMappedCPU, FnNativeThread fnThreadRoutine)
{
	uem_result result = ERR_UEM_UNKNOWN;
	HThread hThread = NULL;
	STaskThreadData *pstThreadData = NULL;

	pstThreadData = UC_malloc(sizeof(STaskThreadData));
	ERRMEMGOTO(pstThreadData, result, _EXIT);

	pstThreadData->pstTaskThread = pstTaskThread;
	pstThreadData->nCurSeqId = pstTaskThread->nSeqId;

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

		nIndex = *((int *) pIndex);

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


static uem_result callInitFunctions(STask *pstParentTask, HStack hStack)
{
	STask *pstCurInitTask = NULL;
	uem_result result = ERR_UEM_UNKNOWN;
	SModeMap *pstCurrentModeMap = NULL;
	int nCurModeIndex = 0;
	int nCurrentIndex = 0;
	SModeMap *pstNextModeMap = NULL;
	int nNextModeIndex = 0;
	int nNextIndex = 0;
	int nStackNum = 0;
	int nNumOfTasks = 0;

	nCurModeIndex = pstParentTask->pstMTMInfo->nCurModeIndex;
	pstCurrentModeMap = &(pstParentTask->pstMTMInfo->astModeMap[nCurModeIndex]);
	nNumOfTasks = pstCurrentModeMap->nRelatedChildTaskNum;

	while(nCurrentIndex < nNumOfTasks || nStackNum > 0)
	{
		if(pstParentTask->pstSubGraph != NULL)
		{
			if(pstCurrentModeMap->pastRelatedChildTasks[nCurrentIndex]->pstMTMInfo != NULL) // MTM Graph
			{
				nNextModeIndex = pstCurrentModeMap->pastRelatedChildTasks[nCurrentIndex]->pstMTMInfo->nCurModeIndex;
				pstNextModeMap = &(pstCurrentModeMap->pastRelatedChildTasks[nCurrentIndex]->pstMTMInfo->astModeMap[nNextModeIndex]);
				nNextIndex = 0;

				// the current task has subgraph, skip current task index
				nCurrentIndex++;

				result = checkAndPopStack(hStack, &pstCurrentModeMap, &nCurrentIndex);
				ERRIFGOTO(result, _EXIT);

				// reset values
				pstCurrentModeMap = pstNextModeMap;
				nCurrentIndex = nNextIndex;

				result = UCDynamicStack_Push(hStack, pstCurrentModeMap);
				ERRIFGOTO(result, _EXIT);
#if SIZEOF_VOID_P == 8
				result = UCDynamicStack_Push(hStack, (void *) (long long) nCurrentIndex);
#else
				result = UCDynamicStack_Push(hStack, (void *) nCurrentIndex);
#endif
				ERRIFGOTO(result, _EXIT);
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
			pstCurInitTask->fnInit(pstCurInitTask->nTaskId);

			// proceed index, if all index is proceeded, pop the mode map from stack
			nCurrentIndex++;

			result = checkAndPopStack(hStack, &pstCurrentModeMap, &nCurrentIndex);
			ERRIFGOTO(result, _EXIT);
		}

		result = UCDynamicStack_Length(hStack, &nStackNum);
		ERRIFGOTO(result, _EXIT);
	}


	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result createCompositeTaskThread(HLinkedList hThreadList, STaskThread *pstTaskThread)
{
	uem_result result = ERR_UEM_UNKNOWN;
	void *pCPUId = 0;
	int nMappedCPUNumber = 0;
	STask *pstParentTask = NULL;
	//SScheduledTasks *pstTasks = NULL;
	HStack hStack = NULL;

	result = UCDynamicLinkedList_GetLength(pstTaskThread->uMappedCPUList.hMappedCPUList, &nMappedCPUNumber);
	ERRIFGOTO(result, _EXIT);

	// call TASK_INIT for the nSeqInMode is 0 which is a representative composite task needed to call multiple Task INIT functions
	if(pstTaskThread->uTargetTask.pstScheduledTasks->nSeqInMode == 0)
	{
		result = UCDynamicStack_Create(&hStack);
		ERRIFGOTO(result, _EXIT);
		// Stack with SModeMap *, current index astRelatedChildTasks

		pstParentTask = pstTaskThread->uTargetTask.pstScheduledTasks->pstParentTask;
		IFVARERRASSIGNGOTO(pstParentTask->pstMTMInfo, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

		result = callInitFunctions(pstParentTask, hStack);
		ERRIFGOTO(result, _EXIT);
	}

	if(nMappedCPUNumber > 0)
	{
		// Composite task only mapped to a single thread
		result = UCDynamicLinkedList_Get(pstTaskThread->uMappedCPUList.hMappedCPUList, LINKED_LIST_OFFSET_FIRST, 0, (void **) &pCPUId);
		ERRIFGOTO(result, _EXIT);

		result = createThread(pstTaskThread, *((int *) pCPUId), scheduledTaskThreadRoutine);
		ERRIFGOTO(result, _EXIT);
	}
	else // nMappedCPUNumber == 0
	{
		result = createThread(pstTaskThread, MAPPING_NOT_SPECIFIED, scheduledTaskThreadRoutine);
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result traverseAndCreateEachThread(IN int nOffset, IN void *pData, IN void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STaskThread *pstTaskThread = NULL;
	int nMappedCPU = 0;

	pstTaskThread = (STaskThread *) pUserData;
	nMappedCPU = *((int *) pData);

	result = createThread(pstTaskThread, nMappedCPU, taskThreadRoutine);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static uem_result createMultipleThreads(HLinkedList hMappedCPUList, STaskThread *pstTaskThread)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nMappedCPUNumber = 0;
	STask *pstTask = NULL;

	result = UCDynamicLinkedList_GetLength(hMappedCPUList, &nMappedCPUNumber);
	ERRIFGOTO(result, _EXIT);

	// call TASK_INIT
	pstTask = pstTaskThread->uTargetTask.pstTask;
	pstTask->fnInit(pstTask->nTaskId);

	if(nMappedCPUNumber > 0)
	{
		result = UCDynamicLinkedList_Traverse(hMappedCPUList, traverseAndCreateEachThread, pstTaskThread);
		ERRIFGOTO(result, _EXIT);
	}
	else // not mapped to specific task
	{
		result = createThread(pstTaskThread, MAPPING_NOT_SPECIFIED, taskThreadRoutine);
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
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
		pstTaskThread->uTargetTask.pstTask->enType == TASK_TYPE_CONTROL &&
		pstTaskThread->uTargetTask.pstTask->enRunCondition != RUN_CONDITION_CONTROL_DRIVEN)
	{
		result = createMultipleThreads(pstTaskThread->uMappedCPUList.hMappedCPUList, pstTaskThread);
		ERRIFGOTO(result, _EXIT);
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

	result = UCDynamicLinkedList_GetLength(pstTaskThread->hThreadList, &nCreatedThreadNum);
	ERRIFGOTO(result, _EXIT);

	if(nCreatedThreadNum == 0)
	{
		if(pstTaskThread->enMappedTaskType == MAPPED_TYPE_GENERAL_TASK &&
		pstTaskThread->uTargetTask.pstTask->enType == TASK_TYPE_COMPUTATIONAL &&
		pstTaskThread->uTargetTask.pstTask->enRunCondition != RUN_CONDITION_CONTROL_DRIVEN)
		{
			result = createMultipleThreads(pstTaskThread->uMappedCPUList.hMappedCPUList, pstTaskThread);
			ERRIFGOTO(result, _EXIT);
		}
		else if(pstTaskThread->enMappedTaskType == MAPPED_TYPE_COMPOSITE_TASK &&
				pstTaskThread->uTargetTask.pstScheduledTasks->pstParentTask->enRunCondition != RUN_CONDITION_CONTROL_DRIVEN)
		{
			result = createCompositeTaskThread(pstTaskThread->hThreadList, pstTaskThread);
			ERRIFGOTO(result, _EXIT);
		}
		else
		{
			// pstTaskThread->enMappedTaskType == MAPPED_TYPE_GENERAL_TASK && pstTaskThread->uTargetTask.pstTask->enType == TASK_TYPE_CONTROL
			// already created

			// pstTaskThread->enMappedTaskType == MAPPED_TYPE_GENERAL_TASK && pstTaskThread->uTargetTask.pstTask->enType == TASK_TYPE_LOOP
			// TODO: decide the behavior

			// do nothing for RUN_CONDITION_CONTROL_DRIVEN tasks
		}
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKCPUTaskManager_RunRegisteredTasks(HCPUTaskManager hCPUThreadPool)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUTaskManager *pstManager = NULL;

#ifdef ARGUMENT_CHECK
	if (IS_VALID_HANDLE(hCPUThreadPool, ID_UEM_CPU_TASK_MANAGER) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}
#endif
	pstManager = hCPUThreadPool;

	result = UCDynamicLinkedList_Traverse(pstManager->uDataAndTimeDrivenTaskList.hTaskList, traverseAndCreateControlTasks, NULL);
	ERRIFGOTO(result, _EXIT);

	result = UCDynamicLinkedList_Traverse(pstManager->uDataAndTimeDrivenTaskList.hTaskList, traverseAndCreateComputationalTasks, NULL);
	ERRIFGOTO(result, _EXIT);

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

uem_result UKCPUTaskManager_StopTask(HCPUTaskManager hCPUThreadPool, int nTaskId)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUTaskManager *pstManager = NULL;
	struct _STaskSearchUserData stCallbackData;
	int nLoop = 0;
	int nTaskInstanceNumber = 0;
#ifdef ARGUMENT_CHECK
	if (IS_VALID_HANDLE(hCPUThreadPool, ID_UEM_CPU_TASK_MANAGER) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}

	if(nTaskId < 0) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
#endif
	pstManager = hCPUThreadPool;

	stCallbackData.nTaskId = nTaskId;
	stCallbackData.pstTargetThread = NULL;

	result = UCThreadMutex_Lock(pstManager->hMutex);
	ERRIFGOTO(result, _EXIT);

	result = UCDynamicLinkedList_Traverse(pstManager->uControlDrivenTaskList.hTaskList, findTaskFromTaskId, &stCallbackData);
	ERRIFGOTO(result, _EXIT_LOCK);
	if(result == ERR_UEM_NOERROR)
	{
		ERRASSIGNGOTO(result, ERR_UEM_NO_DATA, _EXIT_LOCK);
	}
	else // result == ERR_UEM_FOUND_DATA
	{
		if(stCallbackData.pstTargetThread->enTaskState == TASK_STATE_RUNNING ||
			stCallbackData.pstTargetThread->enTaskState == TASK_STATE_SUSPEND)
		{
			stCallbackData.pstTargetThread->nSeqId++;
			stCallbackData.pstTargetThread->enTaskState = TASK_STATE_STOPPING;

			// TODO: release all channel block

			result = UCDynamicLinkedList_GetLength(stCallbackData.pstTargetThread->hThreadList, &nTaskInstanceNumber);
			ERRIFGOTO(result, _EXIT_LOCK);

			result = UCDynamicLinkedList_Traverse(stCallbackData.pstTargetThread->hThreadList, traverseAndDestroyThread, NULL);
			ERRIFGOTO(result, _EXIT_LOCK);

			for(nLoop = 0 ; nLoop < nTaskInstanceNumber ; nLoop++)
			{
				result = UCDynamicLinkedList_Remove(stCallbackData.pstTargetThread->hThreadList, LINKED_LIST_OFFSET_FIRST, 0);
				ERRIFGOTO(result, _EXIT_LOCK);
			}

			stCallbackData.pstTargetThread->enTaskState = TASK_STATE_STOP;
		}
		else
		{
			UEMASSIGNGOTO(result, ERR_UEM_ALREADY_DONE, _EXIT_LOCK);
		}
	}

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
	UCThreadMutex_Unlock(pstManager->hMutex);
_EXIT:
	return result;
}


uem_result UKCPUTaskManager_RunTask(HCPUTaskManager hCPUThreadPool, int nTaskId)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUTaskManager *pstManager = NULL;
	struct _STaskSearchUserData stCallbackData;
#ifdef ARGUMENT_CHECK
	if (IS_VALID_HANDLE(hCPUThreadPool, ID_UEM_CPU_TASK_MANAGER) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}

	if(nTaskId < 0) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}

#endif
	pstManager = hCPUThreadPool;

	stCallbackData.nTaskId = nTaskId;
	stCallbackData.pstTargetThread = NULL;

	//dasdasd


	//if(pstTask->pstSubGraph != NULL)
	{
		// Execute composite tasks if composite task is not existed, run


	}
	//else // Execute general task
	{

	}

	result = UCThreadMutex_Lock(pstManager->hMutex);
	ERRIFGOTO(result, _EXIT);

	result = UCDynamicLinkedList_Traverse(pstManager->uControlDrivenTaskList.hTaskList, findTaskFromTaskId, &stCallbackData);
	ERRIFGOTO(result, _EXIT_LOCK);
	if(result == ERR_UEM_NOERROR)
	{
		ERRASSIGNGOTO(result, ERR_UEM_NO_DATA, _EXIT_LOCK);
	}
	else // result == ERR_UEM_FOUND_DATA
	{
		if(stCallbackData.pstTargetThread->enTaskState == TASK_STATE_STOP)
		{
			result = createMultipleThreads(stCallbackData.pstTargetThread->uMappedCPUList.hMappedCPUList, stCallbackData.pstTargetThread);
			ERRIFGOTO(result, _EXIT_LOCK);
			// TODO: send event signal to execute
		}
		else // consider as an error for other cases
		{
			ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_CONTROL, _EXIT_LOCK);
		}
	}

	result = ERR_UEM_NOERROR;

_EXIT_LOCK:
	UCThreadMutex_Unlock(pstManager->hMutex);
_EXIT:
	return result;
}


uem_result UKCPUTaskManager_ResumeTask(HCPUTaskManager hCPUThreadPool, int nTaskId)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUTaskManager *pstManager = NULL;
	struct _STaskSearchUserData stCallbackData;
#ifdef ARGUMENT_CHECK
	if (IS_VALID_HANDLE(hCPUThreadPool, ID_UEM_CPU_TASK_MANAGER) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}

	if(nTaskId < 0) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
#endif

	pstManager = hCPUThreadPool;

	stCallbackData.nTaskId = nTaskId;
	stCallbackData.pstTargetThread = NULL;

	result = UCThreadMutex_Lock(pstManager->hMutex);
	ERRIFGOTO(result, _EXIT);

	result = UCDynamicLinkedList_Traverse(pstManager->uControlDrivenTaskList.hTaskList, findTaskFromTaskId, &stCallbackData);
	ERRIFGOTO(result, _EXIT_LOCK);
	if(result == ERR_UEM_NOERROR)
	{
		ERRASSIGNGOTO(result, ERR_UEM_NO_DATA, _EXIT_LOCK);
	}
	else // result == ERR_UEM_FOUND_DATA
	{
		// TODO: general task or scheduled task?
		if(stCallbackData.pstTargetThread->enTaskState == TASK_STATE_SUSPEND)
		{
			stCallbackData.pstTargetThread->enTaskState = TASK_STATE_RUNNING;
		}
		else
		{
			ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_CONTROL, _EXIT);
		}
	}

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


uem_result UKCPUTaskManager_Destroy(IN OUT HCPUTaskManager *phCPUThreadPool)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUTaskManager *pstManager = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(phCPUThreadPool, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	if (IS_VALID_HANDLE(phCPUThreadPool, ID_UEM_CPU_TASK_MANAGER) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}

#endif
	pstManager = *phCPUThreadPool;
#ifdef ARGUMENT_CHECK
	if(pstManager->bListStatic == TRUE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
#endif
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


