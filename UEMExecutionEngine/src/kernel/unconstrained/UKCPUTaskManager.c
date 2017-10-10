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

#include <uem_data.h>

#include <UKCPUTaskManager.h>

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

typedef union _UTargetTask {
	STask *pstTask;
	SScheduledTasks *pstScheduledTasks;
} UTargetTask;

typedef union _UMappedCPUList {
	int *anCPUId;
	HLinkedList hMappedCPUList;
} UMappedCPUList;

typedef struct _STaskThread {
	HLinkedList hThreadList;
	HThreadEvent hEvent;
	UTargetTask uTargetTask;
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
	UTaskList uTaskList;
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
	pstManager->uTaskList.hTaskList = NULL;

	result = UCThreadMutex_Create(&(pstManager->hMutex));
	ERRIFGOTO(result, _EXIT);

	result = UCDynamicLinkedList_Create(&(pstManager->uTaskList.hTaskList));
	ERRIFGOTO(result, _EXIT);

	*phCPUThreadPool = pstManager;

	result = ERR_UEM_NOERROR;
_EXIT:
	if(pstManager != NULL && result != ERR_UEM_NOERROR)
	{
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
#if SIZEOF_VOID_P == 8
	long long llData = (long long) pData;
	long long llUserData = (long long) pUserData;

	if(llData == llUserData)
	{
		UEMASSIGNGOTO(result, ERR_UEM_FOUND_DATA, _EXIT);
	}
#else // for other cases, SIZEOF_VOID_P is 4
	int nData = (int) pData;
	int nUserData = (int) nUserData;

	if(nData == nUserData)
	{
		UEMASSIGNGOTO(result, ERR_UEM_FOUND_DATA, _EXIT);
	}
#endif

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
		pCPUId = (int) pstUserData->nCPUId;
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


static uem_result destroyTaskThread(IN OUT STaskThread **ppstTaskThread)
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

static uem_result createTaskThread(uem_bool bIsCompositeTask, UTargetTask uTargetTask, OUT STaskThread **ppstTaskThread)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STaskThread *pstTaskThread = NULL;

	pstTaskThread = UC_malloc(sizeof(STaskThread));
	ERRMEMGOTO(pstTaskThread, result, _EXIT);

	pstTaskThread->enTaskState = TASK_STATE_STOP;
	pstTaskThread->hEvent = NULL;
	pstTaskThread->hThreadList = NULL;
	pstTaskThread->uMappedCPUList.hMappedCPUList = NULL;
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
		destroyTaskThread(&pstTaskThread);
	}
	return result;
}


uem_result UKCPUTaskManager_RegisterTask(HCPUTaskManager hCPUThreadPool, STask *pstTask, int nCPUId)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUTaskManager *pstManager = NULL;
	STaskThread *pstTaskThread = NULL;
	UTargetTask uTargetTask;
	struct _STaskTraverseUserData stUserData;
	void *pCPUId = 0;

#ifdef ARGUMENT_CHECK
	if (IS_VALID_HANDLE(hCPUThreadPool, ID_UEM_CPU_TASK_MANAGER) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}

	IFVARERRASSIGNGOTO(pstTask, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	if(nCPUId < 0) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
#endif
	pstManager = hCPUThreadPool;

	stUserData.nCPUId = nCPUId;
	stUserData.pstTask = pstTask;

	result = UCDynamicLinkedList_Traverse(pstManager->uTaskList.hTaskList, traverseTaskList, &stUserData);
	ERRIFGOTO(result, _EXIT);
	if(result == ERR_UEM_NOERROR)
	{
		uTargetTask.pstTask = pstTask;

		result = createTaskThread(FALSE, uTargetTask, &pstTaskThread);
		ERRIFGOTO(result, _EXIT);

#if SIZEOF_VOID_P == 8
		pCPUId = (void *) (long long) nCPUId;
#else
		pCPUId = (int) pstUserData->nCPUId;
#endif

		result = UCDynamicLinkedList_Add(pstTaskThread->uMappedCPUList.hMappedCPUList, LINKED_LIST_OFFSET_LAST, 0, pCPUId);
		ERRIFGOTO(result, _EXIT);

		result = UCDynamicLinkedList_Add(pstManager->uTaskList.hTaskList, LINKED_LIST_OFFSET_FIRST, 0, pstTaskThread);
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
		destroyTaskThread(&pstTaskThread);
	}
	return result;
}


uem_result UKCPUTaskManager_RegisterCompositeTask(HCPUTaskManager hCPUThreadPool, SScheduledTasks *pstScheduledTasks, int nCPUId)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUTaskManager *pstManager = NULL;
	STaskThread *pstTaskThread = NULL;
	UTargetTask uTargetTask;
	struct _SCompositeTaskTraverseUserData stUserData;
	void *pCPUId = 0;

#ifdef ARGUMENT_CHECK
	if (IS_VALID_HANDLE(hCPUThreadPool, ID_UEM_CPU_TASK_MANAGER) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}

	IFVARERRASSIGNGOTO(pstScheduledTasks, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	if(nCPUId < 0) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
#endif
	pstManager = hCPUThreadPool;

	stUserData.nCPUId = nCPUId;
	stUserData.pstScheduledTasks = pstScheduledTasks;

	result = UCDynamicLinkedList_Traverse(pstManager->uTaskList.hTaskList, traverseCompositeTaskList, &stUserData);
	ERRIFGOTO(result, _EXIT);
	if(result == ERR_UEM_NOERROR)
	{
		uTargetTask.pstScheduledTasks = pstScheduledTasks;

		result = createTaskThread(TRUE, uTargetTask, &pstTaskThread);
		ERRIFGOTO(result, _EXIT);
#if SIZEOF_VOID_P == 8
		pCPUId = (void *) (long long) nCPUId;
#else
		pCPUId = (int) pstUserData->nCPUId;
#endif

		result = UCDynamicLinkedList_Add(pstTaskThread->uMappedCPUList.hMappedCPUList, LINKED_LIST_OFFSET_LAST, 0, pCPUId);
		ERRIFGOTO(result, _EXIT);

		result = UCDynamicLinkedList_Add(pstManager->uTaskList.hTaskList, LINKED_LIST_OFFSET_FIRST, 0, pstTaskThread);
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
		destroyTaskThread(&pstTaskThread);
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

	result = UCDynamicLinkedList_Traverse(pstManager->uTaskList.hTaskList, findTaskFromTaskId, &stCallbackData);
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

static void * taskThreadRoutine(void *pData)
{
	int nNum;

	return NULL;
}


static void * scheduledTaskThreadRoutine(void *pData)
{
	int nNum;

	return NULL;
}

static uem_result createMultipleThreads(HLinkedList hThreadList, HLinkedList hMappedCPUList)
{
	uem_result result = ERR_UEM_UNKNOWN;
	result = ERR_UEM_NOERROR;
	HThread hThread = NULL;
	int nLoop = 0;
	int nTaskInstanceNumber = 0;

	result = UCDynamicLinkedList_GetLength(hMappedCPUList, &nTaskInstanceNumber);
	ERRIFGOTO(result, _EXIT);

	for(nLoop = 0 ; nLoop < nTaskInstanceNumber ; nLoop++)
	{
		result = UCThread_Create(taskThreadRoutine, NULL, &hThread);
		ERRIFGOTO(result, _EXIT);

		result = UCDynamicLinkedList_Add(hThreadList, LINKED_LIST_OFFSET_LAST, 0, (void *) hThread);
		ERRIFGOTO(result, _EXIT);

		hThread = NULL;
	}
_EXIT:
	if(hThread != NULL)
	{
		UCThread_Destroy(&hThread, TRUE);
	}
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
		result = createMultipleThreads(pstTaskThread->hThreadList, pstTaskThread->uMappedCPUList.hMappedCPUList);
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static uem_result traverseAndCreateGeneralTasks(IN int nOffset, IN void *pData, IN void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STaskThread *pstTaskThread = (STaskThread *) pData;
	int nCreatedThreadNum = 0;

	result = UCDynamicLinkedList_GetLength(pstTaskThread->hThreadList, &nCreatedThreadNum);
	ERRIFGOTO(result, _EXIT);

	if(nCreatedThreadNum == 0 && pstTaskThread->enMappedTaskType == MAPPED_TYPE_GENERAL_TASK &&
		pstTaskThread->uTargetTask.pstTask->enType == TASK_TYPE_COMPUTATIONAL)
	{
		result = createMultipleThreads(pstTaskThread->hThreadList, pstTaskThread->uMappedCPUList.hMappedCPUList);
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static uem_result traverseAndCreateCompositeTasks(IN int nOffset, IN void *pData, IN void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STaskThread *pstTaskThread = (STaskThread *) pData;
	int nCreatedThreadNum = 0;

	result = UCDynamicLinkedList_GetLength(pstTaskThread->hThreadList, &nCreatedThreadNum);
	ERRIFGOTO(result, _EXIT);

	if(nCreatedThreadNum == 0 && pstTaskThread->enMappedTaskType == MAPPED_TYPE_COMPOSITE_TASK)
	{
		result = createMultipleThreads(pstTaskThread->hThreadList, pstTaskThread->uMappedCPUList.hMappedCPUList);
		ERRIFGOTO(result, _EXIT);
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

	result = UCDynamicLinkedList_Traverse(pstManager->uTaskList.hTaskList, traverseAndCreateControlTasks, NULL);
	ERRIFGOTO(result, _EXIT);

	result = UCDynamicLinkedList_Traverse(pstManager->uTaskList.hTaskList, traverseAndCreateGeneralTasks, NULL);
	ERRIFGOTO(result, _EXIT);

	result = UCDynamicLinkedList_Traverse(pstManager->uTaskList.hTaskList, traverseAndCreateCompositeTasks, NULL);
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

	result = UCThread_Destroy(&(hThread), TRUE);
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

	result = UCDynamicLinkedList_Traverse(pstManager->uTaskList.hTaskList, findTaskFromTaskId, &stCallbackData);
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

	result = UCThreadMutex_Lock(pstManager->hMutex);
	ERRIFGOTO(result, _EXIT);

	result = UCDynamicLinkedList_Traverse(pstManager->uTaskList.hTaskList, findTaskFromTaskId, &stCallbackData);
	ERRIFGOTO(result, _EXIT_LOCK);
	if(result == ERR_UEM_NOERROR)
	{
		ERRASSIGNGOTO(result, ERR_UEM_NO_DATA, _EXIT_LOCK);
	}
	else // result == ERR_UEM_FOUND_DATA
	{
		if(stCallbackData.pstTargetThread->enTaskState == TASK_STATE_STOP)
		{
			result = createMultipleThreads(stCallbackData.pstTargetThread->hThreadList, stCallbackData.pstTargetThread->uMappedCPUList.hMappedCPUList);
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

	result = UCDynamicLinkedList_Traverse(pstManager->uTaskList.hTaskList, findTaskFromTaskId, &stCallbackData);
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

	destroyTaskThread(&pstTaskThread);

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
	result = UCDynamicLinkedList_Traverse(pstManager->uTaskList.hTaskList, traverseAndDestroyTaskThread, NULL);
	ERRIFGOTO(result, _EXIT);

	result = UCDynamicLinkedList_Destroy(&(pstManager->uTaskList.hTaskList));
	ERRIFGOTO(result, _EXIT);

	result = UCThreadMutex_Destroy(&(pstManager->hMutex));
	ERRIFGOTO(result, _EXIT);

	SAFEMEMFREE(pstManager);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


