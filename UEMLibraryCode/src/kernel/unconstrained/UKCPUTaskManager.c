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
#include <UCString.h>
#include <UCAlloc.h>
#include <UCDynamicLinkedList.h>
#include <UCThreadMutex.h>
#include <UCThreadEvent.h>
#include <UCThread.h>
#include <UCDynamicStack.h>
#include <UCTime.h>

#include <uem_data.h>

#include <UKTask_internal.h>

#include <UKChannel_internal.h>
#include <UKCPUTaskManager.h>
#include <UKCPUCompositeTaskManager.h>
#include <UKCPUGeneralTaskManager.h>
#include <UKModelController.h>

#define THREAD_DESTROY_TIMEOUT (5000)
#define CONTROL_WAIT_TIMEOUT (3000)

typedef struct _SCPUTaskManager {
	EUemModuleId enId;
	HCPUCompositeTaskManager hCompositeManager;
	HCPUGeneralTaskManager hGeneralManager;
	uem_bool bListStatic;
	HThreadMutex hMutex;
} SCPUTaskManager;


typedef struct _SSubgraphTaskStateUserData {
	HCPUGeneralTaskManager hGeneralTaskManager;
	int nTaskStateStop;
	int nTaskStateRunning;
	int nTaskStateStopping;
	int nTaskStateSuspend;
} SSubgraphTaskStateUserData;

typedef struct _SMaxIterationSetCallback {
	int nMaxIteration;
	int nBaseTaskId;
} SMaxIterationSetCallback;

typedef struct _SMappingSetUserData {
	HCPUGeneralTaskManager hGeneralTaskManager;
	const char *pszMappingSet;
} SMappingSetUserData;

uem_result UKCPUTaskManager_Create(OUT HCPUTaskManager *phCPUTaskManager)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUTaskManager *pstManager = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(phCPUTaskManager, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	pstManager = UCAlloc_malloc(sizeof(SCPUTaskManager));
	ERRMEMGOTO(pstManager, result, _EXIT);

	pstManager->enId = ID_UEM_CPU_TASK_MANAGER;
	pstManager->bListStatic = FALSE;
	pstManager->hMutex = NULL;

	// dynamically create managers when registering tasks
	pstManager->hCompositeManager = NULL;
	pstManager->hGeneralManager = NULL;

	result = UCThreadMutex_Create(&(pstManager->hMutex));
	ERRIFGOTO(result, _EXIT);

	*phCPUTaskManager = pstManager;

	result = ERR_UEM_NOERROR;
_EXIT:
	if(pstManager != NULL && result != ERR_UEM_NOERROR)
	{
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

static uem_result checkMappingSetIsEqaul(SMappedGeneralTaskInfo *pstMappedTask, const char *pszMappingName, OUT uem_bool *pbIsEqual)
{
	uem_result result = ERR_UEM_UNKNOWN;
	uem_string_struct stCurrentMappingSet;
	uem_string_struct stTargetMappingSet;

	result = UCString_New(&stCurrentMappingSet, (char *)pstMappedTask->pszMappingSet, UEMSTRING_CONST);
	ERRIFGOTO(result, _EXIT);

	result = UCString_New(&stTargetMappingSet, (char *)pszMappingName, UEMSTRING_CONST);
	ERRIFGOTO(result, _EXIT);

	*pbIsEqual = UCString_IsEqual(&stCurrentMappingSet, &stTargetMappingSet);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKCPUTaskManager_RegisterTask(HCPUTaskManager hCPUTaskManager, SMappedGeneralTaskInfo *pstMappedTask)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUTaskManager *pstManager = NULL;
	uem_bool bIsDefaultMapping = FALSE;

#ifdef ARGUMENT_CHECK
	if (IS_VALID_HANDLE(hCPUTaskManager, ID_UEM_CPU_TASK_MANAGER) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}

	IFVARERRASSIGNGOTO(pstMappedTask, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	if(pstMappedTask->nLocalId < 0 && pstMappedTask->nLocalId != MAPPING_NOT_SPECIFIED) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
#endif
	pstManager = hCPUTaskManager;

	result = checkMappingSetIsEqaul(pstMappedTask, DEFAULT_STRING_NAME, &bIsDefaultMapping);
	ERRIFGOTO(result, _EXIT);

	if(bIsDefaultMapping == FALSE)
	{
		UEMASSIGNGOTO(result, ERR_UEM_NOERROR, _EXIT);	
	}

	if(pstManager->hGeneralManager == NULL)
	{
		result = UKCPUGeneralTaskManager_Create(&(pstManager->hGeneralManager));
		ERRIFGOTO(result, _EXIT);
	}

	result = UKCPUGeneralTaskManager_RegisterTask(pstManager->hGeneralManager, pstMappedTask);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKCPUTaskManager_RegisterCompositeTask(HCPUTaskManager hCPUTaskManager, SMappedCompositeTaskInfo *pstMappedTask)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUTaskManager *pstManager = NULL;
#ifdef ARGUMENT_CHECK
	if (IS_VALID_HANDLE(hCPUTaskManager, ID_UEM_CPU_TASK_MANAGER) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}

	IFVARERRASSIGNGOTO(pstMappedTask, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	if(pstMappedTask->nLocalId < 0) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
#endif
	pstManager = hCPUTaskManager;

	if(pstManager->hCompositeManager == NULL)
	{
		result = UKCPUCompositeTaskManager_Create(&(pstManager->hCompositeManager));
		ERRIFGOTO(result, _EXIT);
	}

	result = UKCPUCompositeTaskManager_RegisterTask(pstManager->hCompositeManager, pstMappedTask);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result suspendDataflowSubgraphTask(STask *pstTask, void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	HCPUGeneralTaskManager hGeneralTaskManager = NULL;

	hGeneralTaskManager = (HCPUGeneralTaskManager) pUserData;

	result = UKCPUGeneralTaskManager_ChangeState(hGeneralTaskManager, pstTask, TASK_STATE_SUSPEND);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKCPUTaskManager_SuspendTask(HCPUTaskManager hCPUTaskManager, int nTaskId)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUTaskManager *pstManager = NULL;
	STask *pstTask = NULL;
#ifdef ARGUMENT_CHECK
	if (IS_VALID_HANDLE(hCPUTaskManager, ID_UEM_CPU_TASK_MANAGER) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}

	if(nTaskId < 0) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
#endif
	pstManager = hCPUTaskManager;

	result = UKTask_GetTaskFromTaskId(nTaskId, &pstTask);
	ERRIFGOTO(result, _EXIT);

	if(pstTask->bStaticScheduled == TRUE && pstTask->pstSubGraph != NULL)
	{
		result = UKCPUCompositeTaskManager_ChangeState(pstManager->hCompositeManager, pstTask, TASK_STATE_SUSPEND);
		ERRIFGOTO(result, _EXIT);
	}
	else if(pstTask->pstSubGraph != NULL) // task with subgraph which is not static scheduled cannot be controlled
	{
		// task with subgraph which is not SDF cannot be controlled
		if(pstTask->pstSubGraph->enType == GRAPH_TYPE_PROCESS_NETWORK)
		{
			ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
		}
		else
		{
			result = UKCPUTaskCommon_TraverseSubGraphTasks(pstTask, suspendDataflowSubgraphTask, pstManager->hGeneralManager);
			ERRIFGOTO(result, _EXIT);
		}
	}
	else
	{
		result = UKCPUGeneralTaskManager_ChangeState(pstManager->hGeneralManager, pstTask, TASK_STATE_SUSPEND);
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result traverseAndCreateControlTasks(STask *pstTask, IN void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	HCPUGeneralTaskManager hManager = NULL;
	ECPUTaskState enTaskState = TASK_STATE_STOP;

	hManager = (HCPUGeneralTaskManager) pUserData;

	if (pstTask->enType == TASK_TYPE_CONTROL)
	{
		result = UKCPUGeneralTaskManager_GetTaskState(hManager, pstTask, &enTaskState);
		ERRIFGOTO(result, _EXIT);

		if(enTaskState == TASK_STATE_RUNNING)
		{
			result = UKCPUGeneralTaskManager_CreateThread(hManager, pstTask);
			ERRIFGOTO(result, _EXIT);

			result = UKCPUGeneralTaskManager_ChangeState(hManager, pstTask, TASK_STATE_RUNNING);
			ERRIFGOTO(result, _EXIT);

			// Send event signal to execute
			result = UKCPUGeneralTaskManager_ActivateThread(hManager, pstTask);
			ERRIFGOTO(result, _EXIT);

			result = UKCPUGeneralTaskManager_WaitTaskActivated(hManager, pstTask, CONTROL_WAIT_TIMEOUT);
			ERRIFGOTO(result, _EXIT);
		}
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result traverseAndCreateGeneralTasks(STask *pstTask, IN void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	HCPUGeneralTaskManager hManager = NULL;
	ECPUTaskState enTaskState = TASK_STATE_STOP;

	hManager = (HCPUGeneralTaskManager) pUserData;

	if ((pstTask->enType == TASK_TYPE_COMPUTATIONAL || pstTask->enType == TASK_TYPE_LOOP)  && pstTask->enRunCondition != RUN_CONDITION_CONTROL_DRIVEN)
	{
		result = UKCPUGeneralTaskManager_GetTaskState(hManager, pstTask, &enTaskState);
		ERRIFGOTO(result, _EXIT);

		if(enTaskState == TASK_STATE_RUNNING)
		{
			result = UKCPUGeneralTaskManager_CreateThread(hManager, pstTask);
			ERRIFGOTO(result, _EXIT);

			result = UKCPUGeneralTaskManager_ChangeState(hManager, pstTask, TASK_STATE_RUNNING);
			ERRIFGOTO(result, _EXIT);

			// Send event signal to execute
			result = UKCPUGeneralTaskManager_ActivateThread(hManager, pstTask);
			ERRIFGOTO(result, _EXIT);
		}
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result traverseAndActivateTasks(STask *pstTask, IN void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	HCPUGeneralTaskManager hManager = NULL;
	ECPUTaskState enTaskState = TASK_STATE_STOP;

	hManager = (HCPUGeneralTaskManager) pUserData;

	if (pstTask->enType == TASK_TYPE_COMPUTATIONAL || pstTask->enType == TASK_TYPE_LOOP)
	{
		result = UKCPUGeneralTaskManager_GetTaskState(hManager, pstTask, &enTaskState);
		ERRIFGOTO(result, _EXIT);

		if(enTaskState == TASK_STATE_RUNNING)
		{
			// Send event signal to execute
			result = UKCPUGeneralTaskManager_ActivateThread(hManager, pstTask);
			ERRIFGOTO(result, _EXIT);
		}
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static uem_result traverseAndCreateCompositeTasks(STask *pstTask, IN void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	HCPUCompositeTaskManager hManager = NULL;
	ECPUTaskState enTaskState = TASK_STATE_STOP;

	hManager = (HCPUCompositeTaskManager) pUserData;

	result = UKCPUCompositeTaskManager_GetTaskState(hManager, pstTask, &enTaskState);
	ERRIFGOTO(result, _EXIT);

	if(pstTask == NULL || (enTaskState == TASK_STATE_RUNNING && pstTask->enRunCondition != RUN_CONDITION_CONTROL_DRIVEN))
	{
		result = UKCPUCompositeTaskManager_CreateThread(hManager, pstTask);
		ERRIFGOTO(result, _EXIT);

		result = UKCPUCompositeTaskManager_ChangeState(hManager, pstTask, TASK_STATE_RUNNING);
		ERRIFGOTO(result, _EXIT);

		// Send event signal to execute
		result = UKCPUCompositeTaskManager_ActivateThread(hManager, pstTask);
		ERRIFGOTO(result, _EXIT);
	}

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

	if(pstManager->hGeneralManager != NULL)
	{
		result = UKCPUGeneralTaskManager_TraverseGeneralTaskList(pstManager->hGeneralManager, traverseAndCreateControlTasks, pstManager->hGeneralManager);
		ERRIFGOTO(result, _EXIT);

		UCThread_Yield();

		result = UKCPUGeneralTaskManager_TraverseGeneralTaskList(pstManager->hGeneralManager, traverseAndCreateGeneralTasks, pstManager->hGeneralManager);
		ERRIFGOTO(result, _EXIT);

		result = UKCPUGeneralTaskManager_TraverseGeneralTaskList(pstManager->hGeneralManager, traverseAndActivateTasks, pstManager->hGeneralManager);
		ERRIFGOTO(result, _EXIT);
	}

	if(pstManager->hCompositeManager != NULL)
	{
		result = UKCPUCompositeTaskManager_TraverseCompositeTaskList(pstManager->hCompositeManager, traverseAndCreateCompositeTasks, pstManager->hCompositeManager);
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result traverseAndDestroyCompositeThread(STask *pstTask, IN void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	HCPUCompositeTaskManager hManager = NULL;

	hManager = (HCPUCompositeTaskManager) pUserData;

	result = UKCPUCompositeTaskManager_DestroyThread(hManager, pstTask);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result traverseAndDestroyGeneralThread(STask *pstTask, IN void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	HCPUGeneralTaskManager hManager = NULL;

	hManager = (HCPUGeneralTaskManager) pUserData;

	result = UKCPUGeneralTaskManager_DestroyThread(hManager, pstTask);
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

	//UKCPUCompositeTaskManager_DestroyThread(hManager, pstTargetTask)
	if(pstManager->hCompositeManager != NULL)
	{
		result = UKCPUCompositeTaskManager_TraverseCompositeTaskList(pstManager->hCompositeManager, traverseAndDestroyCompositeThread, pstManager->hCompositeManager);
		ERRIFGOTO(result, _EXIT);
	}

	if(pstManager->hGeneralManager != NULL)
	{
		result = UKCPUGeneralTaskManager_TraverseGeneralTaskList(pstManager->hGeneralManager, traverseAndDestroyGeneralThread, pstManager->hGeneralManager);
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result stopDataflowSubgraphTask(STask *pstTask, void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	HCPUGeneralTaskManager hGeneralTaskManager = NULL;

	hGeneralTaskManager = (HCPUGeneralTaskManager) pUserData;

	result = UKCPUGeneralTaskManager_DestroyThread(hGeneralTaskManager, pstTask);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static uem_result waitUntilGeneralTaskIsStarted(HCPUGeneralTaskManager hGeneralTaskManager, STask *pstTask)
{
	uem_result result = ERR_UEM_UNKNOWN;
	uem_bool bStarted = FALSE;
	ECPUTaskState enState;

	while(bStarted == FALSE)
	{
		result = UKCPUGeneralTaskManager_GetTaskState(hGeneralTaskManager, pstTask, &enState);
		ERRIFGOTO(result, _EXIT);

		if(enState == TASK_STATE_RUNNING)
		{
			result = UKCPUGeneralTaskManager_CheckTaskStarted(hGeneralTaskManager, pstTask, &bStarted);
			ERRIFGOTO(result, _EXIT);

			UCThread_Yield();
		}
		else if(enState == TASK_STATE_SUSPEND)
		{
			bStarted = TRUE;
		}
		else
		{
			ERRASSIGNGOTO(result, ERR_UEM_INTERNAL_FAIL, _EXIT);
		}
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result runDataflowSubgraphTask(STask *pstTask, void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	HCPUGeneralTaskManager hGeneralTaskManager = NULL;

	hGeneralTaskManager = (HCPUGeneralTaskManager) pUserData;

	result = UKCPUGeneralTaskManager_CreateThread(hGeneralTaskManager, pstTask);
	ERRIFGOTO(result, _EXIT);

	result = UKCPUGeneralTaskManager_ChangeState(hGeneralTaskManager, pstTask, TASK_STATE_RUNNING);
	ERRIFGOTO(result, _EXIT);

	result = UKCPUGeneralTaskManager_ActivateThread(hGeneralTaskManager, pstTask);
	ERRIFGOTO(result, _EXIT);

	result = waitUntilGeneralTaskIsStarted(hGeneralTaskManager, pstTask);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result runOnceDataflowSubgraphTask(STask *pstTask, void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	HCPUGeneralTaskManager hGeneralTaskManager = NULL;

	hGeneralTaskManager = (HCPUGeneralTaskManager) pUserData;

	result = UKCPUGeneralTaskManager_CreateThread(hGeneralTaskManager, pstTask);
			ERRIFGOTO(result, _EXIT);

	result = UKCPUGeneralTaskManager_ChangeState(hGeneralTaskManager, pstTask, TASK_STATE_STOPPING);
	ERRIFGOTO(result, _EXIT);

	result = UKCPUGeneralTaskManager_ActivateThread(hGeneralTaskManager, pstTask);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result resumeDataflowSubgraphTask(STask *pstTask, void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	HCPUGeneralTaskManager hGeneralTaskManager = NULL;

	hGeneralTaskManager = (HCPUGeneralTaskManager) pUserData;

	result = UKCPUGeneralTaskManager_CreateThread(hGeneralTaskManager, pstTask);
			ERRIFGOTO(result, _EXIT);

	result = UKCPUGeneralTaskManager_ChangeState(hGeneralTaskManager, pstTask, TASK_STATE_RUNNING);
	ERRIFGOTO(result, _EXIT);

	result = UKCPUGeneralTaskManager_ActivateThread(hGeneralTaskManager, pstTask);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result setMaximumTaskIteration(STask *pstTask, void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SMaxIterationSetCallback *pstIterationSet = NULL;
	int nCurMaxIteration = 0;
	STask *pstParentTask = NULL;

	pstIterationSet = (SMaxIterationSetCallback *) pUserData;

	nCurMaxIteration = pstIterationSet->nMaxIteration;
	pstParentTask = pstTask;
	while(pstParentTask->pstParentGraph->pstParentTask != NULL)
	{
		pstParentTask = pstParentTask->pstParentGraph->pstParentTask;
	}

	result = UKTask_SetTargetIteration(pstTask, nCurMaxIteration, pstIterationSet->nBaseTaskId);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result clearTaskIteration(STask *pstTask, void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = UKTask_ClearRunCount(pstTask);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result activateDataflowSubgraphTask(STask *pstTask, void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	HCPUGeneralTaskManager hGeneralTaskManager = NULL;

	hGeneralTaskManager = (HCPUGeneralTaskManager) pUserData;

	result = UKCPUGeneralTaskManager_ActivateThread(hGeneralTaskManager, pstTask);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result stoppingDataflowSubgraphTask(STask *pstTask, void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	HCPUGeneralTaskManager hGeneralTaskManager = NULL;

	hGeneralTaskManager = (HCPUGeneralTaskManager) pUserData;

	result = UKCPUGeneralTaskManager_ChangeState(hGeneralTaskManager, pstTask, TASK_STATE_STOPPING);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result getTaskStateDataflowSubgraphTask(STask *pstTask, void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	HCPUGeneralTaskManager hGeneralTaskManager = NULL;
	SSubgraphTaskStateUserData *pstUserData = NULL;
	ECPUTaskState enTaskState;

	pstUserData = (SSubgraphTaskStateUserData *) pUserData;

	hGeneralTaskManager = pstUserData->hGeneralTaskManager;

	result = UKCPUGeneralTaskManager_GetTaskState(hGeneralTaskManager, pstTask, &enTaskState);
	ERRIFGOTO(result, _EXIT);

	switch(enTaskState)
	{
	case TASK_STATE_RUNNING:
		pstUserData->nTaskStateRunning++;
		break;
	case TASK_STATE_STOP:
		pstUserData->nTaskStateStop++;
		break;
	case TASK_STATE_STOPPING:
		pstUserData->nTaskStateStopping++;
		break;
	case TASK_STATE_SUSPEND:
		pstUserData->nTaskStateSuspend++;
		break;
	default:
		ERRASSIGNGOTO(result, ERR_UEM_INTERNAL_FAIL, _EXIT);
		break;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result getSubgraphTaskState(HCPUGeneralTaskManager hGeneralTaskManager, STask *pstTask, OUT ECPUTaskState *penTaskState)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSubgraphTaskStateUserData stUserData;
	ECPUTaskState enTaskState;

	stUserData.hGeneralTaskManager = hGeneralTaskManager;
	stUserData.nTaskStateStopping = 0;
	stUserData.nTaskStateRunning = 0;
	stUserData.nTaskStateStop = 0;
	stUserData.nTaskStateSuspend = 0;

	result = UKCPUTaskCommon_TraverseSubGraphTasks(pstTask, getTaskStateDataflowSubgraphTask, &stUserData);
	ERRIFGOTO(result, _EXIT);

	if(stUserData.nTaskStateRunning > 0)
	{
		enTaskState = TASK_STATE_RUNNING;
	}
	else if(stUserData.nTaskStateSuspend > 0)
	{
		enTaskState = TASK_STATE_SUSPEND;
	}
	else if(stUserData.nTaskStateStopping > 0)
	{
		enTaskState = TASK_STATE_STOPPING;
	}
	else
	{
		enTaskState = TASK_STATE_STOP;
	}

	*penTaskState = enTaskState;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static uem_result setMaximumTaskIterationInSubGraph(HCPUTaskManager hCPUTaskManager, STask* pstTask)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUTaskManager *pstManager = NULL;
	int nLoop = 0 ;
	int nMaxIteration = 0;
	SMaxIterationSetCallback stIterationSet;

	pstManager = hCPUTaskManager;

	result = UKCPUTaskCommon_TraverseSubGraphTasks(pstTask, suspendDataflowSubgraphTask, pstManager->hGeneralManager);
	ERRIFGOTO(result, _EXIT);

	for(nLoop = 0 ; nLoop < pstTask->pstSubGraph->nNumOfTasks ; nLoop++)
	{
		if(nMaxIteration < pstTask->pstSubGraph->astTasks[nLoop].nCurIteration)
		{
			nMaxIteration = pstTask->pstSubGraph->astTasks[nLoop].nCurIteration;
			UEM_DEBUG_PRINT("Max iteration: %d, task name: %s\n", nMaxIteration, pstTask->pstSubGraph->astTasks[nLoop].pszTaskName);
		}
	}

	stIterationSet.nBaseTaskId = pstTask->nTaskId;
	stIterationSet.nMaxIteration = nMaxIteration;

	result = UKCPUTaskCommon_TraverseSubGraphTasks(pstTask, setMaximumTaskIteration, &stIterationSet);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKCPUTaskManager_StopTask(HCPUTaskManager hCPUTaskManager, int nTaskId)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUTaskManager *pstManager = NULL;
	STask *pstTask = NULL;
	ECPUTaskState enTaskState;
#ifdef ARGUMENT_CHECK
	if (IS_VALID_HANDLE(hCPUTaskManager, ID_UEM_CPU_TASK_MANAGER) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}

	if(nTaskId < 0) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
#endif
	pstManager = hCPUTaskManager;

	result = UKTask_GetTaskFromTaskId(nTaskId, &pstTask);
	ERRIFGOTO(result, _EXIT);

	if(pstTask->bStaticScheduled == TRUE && pstTask->pstSubGraph != NULL)
	{
		result = UKCPUCompositeTaskManager_DestroyThread(pstManager->hCompositeManager, pstTask);
		ERRIFGOTO(result, _EXIT);
	}
	else if(pstTask->pstSubGraph != NULL)
	{
		// task with subgraph which is not SDF cannot be controlled
		if(pstTask->pstSubGraph->enType == GRAPH_TYPE_PROCESS_NETWORK)
		{
			ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
		}
		else
		{

			result = setMaximumTaskIterationInSubGraph(hCPUTaskManager, pstTask);
			ERRIFGOTO(result, _EXIT);

			result = UKCPUTaskCommon_TraverseSubGraphTasks(pstTask, stoppingDataflowSubgraphTask, pstManager->hGeneralManager);
			ERRIFGOTO(result, _EXIT);

			do // wait for the tasks are going to be stopped
			{
				UCThread_Yield();

				result = getSubgraphTaskState(pstManager->hGeneralManager, pstTask, &enTaskState);
				ERRIFGOTO(result, _EXIT);
			}while(enTaskState != TASK_STATE_STOP);

			result = UKCPUTaskCommon_TraverseSubGraphTasks(pstTask, stopDataflowSubgraphTask, pstManager->hGeneralManager);
			ERRIFGOTO(result, _EXIT);
		}
	}
	else
	{
		result = UKCPUGeneralTaskManager_DestroyThread(pstManager->hGeneralManager, pstTask);
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKCPUTaskManager_RunTask(HCPUTaskManager hCPUTaskManager, int nTaskId)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUTaskManager *pstManager = NULL;
	STask *pstTask = NULL;
#ifdef ARGUMENT_CHECK
	if (IS_VALID_HANDLE(hCPUTaskManager, ID_UEM_CPU_TASK_MANAGER) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}

	if(nTaskId < 0) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}

#endif
	pstManager = hCPUTaskManager;

	result = UKTask_GetTaskFromTaskId(nTaskId, &pstTask);
	ERRIFGOTO(result, _EXIT);

	if(pstTask->pstSubGraph != NULL)
	{
		if(pstTask->pstSubGraph->enType == GRAPH_TYPE_PROCESS_NETWORK)
		{
			ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
		}

		result = UKChannel_ClearChannelInSubgraph(pstTask->nTaskId);
		ERRIFGOTO(result, _EXIT);

		result = UKModelController_CallSubGraphClearFunctions(pstTask->pstSubGraph);
		ERRIFGOTO(result, _EXIT);

		if(pstTask->bStaticScheduled == TRUE)
		{
			result = UKCPUCompositeTaskManager_CreateThread(pstManager->hCompositeManager, pstTask);
			ERRIFGOTO(result, _EXIT);

			result = UKCPUCompositeTaskManager_ChangeState(pstManager->hCompositeManager, pstTask, TASK_STATE_RUNNING);
			ERRIFGOTO(result, _EXIT);

			result = UKCPUCompositeTaskManager_ActivateThread(pstManager->hCompositeManager, pstTask);
			ERRIFGOTO(result, _EXIT);
		}
		else
		{
			result = UKCPUTaskCommon_TraverseSubGraphTasks(pstTask, clearTaskIteration, NULL);
			ERRIFGOTO(result, _EXIT);

			if(pstTask->enRunCondition == RUN_CONDITION_CONTROL_DRIVEN)
			{
				SMaxIterationSetCallback stIterationSet;

				stIterationSet.nBaseTaskId = nTaskId;
				stIterationSet.nMaxIteration = pstTask->nCurIteration+1;

				result = UKCPUTaskCommon_TraverseSubGraphTasks(pstTask, setMaximumTaskIteration, &stIterationSet);
				ERRIFGOTO(result, _EXIT);

				result = UKCPUTaskCommon_TraverseSubGraphTasks(pstTask, runOnceDataflowSubgraphTask, pstManager->hGeneralManager);
				ERRIFGOTO(result, _EXIT);
			}
			else
			{
				result = UKCPUTaskCommon_TraverseSubGraphTasks(pstTask, runDataflowSubgraphTask, pstManager->hGeneralManager);
				ERRIFGOTO(result, _EXIT);
			}
		}
	}
	else
	{
		result = UKTask_ClearRunCount(pstTask);
		ERRIFGOTO(result, _EXIT);

		result = UKCPUGeneralTaskManager_CreateThread(pstManager->hGeneralManager, pstTask);
		ERRIFGOTO(result, _EXIT);

		result = UKCPUGeneralTaskManager_ChangeState(pstManager->hGeneralManager, pstTask, TASK_STATE_RUNNING);
		ERRIFGOTO(result, _EXIT);

		result = UKCPUGeneralTaskManager_ActivateThread(pstManager->hGeneralManager, pstTask);
		ERRIFGOTO(result, _EXIT);

		result = waitUntilGeneralTaskIsStarted(pstManager->hGeneralManager, pstTask);
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKCPUTaskManager_ResumeTask(HCPUTaskManager hCPUTaskManager, int nTaskId)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUTaskManager *pstManager = NULL;
	STask *pstTask = NULL;
#ifdef ARGUMENT_CHECK
	if (IS_VALID_HANDLE(hCPUTaskManager, ID_UEM_CPU_TASK_MANAGER) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}

	if(nTaskId < 0) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
#endif
	pstManager = hCPUTaskManager;

	result = UKTask_GetTaskFromTaskId(nTaskId, &pstTask);
	ERRIFGOTO(result, _EXIT);

	if(pstTask->bStaticScheduled == TRUE && pstTask->pstSubGraph != NULL)
	{
		result = UKCPUCompositeTaskManager_ChangeState(pstManager->hCompositeManager, pstTask, TASK_STATE_RUNNING);
		ERRIFGOTO(result, _EXIT);

		result = UKCPUCompositeTaskManager_ActivateThread(pstManager->hCompositeManager, pstTask);
		ERRIFGOTO(result, _EXIT);
	}
	else if(pstTask->pstSubGraph != NULL)
	{
		// task with subgraph which is not SDF cannot be controlled
		if(pstTask->pstSubGraph->enType == GRAPH_TYPE_PROCESS_NETWORK)
		{
			ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
		}
		else
		{
			result = UKCPUTaskCommon_TraverseSubGraphTasks(pstTask, resumeDataflowSubgraphTask, pstManager->hGeneralManager);
			ERRIFGOTO(result, _EXIT);
		}
	}
	else
	{
		result = UKCPUGeneralTaskManager_ChangeState(pstManager->hGeneralManager, pstTask, TASK_STATE_RUNNING);
		ERRIFGOTO(result, _EXIT);

		result = UKCPUGeneralTaskManager_ActivateThread(pstManager->hGeneralManager, pstTask);
		ERRIFGOTO(result, _EXIT);
	}


	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKCPUTaskManager_StoppingTask(HCPUTaskManager hCPUTaskManager, int nTaskId)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUTaskManager *pstManager = NULL;
	STask *pstTask = NULL;
#ifdef ARGUMENT_CHECK
	if (IS_VALID_HANDLE(hCPUTaskManager, ID_UEM_CPU_TASK_MANAGER) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}

	if(nTaskId < 0) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
#endif

	pstManager = hCPUTaskManager;

	result = UKTask_GetTaskFromTaskId(nTaskId, &pstTask);
	ERRIFGOTO(result, _EXIT);

	if(pstTask->bStaticScheduled == TRUE && pstTask->pstSubGraph != NULL)
	{
		result = UKCPUCompositeTaskManager_ChangeState(pstManager->hCompositeManager, pstTask, TASK_STATE_STOPPING);
		ERRIFGOTO(result, _EXIT);

		result = UKCPUCompositeTaskManager_ActivateThread(pstManager->hCompositeManager, pstTask);
		ERRIFGOTO(result, _EXIT);
	}
	else if(pstTask->pstSubGraph != NULL)
	{
		// task with subgraph which is not SDF cannot be controlled
		if(pstTask->pstSubGraph->enType == GRAPH_TYPE_PROCESS_NETWORK)
		{
			ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
		}
		else
		{
			result = setMaximumTaskIterationInSubGraph(hCPUTaskManager, pstTask);
			ERRIFGOTO(result, _EXIT);

			result = UKCPUTaskCommon_TraverseSubGraphTasks(pstTask, stoppingDataflowSubgraphTask, pstManager->hGeneralManager);
			ERRIFGOTO(result, _EXIT);

			result = UKCPUTaskCommon_TraverseSubGraphTasks(pstTask, activateDataflowSubgraphTask, pstManager->hGeneralManager);
			ERRIFGOTO(result, _EXIT);
		}
	}
	else
	{
		result = UKTask_SetTargetIteration(pstTask, pstTask->nCurIteration, pstTask->nTaskId);
		ERRIFGOTO(result, _EXIT);

		result = UKCPUGeneralTaskManager_ChangeState(pstManager->hGeneralManager, pstTask, TASK_STATE_STOPPING);
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKCPUTaskManager_GetTaskState(HCPUTaskManager hCPUTaskManager, int nTaskId, EInternalTaskState *penTaskState)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUTaskManager *pstManager = NULL;
	STask *pstTask = NULL;
	ECPUTaskState enTaskState;
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

	result = UKTask_GetTaskFromTaskId(nTaskId, &pstTask);
	ERRIFGOTO(result, _EXIT);

	if(pstTask->bStaticScheduled == TRUE && pstTask->pstSubGraph != NULL)
	{
		result = UKCPUCompositeTaskManager_GetTaskState(pstManager->hCompositeManager, pstTask, &enTaskState);
		ERRIFGOTO(result, _EXIT);
	}
	else if(pstTask->pstSubGraph != NULL)
	{
		// task with subgraph which is not SDF cannot be controlled
		if(pstTask->pstSubGraph->enType == GRAPH_TYPE_PROCESS_NETWORK)
		{
			ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
		}
		else
		{
			result = getSubgraphTaskState(pstManager->hGeneralManager, pstTask, &enTaskState);
			ERRIFGOTO(result, _EXIT);
		}
	}
	else
	{
		result = UKCPUGeneralTaskManager_GetTaskState(pstManager->hGeneralManager, pstTask, &enTaskState);
		ERRIFGOTO(result, _EXIT);
	}

	if(enTaskState == TASK_STATE_RUNNING)
	{
		*penTaskState = INTERNAL_STATE_RUN;
	}
	else if(enTaskState == TASK_STATE_SUSPEND)
	{
		*penTaskState = INTERNAL_STATE_WAIT;
	}
	else if(enTaskState == TASK_STATE_STOPPING)
	{
		*penTaskState = INTERNAL_STATE_END;
	}
	else
	{
		*penTaskState = INTERNAL_STATE_STOP;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKCPUTaskManager_IsAllTaskStopped(HCPUTaskManager hCPUTaskManager, uem_bool *pbStopped)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUTaskManager *pstManager = NULL;
	uem_bool bCompositeTaskStopped = FALSE;
	uem_bool bGeneralTaskStopped = FALSE;
#ifdef ARGUMENT_CHECK
	if (IS_VALID_HANDLE(hCPUTaskManager, ID_UEM_CPU_TASK_MANAGER) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}

	IFVARERRASSIGNGOTO(pbStopped, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	pstManager = hCPUTaskManager;

	if(pstManager->hCompositeManager != NULL)
	{
		result = UKCPUCompositeTaskManager_CheckAllTaskStopped(pstManager->hCompositeManager, &bCompositeTaskStopped);
		ERRIFGOTO(result, _EXIT);
	}
	else
	{
		bCompositeTaskStopped = TRUE;
	}

	if(pstManager->hGeneralManager != NULL)
	{
		result = UKCPUGeneralTaskManager_CheckAllTaskStopped(pstManager->hGeneralManager, &bGeneralTaskStopped);
		ERRIFGOTO(result, _EXIT);
	}
	else
	{
		bGeneralTaskStopped = TRUE;
	}

	if(bCompositeTaskStopped == TRUE && bGeneralTaskStopped == TRUE)
	{
		*pbStopped = TRUE;
	}
	else
	{
		*pbStopped = FALSE;
	}

#endif
	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKCPUTaskManager_ChangeMappedCore(HCPUTaskManager hCPUTaskManager, int nTaskId, int nNewLocalId)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUTaskManager *pstManager = NULL;
	STask *pstTask = NULL;
#ifdef ARGUMENT_CHECK
	if (IS_VALID_HANDLE(hCPUTaskManager, ID_UEM_CPU_TASK_MANAGER) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}

	if(nTaskId < 0 || nNewLocalId < 0) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
#endif

	pstManager = hCPUTaskManager;

	result = UKTask_GetTaskFromTaskId(nTaskId, &pstTask);
	ERRIFGOTO(result, _EXIT);

	if(pstTask->enType != TASK_TYPE_COMPUTATIONAL && pstTask->enType != TASK_TYPE_CONTROL)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_CONTROL, _EXIT);
	}

	if(pstTask->pstSubGraph != NULL)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_CONTROL, _EXIT);
	}

	if(pstTask->bStaticScheduled == TRUE)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_CONTROL, _EXIT);
	}

	result = UKCPUGeneralTaskManager_ChangeMappedCore(pstManager->hGeneralManager, pstTask, nNewLocalId);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static uem_result changeMappedCoreByMappingSet(STask *pstTask, void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;
	HCPUGeneralTaskManager hGeneralTaskManager = NULL;
	char *pszMappingSet = NULL;
	uem_string_struct stCurrentMappingName;
	uem_string_struct stTargetMappingName;
	SMappedGeneralTaskInfo *pstGeneralTaskMappingInfo = NULL;
	int nNewLocalId = MAPPING_NOT_SPECIFIED;

	hGeneralTaskManager = ((SMappingSetUserData *)pUserData)->hGeneralTaskManager;
	pszMappingSet = ((SMappingSetUserData *)pUserData)->pszMappingSet;
	pstGeneralTaskMappingInfo = g_stMappingInfo.pstGeneralTaskMappingInfo;

	result = UCString_New(&stTargetMappingName, (char *)pszMappingSet, UEMSTRING_CONST);
	ERRIFGOTO(result, _EXIT);

	for(nLoop = 0; nLoop < g_stMappingInfo.nMappedGeneralTaskNum; nLoop++)
	{
		if(pstTask->nTaskId == pstGeneralTaskMappingInfo[nLoop].pstTask->nTaskId)
		{
			result = UCString_New(&stCurrentMappingName, (char *)pstGeneralTaskMappingInfo[nLoop].pszMappingSet, UEMSTRING_CONST);
			ERRIFGOTO(result, _EXIT);

			if(UCString_IsEqual(&stCurrentMappingName, &stTargetMappingName) == TRUE)
			{
				nNewLocalId = pstGeneralTaskMappingInfo[nLoop].nLocalId;

				result = UKCPUGeneralTaskManager_ChangeMappingSet(hGeneralTaskManager, pstGeneralTaskMappingInfo[nLoop].pstTask, pszMappingSet, nNewLocalId);
				ERRIFGOTO(result, _EXIT);
			}
		}
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKCPUTaskManager_ChangeMappingSet(HCPUTaskManager hCPUTaskManager, int nTaskId, const char *pszMappingSet)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUTaskManager *pstManager = NULL;
	STask *pstTask = NULL;
	SMappingSetUserData stMappingSetUserData;
#ifdef ARGUMENT_CHECK
	if (IS_VALID_HANDLE(hCPUTaskManager, ID_UEM_CPU_TASK_MANAGER) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}

	if(nTaskId < 0) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
#endif

	pstManager = hCPUTaskManager;

	result = UKTask_GetTaskFromTaskId(nTaskId, &pstTask);
	ERRIFGOTO(result, _EXIT);

	if(pstTask->enType != TASK_TYPE_COMPUTATIONAL && pstTask->enType != TASK_TYPE_CONTROL)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_CONTROL, _EXIT);
	}

	if(pstTask->bStaticScheduled == TRUE)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_CONTROL, _EXIT);
	}

	stMappingSetUserData.hGeneralTaskManager = pstManager->hGeneralManager;
	stMappingSetUserData.pszMappingSet = pszMappingSet;

	if(pstTask->pstSubGraph != NULL)
	{
		result = UKCPUTaskCommon_TraverseSubGraphTasks(pstTask, changeMappedCoreByMappingSet, (void *)&stMappingSetUserData);
		ERRIFGOTO(result, _EXIT);
	}
	else
	{
		result = changeMappedCoreByMappingSet(pstTask, (void *)&stMappingSetUserData);	
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKCPUTaskManager_GetCurrentMappingSet(HCPUTaskManager hCPUTaskManager, int nTaskId, int nBufferLen, char **ppszMappingSet)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SCPUTaskManager *pstManager = NULL;
	STask *pstTask = NULL;
#ifdef ARGUMENT_CHECK
	if (IS_VALID_HANDLE(hCPUTaskManager, ID_UEM_CPU_TASK_MANAGER) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}

	if(nTaskId < 0) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
#endif

	pstManager = hCPUTaskManager;

	result = UKTask_GetTaskFromTaskId(nTaskId, &pstTask);
	ERRIFGOTO(result, _EXIT);

	if(pstTask->enType != TASK_TYPE_COMPUTATIONAL && pstTask->enType != TASK_TYPE_CONTROL)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_CONTROL, _EXIT);
	}

	if(pstTask->bStaticScheduled == TRUE)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_CONTROL, _EXIT);
	}

	if(pstTask->pstSubGraph != NULL)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_CONTROL, _EXIT);
	}

	result = UKCPUGeneralTaskManager_GetCurrentMappingSet(pstManager->hGeneralManager, pstTask, nBufferLen, ppszMappingSet);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
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

	if(pstManager->hCompositeManager != NULL)
	{
		result = UKCPUCompositeTaskManager_Destroy(&(pstManager->hCompositeManager));
		ERRIFGOTO(result, _EXIT);
	}

	if(pstManager->hGeneralManager != NULL)
	{
		result = UKCPUGeneralTaskManager_Destroy(&(pstManager->hGeneralManager));
		ERRIFGOTO(result, _EXIT);
	}

	result = UCThreadMutex_Destroy(&(pstManager->hMutex));
	ERRIFGOTO(result, _EXIT);

	SAFEMEMFREE(pstManager);

	*phCPUTaskManager = NULL;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


