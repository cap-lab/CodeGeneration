/*
 * UKModeTransitionMachineController.c
 *
 *  Created on: 2019. 9. 5.
 *      Author: jej
 */


#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <uem_data.h>

#include <UKChannel_internal.h>
#include <UKTask_internal.h>
#include <UKModeTransition.h>
#include <UKModelController.h>
#include <UKCPUGeneralTaskManager.h>
#include <UKCPUCompositeTaskManager.h>

struct _SModeTransitionSetEventCheck {
	int nModeId;
	void *pCurrentThreadHandle;
	int nTargetIteration;
};

struct _SModeTransitionSuspendCheck {
	uem_bool bAllSuspended;
	void *pTaskHandle;
	void *pThreadHandle;
};

struct _SModeTransitionGeneralSetEventCheck {
	STaskGraph *pstParentTaskGraph;
	void *pCallerTask;
	int nNewStartIteration;
	int nPrevModeIndex;
	int nNewModeIndex;
	uem_bool bModeChanged;
};


static uem_result traverseAndSetEventToTemporarySuspendedTask(void *pCurrentTaskHandle, void *pCurrentThreadHandle, void *pUserData, OUT uem_bool *pbActivateThread)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SModeTransitionSetEventCheck *pstNewModeData = NULL;
	ECPUTaskState enThreadState;
	int nThreadModeId = INVALID_MODE_ID;
	int nThroughputConstraint = 0;
	int nThreadThroughputConstraint = 0;

	pstNewModeData = (struct _SModeTransitionSetEventCheck *) pUserData;

	result = UKCPUCompositeTaskManagerCB_GetThreadState(pCurrentThreadHandle, &enThreadState);
	ERRIFGOTO(result, _EXIT);

	result = UKCPUCompositeTaskManagerCB_GetThreadModeId(pCurrentThreadHandle, &nThreadModeId);
	ERRIFGOTO(result, _EXIT);

	result = UKCPUCompositeTaskManagerCB_GetThreadTargetThroughput(pstNewModeData->pCurrentThreadHandle, &nThroughputConstraint);
	ERRIFGOTO(result, _EXIT);

	result = UKCPUCompositeTaskManagerCB_GetThreadTargetThroughput(pCurrentThreadHandle, &nThreadThroughputConstraint);
	ERRIFGOTO(result, _EXIT);


	if(enThreadState == TASK_STATE_SUSPEND && pstNewModeData->nModeId == nThreadModeId &&
			nThroughputConstraint == nThreadThroughputConstraint)
	{
		result = UKCPUCompositeTaskManagerCB_SetThreadState(pCurrentThreadHandle, TASK_STATE_RUNNING);
		ERRIFGOTO(result, _EXIT);

		*pbActivateThread = TRUE;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result traverseAndCheckDifferentModeIsAllSuspended(void *pCurrentTaskHandle, void *pCurrentThreadHandle, void *pUserData, OUT uem_bool *pbActivateThread)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SModeTransitionSuspendCheck *pstUserData = NULL;
	ECPUTaskState enThreadState;

	pstUserData = (struct _SModeTransitionSuspendCheck *) pUserData;

	result = UKCPUCompositeTaskManagerCB_GetThreadState(pCurrentThreadHandle, &enThreadState);
	ERRIFGOTO(result, _EXIT);

	if(enThreadState != TASK_STATE_SUSPEND && enThreadState != TASK_STATE_STOP)
	{
		//UEM_DEBUG_PRINT("not end task: %p (%d), mode: %d, proc: %d\n", pstTaskThread, pstTaskThread->enTaskState, pstTaskThread->nModeId, pstTaskThread->nProcId);
		pstUserData->bAllSuspended = FALSE;
	}
	else
	{
		//UEM_DEBUG_PRINT("end task: %p (%d), mode: %d, proc: %d\n", pstTaskThread, pstTaskThread->enTaskState, pstTaskThread->nModeId, pstTaskThread->nProcId);
	}

	*pbActivateThread = FALSE;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKModeTransitionMachineController_HandleModelComposite(STaskGraph *pstGraph, void *pCurrentTaskHandle, void *pCurrentThreadHandle)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstCurrentTask = NULL;
	int nCurModeIndex = 0;
	struct _SModeTransitionSuspendCheck stUserData;
	struct _SModeTransitionSetEventCheck stNewModeData;
	EModeState enModeState;
	ECPUTaskState enTaskState;
	int nIteration = 0;
	SModeTransitionController *pstController = NULL;

	pstController = (SModeTransitionController *) pstGraph->pController;
	pstCurrentTask = pstGraph->pstParentTask;

	result = UCThreadMutex_Lock(pstCurrentTask->hMutex);
	ERRIFGOTO(result, _EXIT);

	result = UKCPUCompositeTaskManagerCB_GetTaskState(pCurrentTaskHandle, &enTaskState);
	ERRIFGOTO(result, _EXIT_LOCK);

	enModeState = UKModeTransition_GetModeStateInternal(pstController->pstMTMInfo);

	if(enModeState == MODE_STATE_TRANSITING)
	{
		result = UKCPUCompositeTaskManagerCB_GetThreadIteration(pCurrentThreadHandle, &nIteration);
		ERRIFGOTO(result, _EXIT_LOCK);

		UKModeTransition_UpdateModeStateInternal(pstController->pstMTMInfo, MODE_STATE_NORMAL, nIteration);

		if(enTaskState == TASK_STATE_RUNNING)
		{
			result = UKCPUCompositeTaskManagerCB_SetThreadState(pCurrentThreadHandle, TASK_STATE_SUSPEND);
			ERRIFGOTO(result, _EXIT_LOCK);

			nCurModeIndex = pstController->pstMTMInfo->nCurModeIndex;

			stNewModeData.pCurrentThreadHandle = pCurrentThreadHandle;
			stNewModeData.nModeId = pstController->pstMTMInfo->astModeMap[nCurModeIndex].nModeId;

			result = UKCPUCompositeTaskManagerCB_TraverseThreadsInTask(pCurrentTaskHandle, traverseAndSetEventToTemporarySuspendedTask, &stNewModeData);
			ERRIFGOTO(result, _EXIT_LOCK);
		}

		result = UCThreadMutex_Unlock(pstCurrentTask->hMutex);
		ERRIFGOTO(result, _EXIT);

		result = UKCPUCompositeTaskManagerCB_HandleControlRequest(pCurrentTaskHandle);
		ERRIFGOTO(result, _EXIT);

		result = UCThreadMutex_Lock(pstCurrentTask->hMutex);
		ERRIFGOTO(result, _EXIT);
	}
	else
	{
		stUserData.bAllSuspended = TRUE;
		stUserData.pTaskHandle = pCurrentTaskHandle;
		stUserData.pThreadHandle = pCurrentThreadHandle;

		result = UKCPUCompositeTaskManagerCB_SetThreadState(pCurrentThreadHandle, TASK_STATE_SUSPEND);
		ERRIFGOTO(result, _EXIT_LOCK);

		result = UKCPUCompositeTaskManagerCB_TraverseThreadsInTask(pCurrentTaskHandle, traverseAndCheckDifferentModeIsAllSuspended, &stUserData);
		ERRIFGOTO(result, _EXIT_LOCK);

		if(stUserData.bAllSuspended == TRUE)
		{
			result = UKCPUCompositeTaskManagerCB_GetThreadIteration(pCurrentThreadHandle, &nIteration);
			ERRIFGOTO(result, _EXIT_LOCK);

			UKModeTransition_UpdateModeStateInternal(pstController->pstMTMInfo, MODE_STATE_TRANSITING, nIteration);

			if(enTaskState == TASK_STATE_RUNNING)
			{
				result = UCThreadMutex_Unlock(pstCurrentTask->hMutex);
				ERRIFGOTO(result, _EXIT);

				result = UKCPUCompositeTaskManagerCB_ChangeTaskState(pCurrentTaskHandle, TASK_STATE_RUNNING);
				ERRIFGOTO(result, _EXIT);

				result = UCThreadMutex_Lock(pstCurrentTask->hMutex);
				ERRIFGOTO(result, _EXIT);

				result = UKCPUCompositeTaskManagerCB_WakeUpTask(pCurrentTaskHandle);
				ERRIFGOTO(result, _EXIT_LOCK);
			}
			//UEM_DEBUG_PRINT("mode state: MODE_STATE_NORMAL to MODE_STATE_TRANSITING\n");

			result = UCThreadMutex_Unlock(pstCurrentTask->hMutex);
			ERRIFGOTO(result, _EXIT);

			result = UKCPUCompositeTaskManagerCB_HandleControlRequest(pCurrentTaskHandle);
			ERRIFGOTO(result, _EXIT);

			result = UCThreadMutex_Lock(pstCurrentTask->hMutex);
			ERRIFGOTO(result, _EXIT);
		}
	}

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
	UCThreadMutex_Unlock(pstCurrentTask->hMutex);
_EXIT:
	return result;
}


uem_result UKModeTransitionMachineController_Clear(STaskGraph *pstTaskGraph)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SModeTransitionController *pstController = NULL;

#if defined(ARGUMENT_CHECK)
	IFVARERRASSIGNGOTO(pstTaskGraph, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	IFVARERRASSIGNGOTO(pstTaskGraph->pstParentTask, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	pstController = (SModeTransitionController *) pstTaskGraph->pController;

	result = UKModeTransition_Clear(pstController->pstMTMInfo);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


// if(*penState == TASK_STATE_SUSPEND) => pstTaskThread->bSuspended = TRUE;
uem_result UKModeTransitionMachineController_ChangeTaskThreadState(STaskGraph *pstGraph, void *pCurrentTaskHandle, void *pCurrentThreadHandle, ECPUTaskState enTargetState, OUT ECPUTaskState *penState)
{
	uem_result result = ERR_UEM_UNKNOWN;
	EModeState enModeState;
	int nCurModeId = 0;
	int nThreadModeId = 0;
	int nCurModeIndex = INVALID_MODE_ID;
	uem_bool bHasSourceTask = FALSE;
	SModeTransitionController *pstController = NULL;

	pstController = (SModeTransitionController *) pstGraph->pController;

	result = UCThreadMutex_Lock(pstGraph->pstParentTask->hMutex);
	ERRIFGOTO(result, _EXIT);
	nCurModeIndex = pstController->pstMTMInfo->nCurModeIndex;
	result = UCThreadMutex_Unlock(pstGraph->pstParentTask->hMutex);
	ERRIFGOTO(result, _EXIT);
	nCurModeId = pstController->pstMTMInfo->astModeMap[nCurModeIndex].nModeId;

	result = UKCPUCompositeTaskManagerCB_GetThreadModeId(pCurrentThreadHandle, &nThreadModeId);
	ERRIFGOTO(result, _EXIT);

	if(nThreadModeId == nCurModeId)
	{
		enModeState = UKModeTransition_GetModeStateInternal(pstController->pstMTMInfo);

		if(enModeState == MODE_STATE_TRANSITING && enTargetState == TASK_STATE_RUNNING)
		{
			result = UKCPUCompositeTaskManagerCB_HasSourceTask(pCurrentThreadHandle, &bHasSourceTask);
			ERRIFGOTO(result, _EXIT);

			if(bHasSourceTask == TRUE)
			{
				*penState = TASK_STATE_RUNNING;
			}
			else
			{
				*penState = TASK_STATE_SUSPEND;
			}
		}
		else
		{
			*penState = enTargetState;
		}
	}
	else // Different modes needs to be suspended when specific mode becomes running
	{
		if(enTargetState == TASK_STATE_RUNNING)
		{
			*penState = TASK_STATE_SUSPEND;
		}
		else if(enTargetState == TASK_STATE_STOPPING)
		{
			*penState = enTargetState;
		}
		else
		{
			// Maintain the state if the task mode is not matched to current mode ID
			UEMASSIGNGOTO(result, ERR_UEM_SKIP_THIS, _EXIT);
		}
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result traverseAndSetEventToStoppingTasks(STask *pstTask, void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	HThreadMutex hTaskGraphLock = NULL;
	HCPUGeneralTaskManager hManager = NULL;
	void *pCurrentTaskHandle = NULL;
	STask *pstCallerTask = NULL;

	pCurrentTaskHandle = pUserData;

	result = UKCPUGeneralTaskManagerCB_GetCurrentTaskStructure(pCurrentTaskHandle, &pstCallerTask);
	ERRIFGOTO(result, _EXIT);

	result = UKCPUGeneralTaskManagerCB_GetTaskGraphLock(pCurrentTaskHandle, &hTaskGraphLock);
	ERRIFGOTO(result, _EXIT);

	result = UKCPUGeneralTaskManagerCB_GetManagerHandle(pCurrentTaskHandle, &hManager);
	ERRIFGOTO(result, _EXIT);

	if(pstTask->nTaskId != pstCallerTask->nTaskId)
	{
		result = UKCPUGeneralTaskManager_RequestTaskState(hManager, pstTask, TASK_STATE_STOPPING);
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result traverseAndSetEventToTemporarySuspendedMTMTask(STask *pstTask, void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SModeTransitionGeneralSetEventCheck *pstNewModeData = NULL;
	HCPUGeneralTaskManager hManager = NULL;
	ECPUTaskState enState;
	char *pszOldModeName = NULL;
	char *pszCurModeName = NULL;
	uem_bool bCurrentPortAvailable = FALSE;
	STask *pstCallerTask = NULL;
	HThreadMutex hTaskGraphLock = NULL;
	SModeTransitionController *pstController = NULL;
	int nConvertedStartIteration = 0;
	STask *pstMTMSubTask = NULL;

	pstNewModeData = (struct _SModeTransitionGeneralSetEventCheck *) pUserData;

	pstController = (SModeTransitionController *) pstNewModeData->pstParentTaskGraph->pController;

	result = UKCPUGeneralTaskManagerCB_GetManagerHandle(pstNewModeData->pCallerTask, &hManager);
	ERRIFGOTO(result, _EXIT);

	result = UKCPUGeneralTaskManagerCB_GetCurrentTaskStructure(pstNewModeData->pCallerTask, &pstCallerTask);
	ERRIFGOTO(result, _EXIT);

	result = UKCPUGeneralTaskManagerCB_GetTaskGraphLock(pstNewModeData->pCallerTask, &hTaskGraphLock);
	ERRIFGOTO(result, _EXIT);

	if(pstCallerTask->nTaskId != pstTask->nTaskId)
	{
		result = UKCPUGeneralTaskManager_GetTaskState(hManager, pstTask, &enState);
		ERRIFGOTO(result, _EXIT);

		if(enState == TASK_STATE_SUSPEND)
		{
			pszCurModeName = pstController->pstMTMInfo->astModeMap[pstNewModeData->nNewModeIndex].pszModeName;

			pstMTMSubTask = pstTask;

			while(pstMTMSubTask->pstParentGraph != pstNewModeData->pstParentTaskGraph)
			{
				pstMTMSubTask = pstMTMSubTask->pstParentGraph->pstParentTask;
			}

			bCurrentPortAvailable = UKChannel_IsPortRateAvailableTask(pstMTMSubTask->nTaskId, pszCurModeName);
			//UEM_DEBUG_PRINT("task: %s, available: %d, mode_name: %s\n", pstTask->pszTaskName, bCurrentPortAvailable, pszCurModeName);

			if(pstNewModeData->bModeChanged == TRUE)
			{
				pszOldModeName = pstController->pstMTMInfo->astModeMap[pstNewModeData->nPrevModeIndex].pszModeName;

				if(UKChannel_IsPortRateAvailableTask(pstMTMSubTask->nTaskId, pszOldModeName) == FALSE &&
					bCurrentPortAvailable == TRUE)
				{
					//UEM_DEBUG_PRINT("new task: %s, previous_iteration: %d, new_iteration: %d\n", pstTask->pszTaskName, pstTask->nCurIteration, pstNewModeData->nNewStartIteration);

					result = UKTask_GetIterationNumberBasedOnTargetParentTaskId(pstTask, pstNewModeData->nNewStartIteration,
														pstNewModeData->pstParentTaskGraph->pstParentTask->nTaskId,
														&nConvertedStartIteration);
					ERRIFGOTO(result, _EXIT);

					result = UCThreadMutex_Lock(pstTask->hMutex);
					ERRIFGOTO(result, _EXIT);

					pstTask->nCurIteration = nConvertedStartIteration;

					result = UKTask_UpdateAllSubGraphCurrentIteration(pstNewModeData->pstParentTaskGraph, pstTask, pstNewModeData->nNewStartIteration);
					UCThreadMutex_Unlock(pstTask->hMutex);
					ERRIFGOTO(result, _EXIT);
				}
			}

			if(bCurrentPortAvailable == TRUE)
			{
				result = UKCPUGeneralTaskManager_RequestTaskState(hManager, pstTask, TASK_STATE_RUNNING);
				ERRIFGOTO(result, _EXIT);
			}
			else
			{
				result = UKCPUGeneralTaskManager_RequestTaskState(hManager, pstTask, TASK_STATE_NONE);
				ERRIFGOTO(result, _EXIT);
			}
		}
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}



uem_result UKModeTransitionMachineController_ChangeSubGraphTaskState(STaskGraph *pstGraph, void *pCurrentTaskHandle, void *pCurrentThreadHandle, ECPUTaskState enTargetState, OUT ECPUTaskState *penState)
{
	uem_result result = ERR_UEM_UNKNOWN;
	char *pszModeName = NULL;
	int nCurModeIndex = 0;
	uem_bool bIsSourceTask = FALSE;
	STask *pstCurrentTask = NULL;
	ECPUTaskState enCurrentTaskState = TASK_STATE_STOP;
	SModeTransitionController *pstController = NULL;
	int nConvertedIteration = 0;

	pstController = (SModeTransitionController *) pstGraph->pController;

	result = UKCPUGeneralTaskManagerCB_IsSourceTask(pCurrentTaskHandle, &bIsSourceTask);
	ERRIFGOTO(result, _EXIT);

	result = UKCPUGeneralTaskManagerCB_GetCurrentTaskStructure(pCurrentTaskHandle, &pstCurrentTask);
	ERRIFGOTO(result, _EXIT);

	// change task state to suspend when the target task is included in MTM task graph and is not a source task.
	if(enTargetState == TASK_STATE_RUNNING &&
	UKModeTransition_GetModeStateInternal(pstController->pstMTMInfo) == MODE_STATE_TRANSITING &&
	bIsSourceTask == FALSE)
	{
		enTargetState = TASK_STATE_SUSPEND;
	}

	result = UKCPUGeneralTaskManagerCB_GetCurrentTaskState(pCurrentTaskHandle, &enCurrentTaskState);
	ERRIFGOTO(result, _EXIT);

	if(enCurrentTaskState == TASK_STATE_SUSPEND && enTargetState == TASK_STATE_STOPPING)
	{
		result = UKTask_ConvertIterationToUpperTaskGraphBase(pstCurrentTask, pstGraph, &nConvertedIteration);
		ERRIFGOTO(result, _EXIT);

		result = UKModeTransition_GetCurrentModeIndexByIteration(pstController->pstMTMInfo, nConvertedIteration, &nCurModeIndex);
		if(result == ERR_UEM_NOT_FOUND || nCurModeIndex == INVALID_MODE_ID)
		{
			pstCurrentTask->nCurIteration = pstCurrentTask->nTargetIteration;
		}
		else if(result == ERR_UEM_NOERROR)
		{
			pszModeName = pstController->pstMTMInfo->astModeMap[nCurModeIndex].pszModeName;
			if(UKChannel_IsPortRateAvailableTask(pstCurrentTask->nTaskId, pszModeName) == FALSE)
			{
				pstCurrentTask->nCurIteration = pstCurrentTask->nTargetIteration;
			}
		}
	}

	*penState = enTargetState;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static uem_result updateCurrentIteration(STaskGraph *pstGraph, SModeTransitionMachine *pstMTMInfo, int nMTMCurrentIteration, STask *pstTask, OUT uem_bool *pbSuspended)
{
	uem_result result = ERR_UEM_UNKNOWN;

	int nModeIndex;
	char *pszCurModeName = NULL;
	uem_bool bCurrentPortAvailable = FALSE;
	int nConvertedIteration = 0;
	int nNewIterationNumber = 0;
	STask *pstMTMSubTask = NULL;
	uem_bool bNeedSuspended = FALSE;
	int nNewIteration = 0;

	result = UKTask_ConvertIterationToUpperTaskGraphBase(pstTask,  pstGraph, &nConvertedIteration);
	ERRIFGOTO(result, _EXIT);

	result = UKModeTransition_GetCurrentModeIndexByIteration(pstMTMInfo, nConvertedIteration, &nModeIndex);
	ERRIFGOTO(result, _EXIT);

	while(bCurrentPortAvailable == FALSE)
	{
		pszCurModeName = pstMTMInfo->astModeMap[nModeIndex].pszModeName;

		pstMTMSubTask = pstTask;

		while(pstMTMSubTask->pstParentGraph != pstGraph)
		{
			pstMTMSubTask = pstMTMSubTask->pstParentGraph->pstParentTask;
		}

		bCurrentPortAvailable = UKChannel_IsPortRateAvailableTask(pstMTMSubTask->nTaskId, pszCurModeName);
		if(bCurrentPortAvailable == FALSE)
		{
			result = UKModeTransition_GetNextModeStartIndexByIteration(pstMTMInfo, nConvertedIteration,
																		&nModeIndex, &nNewIterationNumber);
			if(result == ERR_UEM_NO_DATA)
			{
				bNeedSuspended = TRUE;
				break;
			}
			ERRIFGOTO(result, _EXIT);

			result = UKTask_GetIterationNumberBasedOnTargetParentTaskId(pstTask, nNewIterationNumber,
																	pstGraph->pstParentTask->nTaskId, &nNewIteration);
			ERRIFGOTO(result, _EXIT);

			result = UCThreadMutex_Lock(pstTask->hMutex);
			ERRIFGOTO(result, _EXIT);
			pstTask->nCurIteration = nNewIteration;
			result = UKTask_UpdateAllSubGraphCurrentIteration(pstGraph, pstTask, nNewIterationNumber);
			UCThreadMutex_Unlock(pstTask->hMutex);
			ERRIFGOTO(result, _EXIT);
		}
	}

	*pbSuspended = bNeedSuspended;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}



uem_result UKModeTransitionMachineController_HandleModelGeneralDuringStopping(STaskGraph *pstGraph, void *pCurrentTaskHandle, void *pCurrentThreadHandle)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstCurrentTask = NULL;
	SModeTransitionController *pstController = NULL;
	uem_bool bNeedSuspend = FALSE;
	uem_bool bIsSourceTask = FALSE;
	int nConvertedIteration = 0;

	pstController = (SModeTransitionController *) pstGraph->pController;

	result = UKCPUGeneralTaskManagerCB_GetCurrentTaskStructure(pCurrentTaskHandle, &pstCurrentTask);
	ERRIFGOTO(result, _EXIT);

    result = UKCPUGeneralTaskManagerCB_IsSourceTask(pCurrentTaskHandle, &bIsSourceTask);
    ERRIFGOTO(result, _EXIT);

    if(bIsSourceTask == TRUE && pstCurrentTask->pstParentGraph == pstGraph)
    {
		result = UKTask_ConvertIterationToUpperTaskGraphBase(pstCurrentTask, pstGraph, &nConvertedIteration);
		ERRIFGOTO(result, _EXIT);

		UKModeTransition_UpdateModeStateInternal(pstController->pstMTMInfo, MODE_STATE_NORMAL, nConvertedIteration-1);

		pstController->stCommon.nCurrentIteration = nConvertedIteration;

        result = UKCPUTaskCommon_TraverseSubGraphTasks(pstGraph->pstParentTask, traverseAndSetEventToStoppingTasks, pCurrentTaskHandle);
        ERRIFGOTO(result, _EXIT);
    }
    else
    {
    	result = updateCurrentIteration(pstGraph, pstController->pstMTMInfo, pstController->stCommon.nCurrentIteration, pstCurrentTask, &bNeedSuspend);
    	ERRIFGOTO(result, _EXIT);

		result = UKTask_ConvertIterationToUpperTaskGraphBase(pstCurrentTask, pstGraph, &nConvertedIteration);
		ERRIFGOTO(result, _EXIT);

    	if(pstController->stCommon.nCurrentIteration <= nConvertedIteration || bNeedSuspend == TRUE)
    	{
    		UEMASSIGNGOTO(result, ERR_UEM_ALREADY_DONE, _EXIT);
    	}
    }

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result handleModeTransitionInGeneralTasks(STaskGraph *pstGraph, void *pCurrentTaskHandle, void *pCurrentThreadHandle, uem_bool *pbActivated)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SModeTransitionGeneralSetEventCheck stNewModeData;
	EModeState enModeState;
	uem_bool bIsSourceTask = FALSE;
	STask *pstCurrentTask = NULL;
	SModeTransitionController *pstController = NULL;
	int nConvertedIteration = 0;
	uem_bool bNeedSuspend = FALSE;
	ECPUTaskState enTaskState = TASK_STATE_STOP;
	uem_bool bRestarted = FALSE;
	uem_bool bResumedByControl = FALSE;

	pstController = (SModeTransitionController *) pstGraph->pController;

	result = UKCPUGeneralTaskManagerCB_IsSourceTask(pCurrentTaskHandle, &bIsSourceTask);
	ERRIFGOTO(result, _EXIT);

	result = UKCPUGeneralTaskManagerCB_GetCurrentTaskStructure(pCurrentTaskHandle, &pstCurrentTask);
	ERRIFGOTO(result, _EXIT);

	result = UKTask_ConvertIterationToUpperTaskGraphBase(pstCurrentTask, pstGraph, &nConvertedIteration);
	ERRIFGOTO(result, _EXIT);

	if(bIsSourceTask == TRUE && pstCurrentTask->pstParentGraph == pstGraph)
	{
		result = UKCPUGeneralTaskManagerCB_GetRestarted(pCurrentThreadHandle, &bRestarted);
		ERRIFGOTO(result, _EXIT);

		result = UKCPUGeneralTaskManagerCB_IsResumedByControl(pCurrentTaskHandle, &bResumedByControl);
		ERRIFGOTO(result, _EXIT);

		if(bRestarted == FALSE || bResumedByControl == TRUE)
		{
			pstController->pstMTMInfo->fnTransition(pstController->pstMTMInfo);

			enModeState = UKModeTransition_GetModeStateInternal(pstController->pstMTMInfo);

			stNewModeData.nPrevModeIndex = pstController->pstMTMInfo->nCurModeIndex;

			if(enModeState == MODE_STATE_TRANSITING)
			{
				enModeState = UKModeTransition_UpdateModeStateInternal(pstController->pstMTMInfo, MODE_STATE_NORMAL, nConvertedIteration-1);
				stNewModeData.bModeChanged = TRUE;
			}
			else
			{
				stNewModeData.bModeChanged = FALSE;
			}

			stNewModeData.nNewModeIndex = pstController->pstMTMInfo->nCurModeIndex;

			stNewModeData.pCallerTask = pCurrentTaskHandle;
			stNewModeData.nNewStartIteration = nConvertedIteration-1;
			stNewModeData.pstParentTaskGraph = pstGraph;

			pstController->stCommon.nCurrentIteration = nConvertedIteration;

			result = UKCPUTaskCommon_TraverseSubGraphTasks(pstGraph->pstParentTask, traverseAndSetEventToTemporarySuspendedMTMTask, &stNewModeData);
			ERRIFGOTO(result, _EXIT);

			*pbActivated = TRUE;
		}
	}
	else
	{

		result = UKCPUGeneralTaskManagerCB_GetCurrentTaskState(pCurrentTaskHandle, &enTaskState);
		ERRIFGOTO(result, _EXIT);

		if(enTaskState == TASK_STATE_RUNNING) 
		{

			if(pstController->stCommon.nCurrentIteration <= nConvertedIteration)
			{
				result = UKCPUGeneralTaskManagerCB_ChangeTaskState(pCurrentTaskHandle, TASK_STATE_SUSPEND);
				ERRIFGOTO(result, _EXIT);
			}
			else
			{
				result = updateCurrentIteration(pstGraph, pstController->pstMTMInfo, pstController->stCommon.nCurrentIteration, pstCurrentTask, &bNeedSuspend);
				ERRIFGOTO(result, _EXIT);

				result = UKTask_ConvertIterationToUpperTaskGraphBase(pstCurrentTask, pstGraph, &nConvertedIteration);
				ERRIFGOTO(result, _EXIT);

				if(pstController->stCommon.nCurrentIteration <= nConvertedIteration || bNeedSuspend == TRUE)
				{
					result = UKCPUGeneralTaskManagerCB_ChangeTaskState(pCurrentTaskHandle, TASK_STATE_SUSPEND);
					ERRIFGOTO(result, _EXIT);
				}
			}
		}
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKModeTransitionMachineController_HandleModelGeneral(STaskGraph *pstGraph, void *pCurrentTaskHandle, void *pCurrentThreadHandle)
{
	uem_result result = ERR_UEM_UNKNOWN;
	uem_bool bFunctionCalled = FALSE;
	ECPUTaskState enTaskState = TASK_STATE_STOP;
	uem_bool bRestarted = FALSE;
	uem_bool bActivated = FALSE;


	result = UKCPUGeneralTaskManagerCB_GetFunctionCalled(pCurrentThreadHandle, &bFunctionCalled);
	ERRIFGOTO(result, _EXIT);

	result = UKCPUGeneralTaskManagerCB_GetCurrentTaskState(pCurrentTaskHandle, &enTaskState);
	ERRIFGOTO(result, _EXIT);

	result = UKCPUGeneralTaskManagerCB_GetRestarted(pCurrentThreadHandle, &bRestarted);
	ERRIFGOTO(result, _EXIT);

	if((bFunctionCalled == TRUE || bRestarted == TRUE) && enTaskState == TASK_STATE_RUNNING)
	{
		result = handleModeTransitionInGeneralTasks(pstGraph, pCurrentTaskHandle, pCurrentThreadHandle, &bActivated);
		ERRIFGOTO(result, _EXIT);
	}

	if(bActivated == TRUE)
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

