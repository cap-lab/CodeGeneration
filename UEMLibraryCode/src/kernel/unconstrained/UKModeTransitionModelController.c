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

	pstCurrentTask = pstGraph->pstParentTask;

	result = UCThreadMutex_Lock(pstCurrentTask->hMutex);
	ERRIFGOTO(result, _EXIT);

	result = UKCPUCompositeTaskManagerCB_GetTaskState(pCurrentTaskHandle, &enTaskState);
	ERRIFGOTO(result, _EXIT_LOCK);

	enModeState = UKModeTransition_GetModeStateInternal(pstCurrentTask->pstMTMInfo);

	if(enModeState == MODE_STATE_TRANSITING)
	{
		result = UKCPUCompositeTaskManagerCB_GetThreadIteration(pCurrentThreadHandle, &nIteration);
		ERRIFGOTO(result, _EXIT_LOCK);

		UKModeTransition_UpdateModeStateInternal(pstCurrentTask->pstMTMInfo, MODE_STATE_NORMAL, nIteration);

		if(enTaskState == TASK_STATE_RUNNING)
		{
			result = UKCPUCompositeTaskManagerCB_SetThreadState(pCurrentThreadHandle, TASK_STATE_SUSPEND);
			ERRIFGOTO(result, _EXIT_LOCK);

			nCurModeIndex = pstCurrentTask->pstMTMInfo->nCurModeIndex;

			stNewModeData.pCurrentThreadHandle = pCurrentThreadHandle;
			stNewModeData.nModeId = pstCurrentTask->pstMTMInfo->astModeMap[nCurModeIndex].nModeId;

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

			UKModeTransition_UpdateModeStateInternal(pstCurrentTask->pstMTMInfo, MODE_STATE_TRANSITING, nIteration);

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

uem_result UKModeTransitionMachineController_GetTaskIterationIndex(STask *pstMTMTask, int nCurrentIteration, int OUT *pnIndex)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nModeNum;
	int nModeId;
	int nCurModeIndex = INVALID_ARRAY_INDEX;
	int nLoop = 0;
	int nIndex = 0;

	if(pstMTMTask->pstMTMInfo == NULL)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
	}

	nModeNum = pstMTMTask->pstMTMInfo->nNumOfModes;

	if(pstMTMTask->bStaticScheduled == TRUE)
	{
		nCurModeIndex = pstMTMTask->pstMTMInfo->nCurModeIndex;
	}
	else
	{
		result = UKModeTransition_GetCurrentModeIndexByIteration(pstMTMTask->pstMTMInfo, nCurrentIteration, &nCurModeIndex);
		ERRIFGOTO(result, _EXIT);
	}

	//UEM_DEBUG_PRINT("nCurModeIndex: pstTask: %s %d\n", pstTask->pszTaskName, nCurModeIndex);

	nModeId = pstMTMTask->pstMTMInfo->astModeMap[nCurModeIndex].nModeId;

	for(nLoop  = 0 ; nLoop < nModeNum ; nLoop++)
	{
		if(pstMTMTask->astTaskIteration[nLoop].nModeId == nModeId)
		{
			nIndex = nLoop;
			break;
		}
	}

	if(nLoop == nModeNum)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
	}

	*pnIndex = nIndex;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKModeTransitionMachineController_Clear(STaskGraph *pstTaskGraph)
{
	uem_result result = ERR_UEM_UNKNOWN;
#if defined(ARGUMENT_CHECK)
	IFVARERRASSIGNGOTO(pstTaskGraph, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	IFVARERRASSIGNGOTO(pstTaskGraph->pstParentTask, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	result = UKModeTransition_Clear(pstTaskGraph->pstParentTask->pstMTMInfo);
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

	result = UCThreadMutex_Lock(pstGraph->pstParentTask->hMutex);
	ERRIFGOTO(result, _EXIT);
	nCurModeIndex = pstGraph->pstParentTask->pstMTMInfo->nCurModeIndex;
	result = UCThreadMutex_Unlock(pstGraph->pstParentTask->hMutex);
	ERRIFGOTO(result, _EXIT);
	nCurModeId = pstGraph->pstParentTask->pstMTMInfo->astModeMap[nCurModeIndex].nModeId;

	result = UKCPUCompositeTaskManagerCB_GetThreadModeId(pCurrentThreadHandle, &nThreadModeId);
	ERRIFGOTO(result, _EXIT);

	if(nThreadModeId == nCurModeId)
	{
		enModeState = UKModeTransition_GetModeStateInternal(pstGraph->pstParentTask->pstMTMInfo);

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

	pstNewModeData = (struct _SModeTransitionGeneralSetEventCheck *) pUserData;

	result = UKCPUGeneralTaskManagerCB_GetManagerHandle(pstNewModeData->pCallerTask, &hManager);
	ERRIFGOTO(result, _EXIT);

	result = UKCPUGeneralTaskManagerCB_GetCurrentTaskStructure(pstNewModeData->pCallerTask, &pstCallerTask);
	ERRIFGOTO(result, _EXIT);

	result = UKCPUGeneralTaskManagerCB_GetTaskGraphLock(pstNewModeData->pCallerTask, &hTaskGraphLock);
	ERRIFGOTO(result, _EXIT);

	if(pstCallerTask->nTaskId != pstTask->nTaskId)
	{
		result = UCThreadMutex_Unlock(hTaskGraphLock);
		ERRIFGOTO(result, _EXIT);

		result = UKCPUGeneralTaskManager_GetTaskState(hManager, pstTask, &enState);
		ERRIFGOTO(result, _EXIT);

		result = UCThreadMutex_Lock(hTaskGraphLock);
		ERRIFGOTO(result, _EXIT);

		if(enState == TASK_STATE_SUSPEND)
		{
			pszCurModeName = pstNewModeData->pstParentTaskGraph->pstParentTask->pstMTMInfo->astModeMap[pstNewModeData->nNewModeIndex].pszModeName;

			bCurrentPortAvailable = UKChannel_IsPortRateAvailableTask(pstTask->nTaskId, pszCurModeName);
			//UEM_DEBUG_PRINT("task: %s, available: %d, mode_name: %s\n", pstTask->pszTaskName, bCurrentPortAvailable, pszCurModeName);

			if(pstNewModeData->bModeChanged == TRUE)
			{
				pszOldModeName = pstNewModeData->pstParentTaskGraph->pstParentTask->pstMTMInfo->astModeMap[pstNewModeData->nPrevModeIndex].pszModeName;

				if(UKChannel_IsPortRateAvailableTask(pstTask->nTaskId, pszOldModeName) == FALSE &&
					bCurrentPortAvailable == TRUE)
				{
					//UEM_DEBUG_PRINT("new task: %s, previous_iteration: %d, new_iteration: %d\n", pstTask->pszTaskName, pstTask->nCurIteration, pstNewModeData->nNewStartIteration);

					pstTask->nCurIteration = pstNewModeData->nNewStartIteration;
				}
			}

			if(bCurrentPortAvailable == TRUE)
			{
				result = UCThreadMutex_Unlock(hTaskGraphLock);
				ERRIFGOTO(result, _EXIT);

				result = UKCPUGeneralTaskManager_ChangeState(hManager, pstTask, TASK_STATE_RUNNING);
				ERRIFGOTO(result, _EXIT);

				result = UCThreadMutex_Lock(hTaskGraphLock);
				ERRIFGOTO(result, _EXIT);

				result = UKCPUGeneralTaskManager_ActivateThread(hManager, pstTask);
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
	STask *pstMTMTask = NULL;
	uem_bool bIsSourceTask = FALSE;
	STask *pstCurrentTask = NULL;
	ECPUTaskState enCurrentTaskState = TASK_STATE_STOP;

	pstMTMTask = pstGraph->pstParentTask;

	result = UKCPUGeneralTaskManagerCB_IsSourceTask(pCurrentTaskHandle, &bIsSourceTask);
	ERRIFGOTO(result, _EXIT);

	result = UKCPUGeneralTaskManagerCB_GetCurrentTaskStructure(pCurrentTaskHandle, &pstCurrentTask);
	ERRIFGOTO(result, _EXIT);

	// change task state to suspend when the target task is included in MTM task graph and is not a source task.
	if(enTargetState == TASK_STATE_RUNNING &&
	UKModeTransition_GetModeStateInternal(pstMTMTask->pstMTMInfo) == MODE_STATE_TRANSITING &&
	bIsSourceTask == FALSE)
	{
		enTargetState = TASK_STATE_SUSPEND;
	}

	result = UKCPUGeneralTaskManagerCB_GetCurrentTaskState(pCurrentTaskHandle, &enCurrentTaskState);
	ERRIFGOTO(result, _EXIT);

	if(enCurrentTaskState == TASK_STATE_SUSPEND && enTargetState == TASK_STATE_STOPPING)
	{
		result = UKModeTransition_GetCurrentModeIndexByIteration(pstMTMTask->pstMTMInfo, pstCurrentTask->nCurIteration, &nCurModeIndex);
		if(result == ERR_UEM_NOT_FOUND)
		{
			pszModeName = pstMTMTask->pstMTMInfo->astModeMap[nCurModeIndex].pszModeName;
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



uem_result UKModeTransitionMachineController_HandleModelGeneralDuringStopping(STaskGraph *pstGraph, void *pCurrentTaskHandle, void *pCurrentThreadHandle)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstMTMTask = NULL;
	STask *pstCurrentTask = NULL;

	pstMTMTask = pstGraph->pstParentTask;

	result = UKCPUGeneralTaskManagerCB_GetCurrentTaskStructure(pCurrentTaskHandle, &pstCurrentTask);
	ERRIFGOTO(result, _EXIT);

	result = updateCurrentIteration(pstMTMTask->pstMTMInfo, pstCurrentTask);
	ERRIFGOTO(result, _EXIT);

	if(pstMTMTask->pstMTMInfo->nCurrentIteration <= pstCurrentTask->nCurIteration)
	{
		UEMASSIGNGOTO(result, ERR_UEM_ALREADY_DONE, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result handleModeTransitionInGeneralTasks(STaskGraph *pstGraph, void *pCurrentTaskHandle, void *pCurrentThreadHandle)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstMTMTask = NULL;
	struct _SModeTransitionGeneralSetEventCheck stNewModeData;
	EModeState enModeState;
	uem_bool bIsSourceTask = FALSE;
	STask *pstCurrentTask = NULL;

	pstMTMTask = pstGraph->pstParentTask;

	result = UKCPUGeneralTaskManagerCB_IsSourceTask(pCurrentTaskHandle, &bIsSourceTask);
	ERRIFGOTO(result, _EXIT);

	result = UKCPUGeneralTaskManagerCB_GetCurrentTaskStructure(pCurrentTaskHandle, &pstCurrentTask);
	ERRIFGOTO(result, _EXIT);

	if(bIsSourceTask == TRUE)
	{
		pstMTMTask->pstMTMInfo->fnTransition(pstMTMTask->pstMTMInfo);

		enModeState = UKModeTransition_GetModeStateInternal(pstMTMTask->pstMTMInfo);

		stNewModeData.nPrevModeIndex = pstMTMTask->pstMTMInfo->nCurModeIndex;

		if(enModeState == MODE_STATE_TRANSITING)
		{
			enModeState = UKModeTransition_UpdateModeStateInternal(pstMTMTask->pstMTMInfo, MODE_STATE_NORMAL, pstCurrentTask->nCurIteration-1);
			stNewModeData.bModeChanged = TRUE;
		}
		else
		{
			stNewModeData.bModeChanged = FALSE;
		}

		stNewModeData.nNewModeIndex = pstMTMTask->pstMTMInfo->nCurModeIndex;

		stNewModeData.pCallerTask = pCurrentTaskHandle;
		stNewModeData.nNewStartIteration = pstCurrentTask->nCurIteration-1;
		stNewModeData.pstParentTaskGraph = pstGraph;

		pstMTMTask->pstMTMInfo->nCurrentIteration = pstCurrentTask->nCurIteration;

		result = UKCPUTaskCommon_TraverseSubGraphTasks(pstMTMTask, traverseAndSetEventToTemporarySuspendedMTMTask, &stNewModeData);
		ERRIFGOTO(result, _EXIT);
	}
	else
	{
		if(pstMTMTask->pstMTMInfo->nCurrentIteration <= pstCurrentTask->nCurIteration)
		{
			result = UKCPUGeneralTaskManagerCB_ChangeTaskState(pCurrentTaskHandle, TASK_STATE_SUSPEND);
			ERRIFGOTO(result, _EXIT);
		}
		else
		{
			result = updateCurrentIteration(pstMTMTask->pstMTMInfo, pstCurrentTask);
			ERRIFGOTO(result, _EXIT);

			if(pstMTMTask->pstMTMInfo->nCurrentIteration <= pstCurrentTask->nCurIteration)
			{
				result = UKCPUGeneralTaskManagerCB_ChangeTaskState(pCurrentTaskHandle, TASK_STATE_SUSPEND);
				ERRIFGOTO(result, _EXIT);
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

	result = UKCPUGeneralTaskManagerCB_GetFunctionCalled(pCurrentThreadHandle, &bFunctionCalled);
	ERRIFGOTO(result, _EXIT);

	result = UKCPUGeneralTaskManagerCB_GetCurrentTaskState(pCurrentTaskHandle, &enTaskState);
	ERRIFGOTO(result, _EXIT);

	if(bFunctionCalled == TRUE && enTaskState == TASK_STATE_RUNNING)
	{
		result = handleModeTransitionInGeneralTasks(pstGraph, pCurrentTaskHandle, pCurrentThreadHandle);
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;

_EXIT:
	return result;
}

