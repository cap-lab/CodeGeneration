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

#include <UKTask_internal.h>
#include <UKModeTransition.h>
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


uem_result UKModeTransitionMachineController_HandleModelGeneral(STaskGraph *pstGraph, void *pCurrentTaskHandle, void *pCurrentThreadHandle)
{
	uem_result result = ERR_UEM_UNKNOWN;



	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

