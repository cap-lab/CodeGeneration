/*
 * UKLoopTaskController.c
 *
 *  Created on: 2019. 7. 15.
 *      Author: jej
 */


#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <uem_data.h>

#include <UKChannel_internal.h>
#include <UKTask_internal.h>
#include <UKCPUGeneralTaskManager.h>


struct _STaskHandleAndTaskGraph {
	STaskGraph *pstTaskGraph;
	void *pTaskHandle;
};

static uem_result updateLoopIterationHistory(STask *pstParentTask, STask *pstCurrentTask)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SLoopInfo *pstLoopInfo = NULL;
	int nHistoryEnd;
	int nLoopCount = 0;
	int nCurIteration = 0;
	int nTargetIteration = 0;

	pstLoopInfo = pstParentTask->pstLoopInfo;
	nCurIteration = pstLoopInfo->nCurrentIteration;
	nTargetIteration = pstCurrentTask->nTargetIteration;
	nLoopCount = pstLoopInfo->nLoopCount;

	if (pstLoopInfo->nCurHistoryLen < LOOP_HISTORY_ARRAY_SIZE) {
		pstLoopInfo->nCurHistoryLen++;
	}
	else {
		pstLoopInfo->nCurHistoryStartIndex++;
		if (pstLoopInfo->nCurHistoryStartIndex >= LOOP_HISTORY_ARRAY_SIZE) {
			pstLoopInfo->nCurHistoryStartIndex = 0;
		}
	}

	nHistoryEnd = pstLoopInfo->nCurHistoryStartIndex + pstLoopInfo->nCurHistoryLen - 1;
	if (nHistoryEnd >= LOOP_HISTORY_ARRAY_SIZE) {
		nHistoryEnd -= LOOP_HISTORY_ARRAY_SIZE;
	}

	pstLoopInfo->astLoopIteration[nHistoryEnd].nPrevIteration = nCurIteration;
	pstLoopInfo->astLoopIteration[nHistoryEnd].nNextIteration = nCurIteration - (nCurIteration % nLoopCount) + nLoopCount;

	if(pstLoopInfo->astLoopIteration[nHistoryEnd].nNextIteration > nTargetIteration)
	{
		pstLoopInfo->astLoopIteration[nHistoryEnd].nNextIteration = nTargetIteration;
	}

	if(pstLoopInfo->astLoopIteration[nHistoryEnd].nPrevIteration > pstLoopInfo->astLoopIteration[nHistoryEnd].nNextIteration)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
	}
	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static uem_result setLoopTaskCurrentIteration(STask *pstTask, void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstParentTask = NULL;
	SLoopInfo *pstLoopInfo = NULL;
	int nSavedIteration = 1;
	int nLoop = 0;
	int nHistoryEnd;
	int nCheckNum = 0;
	STask *pstParentLoopTask = NULL;
	int nNumOfDataToPop = 0;

	pstParentTask = pstTask->pstParentGraph->pstParentTask;
	pstParentLoopTask = (STask *) pUserData;

	while(pstParentTask != NULL )
	{
		if(pstParentTask->pstLoopInfo != NULL)
		{
			if(pstParentTask->pstLoopInfo->enType == LOOP_TYPE_CONVERGENT)
			{
				pstLoopInfo = pstParentTask->pstLoopInfo;

				nHistoryEnd = pstLoopInfo->nCurHistoryStartIndex + pstLoopInfo->nCurHistoryLen - 1;

				if(nHistoryEnd >= LOOP_HISTORY_ARRAY_SIZE)
				{
					nHistoryEnd = nHistoryEnd - LOOP_HISTORY_ARRAY_SIZE;
				}

				result = UCThreadMutex_Lock(pstTask->hMutex);
				ERRIFGOTO(result, _EXIT);

				for(nLoop = nHistoryEnd; nCheckNum < pstLoopInfo->nCurHistoryLen ; nLoop--)
				{
					//UEM_DEBUG_PRINT("pstLoopInfo->astLoopIteration[%d]: prev: %d, next: %d, nCurrentIteration: %d\n", nLoop, pstTask->nCurIteration == pstLoopInfo->astLoopIteration[nLoop].nPrevIteration, pstTask->nCurIteration = pstLoopInfo->astLoopIteration[nLoop].nNextIteration, nCurrentIteration);
					if (pstTask->nCurIteration / nSavedIteration > pstLoopInfo->astLoopIteration[nLoop].nPrevIteration  &&
						pstTask->nCurIteration / nSavedIteration < pstLoopInfo->astLoopIteration[nLoop].nNextIteration) {
						nNumOfDataToPop = pstLoopInfo->astLoopIteration[nLoop].nNextIteration * nSavedIteration - pstTask->nCurIteration;
						pstTask->nCurIteration = pstLoopInfo->astLoopIteration[nLoop].nNextIteration * nSavedIteration;
						break;
					}
					else if(pstTask->nCurIteration / nSavedIteration >= pstLoopInfo->astLoopIteration[nLoop].nNextIteration)
					{
						break;
					}

					if (nLoop <= 0) {
						nLoop = LOOP_HISTORY_ARRAY_SIZE;
					}
					nCheckNum++;
				}

				result = UCThreadMutex_Unlock(pstTask->hMutex);
				ERRIFGOTO(result, _EXIT);

			}
			nSavedIteration = nSavedIteration * pstParentTask->pstLoopInfo->nLoopCount;
		}
		pstParentTask = pstParentTask->pstParentGraph->pstParentTask;
	}

	if(pstParentLoopTask->nTaskId != pstTask->nTaskId && nNumOfDataToPop > 0)
	{
		result = UKChannel_PopLoopTaskBroadcastingDataFromQueueByTaskId(pstParentLoopTask->nTaskId, pstTask->nTaskId, nNumOfDataToPop);
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result setLoopTaskCurrentIterationIfSuspended(STask *pstTask, void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	ECPUTaskState enState;
	HCPUGeneralTaskManager hManager = NULL;
	struct _STaskHandleAndTaskGraph *pstUserData;
	HThreadMutex hTaskGraphLock = NULL;

	pstUserData = (struct _STaskHandleAndTaskGraph *) pUserData;

	result = UKCPUGeneralTaskManagerCB_GetManagerHandle(pstUserData->pTaskHandle, &hManager);
	ERRIFGOTO(result, _EXIT);

	result = UKCPUGeneralTaskManagerCB_GetTaskGraphLock(pstUserData->pTaskHandle, &hTaskGraphLock);
	ERRIFGOTO(result, _EXIT);

	result = UCThreadMutex_Unlock(hTaskGraphLock);
	ERRIFGOTO(result, _EXIT);

	result = UKCPUGeneralTaskManager_GetTaskState(hManager, pstTask, &enState);
	ERRIFGOTO(result, _EXIT);

	result = UCThreadMutex_Lock(hTaskGraphLock);
	ERRIFGOTO(result, _EXIT);

	if(enState == TASK_STATE_SUSPEND)
	{
		result = setLoopTaskCurrentIteration(pstTask, pstUserData->pstTaskGraph->pstParentTask);
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static uem_result changeOwnTaskState(void *pCurrentTaskHandle, ECPUTaskState enTaskStateToChange)
{
	uem_result result = ERR_UEM_UNKNOWN;
	ECPUTaskState enState;

	result = UKCPUGeneralTaskManagerCB_GetCurrentTaskState(pCurrentTaskHandle, &enState);
	ERRIFGOTO(result, _EXIT);

	if(enState == TASK_STATE_SUSPEND)
	{
		result = UKCPUGeneralTaskManagerCB_ChangeTaskState(pCurrentTaskHandle, enTaskStateToChange);
		ERRIFGOTO(result, _EXIT);

		result = UKCPUGeneralTaskManagerCB_ActivateTask(pCurrentTaskHandle);
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result changeOtherTaskState(HCPUGeneralTaskManager hManager, STask *pstTask, HThreadMutex hTaskGraphLock,
										ECPUTaskState enTaskStateToChange)
{
	uem_result result = ERR_UEM_UNKNOWN;
	ECPUTaskState enState;

	result = UCThreadMutex_Unlock(hTaskGraphLock);
	ERRIFGOTO(result, _EXIT);

	result = UKCPUGeneralTaskManager_GetTaskState(hManager, pstTask, &enState);
	ERRIFGOTO(result, _EXIT);

	if(enState == TASK_STATE_SUSPEND)
	{
		result = UKCPUGeneralTaskManager_ChangeState(hManager, pstTask, enTaskStateToChange);
		ERRIFGOTO(result, _EXIT);

		result = UKCPUGeneralTaskManager_ActivateThread(hManager, pstTask);
		ERRIFGOTO(result, _EXIT);
	}

	result = UCThreadMutex_Lock(hTaskGraphLock);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result traverseAndSetEventToStopTask(STask *pstTask, void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	HCPUGeneralTaskManager hManager = NULL;
	struct _STaskHandleAndTaskGraph *pstUserData;
	STask *pstCallerTask = NULL;
	HThreadMutex hTaskGraphLock = NULL;

	pstUserData = (struct _STaskHandleAndTaskGraph *) pUserData;

	result = UKCPUGeneralTaskManagerCB_GetManagerHandle(pstUserData->pTaskHandle, &hManager);
	ERRIFGOTO(result, _EXIT);

	result = UKCPUGeneralTaskManagerCB_GetCurrentTaskStructure(pstUserData->pTaskHandle, &pstCallerTask);
	ERRIFGOTO(result, _EXIT);

	result = UKCPUGeneralTaskManagerCB_GetTaskGraphLock(pstUserData->pTaskHandle, &hTaskGraphLock);
	ERRIFGOTO(result, _EXIT);

	if(pstTask->nCurIteration >= pstTask->nTargetIteration)
	{
		if(pstCallerTask->nTaskId == pstTask->nTaskId)
		{
			result = changeOwnTaskState(pstUserData->pTaskHandle, TASK_STATE_STOP);
			ERRIFGOTO(result, _EXIT);
		}
		else
		{
			result = changeOtherTaskState(hManager, pstTask, hTaskGraphLock, TASK_STATE_STOP);
			ERRIFGOTO(result, _EXIT);
		}
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}



static uem_result traverseAndSetEventToTemporarySuspendedTask(STask *pstTask, void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	HCPUGeneralTaskManager hManager = NULL;
	struct _STaskHandleAndTaskGraph *pstUserData;
	STask *pstCallerTask = NULL;
	HThreadMutex hTaskGraphLock = NULL;

	pstUserData = (struct _STaskHandleAndTaskGraph *) pUserData;

	result = UKCPUGeneralTaskManagerCB_GetCurrentTaskStructure(pstUserData->pTaskHandle, &pstCallerTask);
	ERRIFGOTO(result, _EXIT);

	result = UKCPUGeneralTaskManagerCB_GetManagerHandle(pstUserData->pTaskHandle, &hManager);
	ERRIFGOTO(result, _EXIT);

	result = UKCPUGeneralTaskManagerCB_GetTaskGraphLock(pstUserData->pTaskHandle, &hTaskGraphLock);
	ERRIFGOTO(result, _EXIT);

	if(pstCallerTask->nTaskId == pstTask->nTaskId)
	{
		result = changeOwnTaskState(pstUserData->pTaskHandle, TASK_STATE_RUNNING);
		ERRIFGOTO(result, _EXIT);
	}
	else
	{
		result = changeOtherTaskState(hManager, pstTask, hTaskGraphLock, TASK_STATE_RUNNING);
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result handleLoopTaskIteration(STaskGraph *pstGraph, void *pCurrentTaskHandle, void *pCurrentThreadHandle)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstCurrentTask = NULL;
	STask *pstParentTask = NULL;
	int nLoopCount = 0;
	int nCurIteration = 0;
	struct _STaskHandleAndTaskGraph stUserData;
	ECPUTaskState enState;

	pstParentTask = pstGraph->pstParentTask;

	result = UKCPUGeneralTaskManagerCB_GetCurrentTaskStructure(pCurrentTaskHandle, &pstCurrentTask);
	ERRIFGOTO(result, _EXIT);

	if(pstParentTask->pstLoopInfo->enType == LOOP_TYPE_CONVERGENT && pstParentTask->pstLoopInfo->nDesignatedTaskId == pstCurrentTask->nTaskId)
	{
		if(pstParentTask->pstLoopInfo->bDesignatedTaskState == TRUE)
		{
			//stop loop task iteration
			//set iteration to target iteration
			result = UKCPUGeneralTaskManagerCB_ChangeTaskState(pCurrentTaskHandle, TASK_STATE_SUSPEND);
			ERRIFGOTO(result, _EXIT);

			result = updateLoopIterationHistory(pstParentTask, pstCurrentTask);
			ERRIFGOTO(result, _EXIT);

			nCurIteration = pstParentTask->pstLoopInfo->nCurrentIteration;
			nLoopCount = pstParentTask->pstLoopInfo->nLoopCount;
			pstParentTask->pstLoopInfo->nCurrentIteration = nCurIteration - (nCurIteration % nLoopCount) + nLoopCount;

			stUserData.pTaskHandle = pCurrentTaskHandle;
			stUserData.pstTaskGraph = pstGraph;

//			setLoopTaskCurrentIterationIfSuspended
			result = UKCPUTaskCommon_TraverseSubGraphTasks(pstParentTask, setLoopTaskCurrentIterationIfSuspended, &stUserData);
			ERRIFGOTO(result, _EXIT);

			result = UKCPUTaskCommon_TraverseSubGraphTasks(pstParentTask, traverseAndSetEventToStopTask, &stUserData);
			ERRIFGOTO(result, _EXIT);

			pstParentTask->pstLoopInfo->bDesignatedTaskState = FALSE;

			result = UKCPUGeneralTaskManagerCB_ClearLoopIndex(pCurrentTaskHandle);
			ERRIFGOTO(result, _EXIT);
		}
		else
		{
			//run next iteration
			pstParentTask->pstLoopInfo->nCurrentIteration++;
			//pstParentTask->pstLoopInfo->nCurrentIteration = pstGeneralTask->pstTask->nCurIteration + 1;
		}

		result = UKCPUGeneralTaskManagerCB_GetCurrentTaskState(pCurrentTaskHandle, &enState);
		ERRIFGOTO(result, _EXIT);

		if(enState != TASK_STATE_STOP)
		{
			stUserData.pTaskHandle = pCurrentTaskHandle;
			stUserData.pstTaskGraph = pstGraph;

			result = UKCPUTaskCommon_TraverseSubGraphTasks(pstParentTask, traverseAndSetEventToTemporarySuspendedTask, &stUserData);
			ERRIFGOTO(result, _EXIT);
		}
	}

	result = setLoopTaskCurrentIteration(pstCurrentTask, pstParentTask);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;

_EXIT:
	return result;
}


static uem_bool compareIterationtoAllParentLoopTask(STask *pstTask, int nLoopIndex)
{
	STask *pstCurrentTask = NULL;
	uem_bool bNeedtoSuspend = FALSE;
	int nCurIteration = 0;

	pstCurrentTask = pstTask;
	nCurIteration = pstTask->nCurIteration;

	while(pstCurrentTask != NULL)
	{
		if(pstCurrentTask->pstLoopInfo != NULL)
		{
			if(pstCurrentTask->pstLoopInfo->enType == LOOP_TYPE_CONVERGENT)
			{
				if(pstCurrentTask->pstLoopInfo->nCurrentIteration < nCurIteration)
				{
					bNeedtoSuspend = TRUE;
					break;
				}
				else if(pstCurrentTask->pstLoopInfo->nCurrentIteration < nLoopIndex)
				{
					bNeedtoSuspend = TRUE;
					break;
				}
			}
			else // LOOP_TYPE_DATA
			{

			}

			if(pstTask != pstCurrentTask)
			{
				nCurIteration = nCurIteration / pstCurrentTask->pstLoopInfo->nLoopCount;
			}

			nLoopIndex = nLoopIndex / pstCurrentTask->pstLoopInfo->nLoopCount;
		}

		pstCurrentTask = pstCurrentTask->pstParentGraph->pstParentTask;
	}

	return bNeedtoSuspend;
}


static uem_result updateTaskThreadIterationInConvergentLoop(STask *pstParentTask, void *pTaskHandle, void *pThreadHandle)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstCurrentTask = NULL;
	int nIndex = 0;
	int nCurIteration;
	int nCurLoopIndex = 0;
	STaskGraph *pstIntermediateGraph = NULL;

	result = UKCPUGeneralTaskManagerCB_GetThreadIndex(pThreadHandle, &nIndex);
	ERRIFGOTO(result, _EXIT);

	result = UKCPUGeneralTaskManagerCB_GetCurrentTaskStructure(pTaskHandle, &pstCurrentTask);
	ERRIFGOTO(result, _EXIT);

	result = UCThreadMutex_Lock(pstCurrentTask->hMutex);
	ERRIFGOTO(result, _EXIT);

	result = UKCPUGeneralTaskManagerCB_GetLoopIndex(pTaskHandle, &nCurLoopIndex);
	ERRIFGOTO(result, _EXIT_LOCK);

    if(pstParentTask != NULL && pstParentTask->pstLoopInfo != NULL)
    {
        nCurIteration = pstParentTask->pstLoopInfo->nCurrentIteration;
        pstIntermediateGraph = pstCurrentTask->pstParentGraph;

        while(pstIntermediateGraph->pstParentTask != NULL && pstParentTask != pstIntermediateGraph->pstParentTask)
        {
            if(pstIntermediateGraph->pstParentTask->pstLoopInfo != NULL)
            {
                nCurIteration = nCurIteration * pstIntermediateGraph->pstParentTask->pstLoopInfo->nLoopCount;
            }

            pstIntermediateGraph = pstIntermediateGraph->pstParentTask->pstParentGraph;
        }
    }
	else
	{
		nCurIteration = pstCurrentTask->nCurIteration;
	}

	if(nCurLoopIndex < nCurIteration)
	{
		nCurLoopIndex = nCurIteration;
	}

	result = UKCPUGeneralTaskManagerCB_SetLoopIndex(pTaskHandle, nCurLoopIndex);
	ERRIFGOTO(result, _EXIT_LOCK);

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
	UCThreadMutex_Unlock(pstCurrentTask->hMutex);
_EXIT:
	return result;
}


uem_result UKLoopModelController_HandleConvergentLoop(STaskGraph *pstGraph, void *pCurrentTaskHandle, void *pCurrentThreadHandle)
{
	uem_result result = ERR_UEM_UNKNOWN;
	uem_bool bFunctionCalled = FALSE;
	STask *pstCurrentTask = NULL;
	int nLoopIndex = 0;
	ECPUTaskState enState;
	//uem_bool bNeedSuspended = TRUE;

	result = UKCPUGeneralTaskManagerCB_GetCurrentTaskStructure(pCurrentTaskHandle, &pstCurrentTask);
	ERRIFGOTO(result, _EXIT);

	result = UKCPUGeneralTaskManagerCB_GetFunctionCalled(pCurrentThreadHandle, &bFunctionCalled);
	ERRIFGOTO(result, _EXIT);

	if(bFunctionCalled == TRUE)
	{
		//handle Loop Task and change TaskState if necessary.
		result = handleLoopTaskIteration(pstGraph, pCurrentTaskHandle, pCurrentThreadHandle);
		ERRIFGOTO(result, _EXIT);
	}

	result = UKCPUGeneralTaskManagerCB_GetLoopIndex(pCurrentTaskHandle, &nLoopIndex);
	ERRIFGOTO(result, _EXIT);

	if(pstCurrentTask->nTargetIteration > 0 && pstCurrentTask->nCurIteration >= pstCurrentTask->nTargetIteration)
	{
		result = UKCPUGeneralTaskManagerCB_ChangeTaskState(pCurrentTaskHandle, TASK_STATE_STOP);
		ERRIFGOTO(result, _EXIT);
	}
	else if(compareIterationtoAllParentLoopTask(pstCurrentTask, nLoopIndex) == TRUE)
	{
		result = UKCPUGeneralTaskManagerCB_ChangeTaskState(pCurrentTaskHandle, TASK_STATE_SUSPEND);
		ERRIFGOTO(result, _EXIT);
	}

	result = UKCPUGeneralTaskManagerCB_GetCurrentTaskState(pCurrentTaskHandle, &enState);
	ERRIFGOTO(result, _EXIT);

	if(enState == TASK_STATE_RUNNING &&
		(pstCurrentTask->enRunCondition != RUN_CONDITION_TIME_DRIVEN ||
		(pstCurrentTask->enRunCondition == RUN_CONDITION_TIME_DRIVEN && bFunctionCalled == TRUE)))
	{
		result = updateTaskThreadIterationInConvergentLoop(pstGraph->pstParentTask, pCurrentTaskHandle, pCurrentThreadHandle);
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKLoopModelController_HandleDataLoop(STaskGraph *pstGraph, void *pCurrentTaskHandle, void *pCurrentThreadHandle)
{
	uem_result result = ERR_UEM_UNKNOWN;



	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


