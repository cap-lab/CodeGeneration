/*
 * UKTask.c
 *
 *  Created on: 2017. 9. 2.
 *      Author: jej
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <uem_data.h>

#include <UCString.h>

#include <UKTask.h>
#include <UKTask_internal.h>
#include <UKModeTransition.h>

#include <UKCPUTaskManager.h>

uem_bool g_bSystemExit = FALSE;

uem_result UKTask_Initialize()
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;

	for(nLoop = 0 ; nLoop < g_nTaskIdToTaskNum ; nLoop++)
	{
		result = UCThreadMutex_Create(&(g_astTaskIdToTask[nLoop].pstTask->hMutex));
		ERRIFGOTO(result, _EXIT);

		result = UCThreadEvent_Create(&(g_astTaskIdToTask[nLoop].pstTask->hEvent));
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	if(result != ERR_UEM_NOERROR)
	{
		UKTask_Finalize();
	}
	return result;
}


void UKTask_Finalize()
{
	int nLoop = 0;

	for(nLoop = 0 ; nLoop < g_nTaskIdToTaskNum ; nLoop++)
	{
		if(g_astTaskIdToTask[nLoop].pstTask->hMutex != NULL)
		{
			UCThreadMutex_Destroy(&(g_astTaskIdToTask[nLoop].pstTask->hMutex));
		}

		if(g_astTaskIdToTask[nLoop].pstTask->hEvent != NULL)
		{
			UCThreadEvent_Destroy(&(g_astTaskIdToTask[nLoop].pstTask->hEvent));
		}
	}
}


uem_result UKTask_GetTaskState(IN int nCallerTaskId, IN char *pszTaskName, OUT EInternalTaskState *penTaskState)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstTask = NULL;
	STask *pstCallerTask = NULL;

	result = UKTask_GetTaskFromTaskId(nCallerTaskId, &pstCallerTask);
	ERRIFGOTO(result, _EXIT);

	result = UKTask_GetTaskByTaskNameAndCallerTask(pstCallerTask, pszTaskName, &pstTask);
	ERRIFGOTO(result, _EXIT);

	if(pstCallerTask->enType != TASK_TYPE_CONTROL && pstTask->nTaskId != nCallerTaskId)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_CONTROL, _EXIT);
	}

	result = UKCPUTaskManager_GetTaskState(g_hCPUTaskManager, pstTask->nTaskId, penTaskState);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKTask_RunTask (IN int nCallerTaskId, IN char *pszTaskName)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstTask = NULL;
	STask *pstCallerTask = NULL;

	result = UKTask_GetTaskFromTaskId(nCallerTaskId, &pstCallerTask);
	ERRIFGOTO(result, _EXIT);

	if(pstCallerTask->enType != TASK_TYPE_CONTROL)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_CONTROL, _EXIT);
	}

	result = UKTask_GetTaskByTaskNameAndCallerTask(pstCallerTask, pszTaskName, &pstTask);
	ERRIFGOTO(result, _EXIT);

	result = UKCPUTaskManager_RunTask(g_hCPUTaskManager, pstTask->nTaskId);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKTask_StopTask (IN int nCallerTaskId, IN char *pszTaskName, IN uem_bool bDelayedStop)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstTask = NULL;
	STask *pstCallerTask = NULL;

	if(bDelayedStop == FALSE)
	{
		result = UKTask_GetTaskFromTaskId(nCallerTaskId, &pstCallerTask);
		ERRIFGOTO(result, _EXIT);

		if(pstCallerTask->enType != TASK_TYPE_CONTROL)
		{
			ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_CONTROL, _EXIT);
		}

		result = UKTask_GetTaskByTaskNameAndCallerTask(pstCallerTask, pszTaskName, &pstTask);
		ERRIFGOTO(result, _EXIT);

		result = UKCPUTaskManager_StopTask(g_hCPUTaskManager, pstTask->nTaskId);
		ERRIFGOTO(result, _EXIT);
	}
	else // bDelayedStop == TRUE
	{
		if(pstCallerTask->enType == TASK_TYPE_COMPUTATIONAL &&
						pstCallerTask->pstParentGraph->pstParentTask == NULL && pstCallerTask->nTaskId != pstTask->nTaskId)
		{
			ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_CONTROL, _EXIT);
		}

		result = UKTask_GetTaskFromTaskName(pszTaskName, &pstTask);
		ERRIFGOTO(result, _EXIT);

		result = UKCPUTaskManager_StoppingTask(g_hCPUTaskManager, pstTask->nTaskId);
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKTask_SuspendTask (IN int nCallerTaskId, IN char *pszTaskName)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstTask = NULL;
	STask *pstCallerTask = NULL;

	result = UKTask_GetTaskFromTaskId(nCallerTaskId, &pstCallerTask);
	ERRIFGOTO(result, _EXIT);

	if(pstCallerTask->enType != TASK_TYPE_CONTROL)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_CONTROL, _EXIT);
	}

	result = UKTask_GetTaskByTaskNameAndCallerTask(pstCallerTask, pszTaskName, &pstTask);
	ERRIFGOTO(result, _EXIT);

	result = UKCPUTaskManager_SuspendTask(g_hCPUTaskManager, pstTask->nTaskId);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKTask_ResumeTask (IN int nCallerTaskId, IN char *pszTaskName)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstTask = NULL;
	STask *pstCallerTask = NULL;

	result = UKTask_GetTaskFromTaskId(nCallerTaskId, &pstCallerTask);
	ERRIFGOTO(result, _EXIT);

	if(pstCallerTask->enType != TASK_TYPE_CONTROL)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_CONTROL, _EXIT);
	}

	result = UKTask_GetTaskByTaskNameAndCallerTask(pstCallerTask, pszTaskName, &pstTask);
	ERRIFGOTO(result, _EXIT);

	result = UKCPUTaskManager_ResumeTask(g_hCPUTaskManager, pstTask->nTaskId);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKTask_CallTask (IN int nCallerTaskId, IN char *pszTaskName)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstTask = NULL;
	STask *pstCallerTask = NULL;

	result = UKTask_GetTaskFromTaskId(nCallerTaskId, &pstCallerTask);
	ERRIFGOTO(result, _EXIT);

	result = UKTask_GetTaskByTaskNameAndCallerTask(pstCallerTask, pszTaskName, &pstTask);
	ERRIFGOTO(result, _EXIT);

	if(pstTask->nTaskThreadSetNum == 0)
	{
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}

	// CallTask calls the first init/go/wrapup functions
	pstTask->astTaskThreadFunctions[0].fnInit(pstTask->nTaskId);
	pstTask->astTaskThreadFunctions[0].fnGo(pstTask->nTaskId);
	pstTask->astTaskThreadFunctions[0].fnWrapup();

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKTask_TraverseAllTasks(FnTaskTraverse fnCallback, void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(fnCallback, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	for(nLoop = 0 ; nLoop <  g_nTaskIdToTaskNum ; nLoop++)
	{
		result = fnCallback(g_astTaskIdToTask[nLoop].pstTask, pUserData);
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKTask_SetThroughputConstraint (IN int nCallerTaskId, IN char *pszTaskName, IN char *pszValue, IN char *pszUnit)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstTask = NULL;
	uem_string_struct stValue;
	int nValue = 0;
	STask *pstCallerTask = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(pszValue, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	IFVARERRASSIGNGOTO(pszUnit, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif

	result = UKTask_GetTaskFromTaskId(nCallerTaskId, &pstCallerTask);
	ERRIFGOTO(result, _EXIT);

	if(pstCallerTask->enType != TASK_TYPE_CONTROL)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_CONTROL, _EXIT);
	}

	result = UKTask_GetTaskByTaskNameAndCallerTask(pstCallerTask, pszTaskName, &pstTask);
	ERRIFGOTO(result, _EXIT);

	result = UCString_New(&stValue, pszValue, UEMSTRING_CONST);
	ERRIFGOTO(result, _EXIT);

	nValue = UCString_ToInteger(&stValue, 0, NULL);
	ERRIFGOTO(result, _EXIT);

	result = UCThreadMutex_Lock(pstTask->hMutex);
	ERRIFGOTO(result, _EXIT);

	// TODO: need to adjust pszUnit value
	pstTask->nThroughputConstraint = nValue;

	result = UCThreadMutex_Unlock(pstTask->hMutex);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKTask_ClearRunCount(STask *pstTask)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = UCThreadMutex_Lock(pstTask->hMutex);
	ERRIFGOTO(result, _EXIT);

	pstTask->nCurRunInIteration = 0;
	pstTask->nCurIteration = 0;
	pstTask->nTargetIteration = 0;

	result = ERR_UEM_NOERROR;

	result = UCThreadMutex_Unlock(pstTask->hMutex);
	ERRIFGOTO(result, _EXIT);
_EXIT:
	return result;
}


static uem_result getTaskIterationIndex(STask *pstTask, int nCurrentIteration, int *pbIndex)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nModeNum;
	int nModeId;
	int nCurModeIndex = INVALID_ARRAY_INDEX;
	int nLoop = 0;
	int nIndex = 0;

	if(pstTask->pstMTMInfo != NULL)
	{
		nModeNum = pstTask->pstMTMInfo->nNumOfModes;

		if(pstTask->bStaticScheduled == TRUE)
		{
			nCurModeIndex = pstTask->pstMTMInfo->nCurModeIndex;
		}
		else
		{
			result = UKModeTransition_GetCurrentModeIndexByIteration(pstTask->pstMTMInfo, nCurrentIteration, &nCurModeIndex);
			ERRIFGOTO(result, _EXIT);
		}

		//UEM_DEBUG_PRINT("nCurModeIndex: pstTask: %s %d\n", pstTask->pszTaskName, nCurModeIndex);

		nModeId = pstTask->pstMTMInfo->astModeMap[nCurModeIndex].nModeId;

		for(nLoop  = 0 ; nLoop < nModeNum ; nLoop++)
		{
			if(pstTask->astTaskIteration[nLoop].nModeId == nModeId)
			{
				nIndex = nLoop;
				break;
			}
		}

		if(nLoop == nModeNum)
		{
			ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
		}
	}
	else
	{
		nIndex = 0;
	}

	*pbIndex = nIndex;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKTask_SetTargetIteration(STask *pstTask, int nTargetIteration, int nTargetTaskId)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nIndex = 0;
	int nNewIteration = 1;
	STask *pstCurrentTask = NULL;
	uem_bool bFound = FALSE;
	int nCurrentIteration;

	result = UCThreadMutex_Lock(pstTask->hMutex);
	ERRIFGOTO(result, _EXIT);

	nCurrentIteration = pstTask->nCurIteration;

	pstCurrentTask = pstTask;
	while(pstCurrentTask != NULL)
	{
		if(pstCurrentTask->nTaskId == nTargetTaskId ||
			(pstCurrentTask->pstParentGraph->pstParentTask != NULL &&
			pstCurrentTask->pstParentGraph->pstParentTask->nTaskId == nTargetTaskId))
		{
			if(pstCurrentTask->pstParentGraph->pstParentTask != NULL && pstCurrentTask->pstParentGraph->pstParentTask->pstLoopInfo != NULL)
			{
				result = getTaskIterationIndex(pstCurrentTask->pstParentGraph->pstParentTask, nCurrentIteration, &nIndex);
				ERRIFGOTO(result, _EXIT_LOCK);

				nNewIteration = nNewIteration * pstCurrentTask->pstParentGraph->pstParentTask->astTaskIteration[nIndex].nRunInIteration;
			}
			nNewIteration = nNewIteration * nTargetIteration;
			bFound = TRUE;
			break;
		}
		result = getTaskIterationIndex(pstCurrentTask, nCurrentIteration, &nIndex);
		ERRIFGOTO(result, _EXIT_LOCK);

		nNewIteration = nNewIteration * pstCurrentTask->astTaskIteration[nIndex].nRunInIteration;
		nCurrentIteration = nCurrentIteration / pstCurrentTask->astTaskIteration[nIndex].nRunInIteration;

		pstCurrentTask = pstCurrentTask->pstParentGraph->pstParentTask;
	}

	if(bFound == TRUE)
	{
		int nRemainder = 0;
		int nLoop = 0;
		pstTask->nTargetIteration = nNewIteration;

		if(pstTask->nTaskThreadSetNum > 0)
		{
			nRemainder =  (pstTask->astTaskIteration[nIndex].nRunInIteration * pstTask->nTargetIteration) % pstTask->nTaskThreadSetNum;

			for(nLoop = 0 ; nLoop < pstTask->nTaskThreadSetNum ; nLoop++)
			{
				pstTask->astThreadContext[nLoop].nTargetThreadIteration = (pstTask->astTaskIteration[nIndex].nRunInIteration * pstTask->nTargetIteration) / pstTask->nTaskThreadSetNum;

				if(nLoop < nRemainder)
				{
					pstTask->astThreadContext[nLoop].nTargetThreadIteration++;
				}
			}
		}
	}
	else
	{
		ERRASSIGNGOTO(result, ERR_UEM_NOT_FOUND, _EXIT_LOCK);
	}

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
	UCThreadMutex_Unlock(pstTask->hMutex);
_EXIT:
	return result;
}


uem_result UKTask_SetAllTargetIteration(int nTargetIteration)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;
	STask *pstTask = NULL;
	STask *pstTopParentTask = NULL;
	int nCurTargetIteration = 0;

#ifdef ARGUMENT_CHECK
	if(nTargetIteration < 0) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
#endif
	for(nLoop = 0 ; nLoop < g_nTaskIdToTaskNum ; nLoop++)
	{
		nCurTargetIteration = nTargetIteration;
		pstTask = g_astTaskIdToTask[nLoop].pstTask;
		pstTopParentTask = pstTask;
		while(pstTopParentTask->pstParentGraph->pstParentTask != NULL)
		{
			if(pstTopParentTask->pstParentGraph->pstParentTask->pstLoopInfo != NULL)
			{
				nCurTargetIteration = nCurTargetIteration * pstTopParentTask->pstParentGraph->pstParentTask->pstLoopInfo->nLoopCount;
			}
			pstTopParentTask = pstTopParentTask->pstParentGraph->pstParentTask;
		}
		result = UKTask_SetTargetIteration(pstTask, nCurTargetIteration, pstTopParentTask->nTaskId);
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKTask_CheckIterationRunCount(STask *pstTask, int nThreadId, OUT uem_bool *pbTargetIterationReached)
{
	uem_result result = ERR_UEM_UNKNOWN;
	uem_bool bTargetIterationReached = FALSE;

	result = UCThreadMutex_Lock(pstTask->hMutex);
	ERRIFGOTO(result, _EXIT);

	if(pstTask->astThreadContext[nThreadId].nCurThreadIteration >= pstTask->astThreadContext[nThreadId].nTargetThreadIteration)
	{
		bTargetIterationReached = TRUE;
	}

	if(pstTask->nTargetIteration > 0 && pstTask->nCurIteration >= pstTask->nTargetIteration)
	{
		//UEM_DEBUG_PRINT("Task2: %s => nTargetIteration: %d, current: %d\n", pstTask->pszTaskName, pstTask->nTargetIteration, pstTask->nCurIteration);
		bTargetIterationReached = TRUE;
	}

	result = UCThreadMutex_Unlock(pstTask->hMutex);
	ERRIFGOTO(result, _EXIT);

	if(pbTargetIterationReached != NULL)
	{
		*pbTargetIterationReached = bTargetIterationReached;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKTask_IncreaseRunCount(STask *pstTask, int nThreadId, OUT uem_bool *pbTargetIterationReached)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nIndex = 0;
	uem_bool bTargetIterationReached = FALSE;
	int nModeNum;
	int nModeId;
	int nCurModeIndex = INVALID_ARRAY_INDEX;
	int nLoop = 0;
	STask *pstCurrentTask = NULL;
	STask *pstMTMTask = NULL;

	result = UCThreadMutex_Lock(pstTask->hMutex);
	ERRIFGOTO(result, _EXIT);

	pstCurrentTask = pstTask;

	while(pstCurrentTask != NULL)
	{
		if(pstCurrentTask->pstMTMInfo != NULL)
		{
			pstMTMTask = pstCurrentTask;
			break;
		}

		pstCurrentTask = pstCurrentTask->pstParentGraph->pstParentTask;
	}

	if(pstMTMTask != NULL && pstMTMTask->pstMTMInfo->nNumOfModes > 1)
	{
		nModeNum = pstMTMTask->pstMTMInfo->nNumOfModes;

		if(pstMTMTask->bStaticScheduled == TRUE)
		{
			nCurModeIndex = pstTask->pstMTMInfo->nCurModeIndex;
		}
		else
		{
			result = UKModeTransition_GetCurrentModeIndexByIteration(pstMTMTask->pstMTMInfo, pstTask->nCurIteration, &nCurModeIndex);
			ERRIFGOTO(result, _EXIT_LOCK);
		}

		nModeId = pstMTMTask->pstMTMInfo->astModeMap[nCurModeIndex].nModeId;

		for(nLoop  = 0 ; nLoop < nModeNum ; nLoop++)
		{
			if(pstTask->astTaskIteration[nLoop].nModeId == nModeId)
			{
				nIndex = nLoop;
				break;
			}
		}

		if(nLoop == nModeNum)
		{
			ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT_LOCK);
		}
	}
	else
	{
		nIndex = 0;
	}

	//UEM_DEBUG_PRINT("Task!!: %s => pstTask->astTaskIteration[%d].nRunInIteration: %d, nCurModeIndex: %d\n", pstTask->pszTaskName, nIndex, pstTask->astTaskIteration[nIndex].nRunInIteration, nCurModeIndex);

	pstTask->nCurRunInIteration++;
	pstTask->astThreadContext[nThreadId].nCurThreadIteration++;

	if(pstTask->astThreadContext[nThreadId].nCurThreadIteration >= pstTask->astThreadContext[nThreadId].nTargetThreadIteration)
	{
		bTargetIterationReached = TRUE;
	}

	if(pstTask->nCurRunInIteration >= pstTask->astTaskIteration[nIndex].nRunInIteration)
	{
		pstTask->nCurRunInIteration = 0;
		pstTask->nCurIteration++;
		if(pstTask->nTargetIteration > 0 && pstTask->nCurIteration >= pstTask->nTargetIteration)
		{
			//UEM_DEBUG_PRINT("Task2: %s => nTargetIteration: %d, current: %d\n", pstTask->pszTaskName, pstTask->nTargetIteration, pstTask->nCurIteration);
			bTargetIterationReached = TRUE;
		}
	}

	if(pbTargetIterationReached != NULL)
	{
		*pbTargetIterationReached = bTargetIterationReached;
	}

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
	UCThreadMutex_Unlock(pstTask->hMutex);
_EXIT:
	return result;
}




