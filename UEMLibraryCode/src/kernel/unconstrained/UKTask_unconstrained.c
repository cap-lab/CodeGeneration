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
#include <UKTime.h>
#include <UKModeTransition.h>

#include <UKCPUTaskManager.h>

uem_bool g_bSystemExit = FALSE;

uem_result UKTask_Initialize()
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;
	SModelControllerCommon *pstCommonController = NULL;

	for(nLoop = 0 ; nLoop < g_nTaskIdToTaskNum ; nLoop++)
	{
		result = UCThreadMutex_Create(&(g_astTaskIdToTask[nLoop].pstTask->hMutex));
		ERRIFGOTO(result, _EXIT);

		result = UCThreadEvent_Create(&(g_astTaskIdToTask[nLoop].pstTask->hEvent));
		ERRIFGOTO(result, _EXIT);

		if(g_astTaskIdToTask[nLoop].pstTask->pstParentGraph->pController != NULL)
		{
			pstCommonController = (SModelControllerCommon *) g_astTaskIdToTask[nLoop].pstTask->pstParentGraph->pController;

			if(pstCommonController->hMutex == NULL)
			{
				result = UCThreadMutex_Create(&(pstCommonController->hMutex));
				ERRIFGOTO(result, _EXIT);
			}
		}
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
	SModelControllerCommon *pstCommonController = NULL;

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

		if(g_astTaskIdToTask[nLoop].pstTask->pstParentGraph->pController != NULL)
		{
			pstCommonController = (SModelControllerCommon *) g_astTaskIdToTask[nLoop].pstTask->pstParentGraph->pController;

			if(pstCommonController->hMutex != NULL)
			{
				UCThreadMutex_Destroy(&(pstCommonController->hMutex));
			}
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

	result = UKTask_GetTaskFromTaskId(nCallerTaskId, &pstCallerTask);
	ERRIFGOTO(result, _EXIT);

	if(bDelayedStop == FALSE)
	{
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
		if (pszTaskName == NULL)
		{
			pstTask = pstCallerTask;
		}
		else // pstCallerTask != NULL
		{
			result = UKTask_GetTaskFromTaskName(pszTaskName, &pstTask);
			ERRIFGOTO(result, _EXIT);
		}

		if(pstCallerTask->enType == TASK_TYPE_COMPUTATIONAL &&
						pstCallerTask->pstParentGraph->pstParentTask == NULL && pstCallerTask->nTaskId != pstTask->nTaskId)
		{
			ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_CONTROL, _EXIT);
		}

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
	int nLoop = 0;

	result = UCThreadMutex_Lock(pstTask->hMutex);
	ERRIFGOTO(result, _EXIT);

	pstTask->nCurRunInIteration = 0;
	pstTask->nCurIteration = 0;
	pstTask->nTargetIteration = 0;

	for(nLoop = 0; nLoop < pstTask->nTaskThreadSetNum ; nLoop++)
	{
		pstTask->astThreadContext[nLoop].nCurThreadIteration = 0;
	}

	result = ERR_UEM_NOERROR;

	result = UCThreadMutex_Unlock(pstTask->hMutex);
	ERRIFGOTO(result, _EXIT);
_EXIT:
	return result;
}

static uem_result convertIterationToUpperTaskGraphBase(STask *pstTask, STaskGraph *pstTaskGraph, OUT int *pnConvertedIteration)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nCurIteration = 0;
	STaskGraph *pstParentGraph = NULL;
	SLoopController *pstLoopController = NULL;
	SModeTransitionController *pstMTMController = NULL;
	int nCurModeIndex = 0;
	int nLoop = 0;
	int nModeId = 0;
	int nIndex = 0;
	STask *pstChildTask = NULL;
	uem_bool bStop = FALSE;

	if(pstTaskGraph == pstTask->pstParentGraph)
	{
		*pnConvertedIteration = pstTask->nCurIteration;
	}
	else
	{
		pstParentGraph = pstTask->pstParentGraph;
		nCurIteration = pstTask->nCurIteration;

		while(pstParentGraph != pstTaskGraph && pstParentGraph->pstParentTask != NULL && bStop == FALSE)
		{
			switch(pstParentGraph->enControllerType)
			{
			case CONTROLLER_TYPE_VOID:
				// do nothing
				break;
			case CONTROLLER_TYPE_CONTROL_TASK_INCLUDED:
				bStop = TRUE;
				break;
			case CONTROLLER_TYPE_DYNAMIC_CONVERGENT_LOOP:
			case CONTROLLER_TYPE_DYNAMIC_DATA_LOOP:
			case CONTROLLER_TYPE_STATIC_CONVERGENT_LOOP:
			case CONTROLLER_TYPE_STATIC_DATA_LOOP:
				// divided by loop count
				pstLoopController = (SLoopController *) pstParentGraph->pController;
				nCurIteration = nCurIteration / pstLoopController->pstLoopInfo->nLoopCount;
				break;
			case CONTROLLER_TYPE_DYNAMIC_MODE_TRANSITION:
			case CONTROLLER_TYPE_STATIC_MODE_TRANSITION:
				// divided by this task iteration number
				pstMTMController = (SModeTransitionController *) pstParentGraph->pController;
				if(pstTask != pstChildTask  && pstChildTask != NULL)
				{
					result = UKModeTransition_GetCurrentModeIndexByIteration(pstMTMController->pstMTMInfo, nCurIteration, &nCurModeIndex);
					ERRIFGOTO(result, _EXIT);

					nModeId = pstMTMController->pstMTMInfo->astModeMap[nCurModeIndex].nModeId;

					for(nLoop  = 0 ; nLoop < pstChildTask->nTaskIterationArrayNum ; nLoop++)
					{
						if(pstChildTask->astTaskIteration[nLoop].nModeId == nModeId)
						{
							nIndex = nLoop;
							break;
						}
					}

					if(nLoop == pstChildTask->nTaskIterationArrayNum || pstChildTask->astTaskIteration[nIndex].nRunInIteration == 0)
					{
						//printf("FAIL: task: %s, mode index: %d, mode ID: %d \n", pstChildTask->pszTaskName, nCurModeIndex, nModeId);
						//UEM_DEBUG_PRINT("iteration number is not found, set default to 1\n");
						//ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
					}
					else
					{
						nCurIteration = nCurIteration / pstChildTask->astTaskIteration[nIndex].nRunInIteration;
					}
				}
				break;
			}

			pstChildTask = pstParentGraph->pstParentTask;
			pstParentGraph = pstParentGraph->pstParentTask->pstParentGraph;
		}

		*pnConvertedIteration = nCurIteration;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKTask_ConvertIterationToUpperTaskGraphBase(STask *pstTask, STaskGraph *pstTaskGraph, OUT int *pnConvertedIteration)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = convertIterationToUpperTaskGraphBase(pstTask, pstTaskGraph, pnConvertedIteration);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKTask_GetTaskIteration(STask *pstTask, OUT int *pnTaskIteration)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nModeId;
	int nCurModeIndex = INVALID_ARRAY_INDEX;
	int nLoop = 0;
	int nIndex = 0;
	STaskGraph *pstMTMTaskGraph = NULL;
	STaskGraph *pstTaskGraph = NULL;
	SModeTransitionController *pstController = NULL;
	int nCurrentIteration = 0;

	if(pstTask->pstSubGraph != NULL)
	{
		pstTaskGraph = pstTask->pstSubGraph;
	}
	else
	{
		pstTaskGraph = pstTask->pstParentGraph;
	}

	while(pstTaskGraph->pstParentTask != NULL)
	{
		if(pstTaskGraph->enControllerType == CONTROLLER_TYPE_DYNAMIC_MODE_TRANSITION ||
			pstTaskGraph->enControllerType == CONTROLLER_TYPE_STATIC_MODE_TRANSITION)
		{
			pstMTMTaskGraph = pstTaskGraph;
			break;
		}

		pstTaskGraph = pstTaskGraph->pstParentTask->pstParentGraph;
	}

	if(pstMTMTaskGraph != NULL && pstTask->nTaskIterationArrayNum > 1)
	{
		pstController = (SModeTransitionController *) pstMTMTaskGraph->pController;

		if(pstMTMTaskGraph->enControllerType == CONTROLLER_TYPE_STATIC_MODE_TRANSITION)
		{
			nCurModeIndex = pstController->pstMTMInfo->nCurModeIndex;
		}
		else
		{
			result = convertIterationToUpperTaskGraphBase(pstTask, pstMTMTaskGraph, &nCurrentIteration);
			ERRIFGOTO(result, _EXIT);
			result = UKModeTransition_GetCurrentModeIndexByIteration(pstController->pstMTMInfo, nCurrentIteration, &nCurModeIndex);
			ERRIFGOTO(result, _EXIT);
		}

		//UEM_DEBUG_PRINT("nCurModeIndex: pstTask: %s %d\n", pstTask->pszTaskName, nCurModeIndex);

		nModeId = pstController->pstMTMInfo->astModeMap[nCurModeIndex].nModeId;

		for(nLoop  = 0 ; nLoop < pstTask->nTaskIterationArrayNum ; nLoop++)
		{
			if(pstTask->astTaskIteration[nLoop].nModeId == nModeId)
			{
				nIndex = nLoop;
				break;
			}
		}

		if(nLoop == pstTask->nTaskIterationArrayNum)
		{
			ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
		}
	}
	else
	{
		nIndex = 0;
	}

	*pnTaskIteration = pstTask->astTaskIteration[nIndex].nRunInIteration;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result getIterationNumberBasedOnTargetTaskId(STask *pstTask, int nIterationNumber, int nTargetTaskId, OUT int *pnConvertedIterationNumber)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nIndex = 0;
	int nNewIteration = 1;
	STask *pstCurrentTask = NULL;
	uem_bool bFound = FALSE;

	pstCurrentTask = pstTask;

	while(pstCurrentTask != NULL)
	{

		if(nTargetTaskId == pstCurrentTask->nTaskId)
		{
			nNewIteration = nNewIteration * nIterationNumber;

			bFound = TRUE;
			break;
		}

		if(pstCurrentTask->pstParentGraph->pstParentTask != NULL)
		{
			switch(pstCurrentTask->pstParentGraph->enControllerType)
			{
			case CONTROLLER_TYPE_DYNAMIC_CONVERGENT_LOOP:
			case CONTROLLER_TYPE_DYNAMIC_DATA_LOOP:
				nNewIteration = nNewIteration * pstCurrentTask->pstParentGraph->pstParentTask->astTaskIteration[nIndex].nRunInIteration;
				{
					SLoopController *pstController = NULL;
					pstController = (SLoopController *) pstCurrentTask->pstParentGraph->pController;
					nNewIteration = nNewIteration * pstController->pstLoopInfo->nLoopCount;
				}
				break;
			case CONTROLLER_TYPE_DYNAMIC_MODE_TRANSITION:
				nNewIteration = nNewIteration * pstCurrentTask->pstParentGraph->pstParentTask->astTaskIteration[nIndex].nRunInIteration;
				break;
			default:
				// do nothing
				break;
			}
		}

		pstCurrentTask = pstCurrentTask->pstParentGraph->pstParentTask;
	}

	if(bFound == FALSE)
	{
		ERRASSIGNGOTO(result, ERR_UEM_NOT_FOUND, _EXIT);
	}

	*pnConvertedIterationNumber = nNewIteration;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKTask_GetIterationNumberBasedOnTargetParentTaskId(STask *pstTask, int nIterationNumber, int nTargetTaskId, OUT int *pnConvertedIterationNumber)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = getIterationNumberBasedOnTargetTaskId(pstTask, nIterationNumber, nTargetTaskId, pnConvertedIterationNumber);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKTask_CheckTaskToBeControlled(STaskGraph *pstControlledTaskGraph, STask *pstTask, OUT uem_bool *pbControlled)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STaskGraph *pstParentGraph = NULL;
	uem_bool bControlled = TRUE;

	pstParentGraph = pstTask->pstParentGraph;

	while(pstParentGraph != pstControlledTaskGraph)
	{
		switch(pstParentGraph->enControllerType)
		{
		case CONTROLLER_TYPE_VOID:
			// skip
			break;
		case CONTROLLER_TYPE_CONTROL_TASK_INCLUDED:
			// skip
			break;
		case CONTROLLER_TYPE_DYNAMIC_MODE_TRANSITION:
		case CONTROLLER_TYPE_STATIC_MODE_TRANSITION:
			if(pstParentGraph->pstParentTask->pstParentGraph != pstControlledTaskGraph)
			{
				bControlled = FALSE;
			}
			else
			{
				if(pstParentGraph == pstTask->pstParentGraph)
				{
					bControlled = TRUE;
				}
				else
				{
					bControlled = FALSE;
				}
			}
			break;
		case CONTROLLER_TYPE_DYNAMIC_CONVERGENT_LOOP:
		case CONTROLLER_TYPE_DYNAMIC_DATA_LOOP:
		case CONTROLLER_TYPE_STATIC_CONVERGENT_LOOP:
		case CONTROLLER_TYPE_STATIC_DATA_LOOP:
			break;
		default:
			ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
			break;
		}

		if(bControlled == FALSE)
		{
			break;
		}

		pstParentGraph = pstParentGraph->pstParentTask->pstParentGraph;
	}

	*pbControlled = bControlled;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result updateNewLoopIterationHistory(STaskGraph *pstGraph, STask *pstCurrentTask, int nNewIterationNumber)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SLoopInfo *pstLoopInfo = NULL;
	int nHistoryEnd;
	int nCurIteration = 0;
	int nTargetIteration = 0;

	SLoopController *pstController = NULL;

	pstController = (SLoopController *) pstGraph->pController;

	pstLoopInfo = pstController->pstLoopInfo;
	nCurIteration = pstController->stCommon.nCurrentIteration;
	nTargetIteration = pstCurrentTask->nTargetIteration;

	nHistoryEnd = pstLoopInfo->nCurHistoryStartIndex + pstLoopInfo->nCurHistoryLen - 1;

	if(nHistoryEnd >= LOOP_HISTORY_ARRAY_SIZE)
	{
		nHistoryEnd = nHistoryEnd - LOOP_HISTORY_ARRAY_SIZE;
	}

	if(pstLoopInfo->astLoopIteration[nHistoryEnd].nNextIteration >= nNewIterationNumber)
	{
		UEMASSIGNGOTO(result, ERR_UEM_NOERROR, _EXIT);
	}

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
	pstLoopInfo->astLoopIteration[nHistoryEnd].nNextIteration = nNewIterationNumber;

	if(nTargetIteration > 0 && pstLoopInfo->astLoopIteration[nHistoryEnd].nNextIteration > nTargetIteration)
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


uem_result UKTask_UpdateAllSubGraphCurrentIteration(STaskGraph *pstTaskGraph, STask *pstLeafTask, int nNewIterationNumber)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STaskGraph *pstCurrentGraph = NULL;
	int nConvertedIterationNumber = 0;
	SModelControllerCommon *pstCommon = NULL;
	STask *pstChildTask = NULL;
	SModeTransitionController *pstModeController = NULL;

	pstCurrentGraph = pstLeafTask->pstParentGraph;
	pstChildTask = pstLeafTask;

	while(pstCurrentGraph != pstTaskGraph)
	{
		switch(pstCurrentGraph->enControllerType)
		{
		case CONTROLLER_TYPE_VOID:
			// skip
			break;
		case CONTROLLER_TYPE_CONTROL_TASK_INCLUDED:
			// skip
			break;
		case CONTROLLER_TYPE_DYNAMIC_MODE_TRANSITION:
		case CONTROLLER_TYPE_STATIC_MODE_TRANSITION:
			pstModeController = (SModeTransitionController *) pstCurrentGraph->pController;

			result = getIterationNumberBasedOnTargetTaskId(pstChildTask, nNewIterationNumber, pstTaskGraph->pstParentTask->nTaskId, &nConvertedIterationNumber);
			ERRIFGOTO(result, _EXIT);

			if(pstModeController->stCommon.nCurrentIteration < nConvertedIterationNumber)
			{
				UKModeTransition_UpdateModeStateInternal(pstModeController->pstMTMInfo, MODE_STATE_TRANSITING, nConvertedIterationNumber-1);
			}
			break;
		case CONTROLLER_TYPE_DYNAMIC_CONVERGENT_LOOP:
		case CONTROLLER_TYPE_DYNAMIC_DATA_LOOP:
		case CONTROLLER_TYPE_STATIC_CONVERGENT_LOOP:
		case CONTROLLER_TYPE_STATIC_DATA_LOOP:
			pstCommon = (SModelControllerCommon *) pstCurrentGraph->pController;

			result = getIterationNumberBasedOnTargetTaskId(pstChildTask, nNewIterationNumber, pstTaskGraph->pstParentTask->nTaskId, &nConvertedIterationNumber);
			ERRIFGOTO(result, _EXIT);

			if(pstCommon->nCurrentIteration < nConvertedIterationNumber)
			{
				result = updateNewLoopIterationHistory(pstCurrentGraph, pstChildTask, nConvertedIterationNumber);
				ERRIFGOTO(result, _EXIT);
			}
			break;
		default:
			ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
			break;
		}

		if(pstCurrentGraph->pstParentTask == NULL)
		{
			break;
		}

		pstChildTask = pstCurrentGraph->pstParentTask;
		pstCurrentGraph = pstCurrentGraph->pstParentTask->pstParentGraph;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKTask_SetTargetIteration(STask *pstTask, int nTargetIteration, int nTargetTaskId)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nIndex = 0;
	int nNewIteration = 1;
	int nRemainder = 0;
	int nLoop = 0;

	result = getIterationNumberBasedOnTargetTaskId(pstTask, nTargetIteration, nTargetTaskId, &nNewIteration);
	ERRIFGOTO(result, _EXIT);

	result = UCThreadMutex_Lock(pstTask->hMutex);
	ERRIFGOTO(result, _EXIT);

	pstTask->nTargetIteration = nNewIteration;

	if(pstTask->nTaskThreadSetNum > 1) // This is used only when the task is mapped to multiple processors
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

	result = UCThreadMutex_Unlock(pstTask->hMutex);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
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

	if(pstTask->astThreadContext[nThreadId].nTargetThreadIteration > 0 &&
		pstTask->astThreadContext[nThreadId].nCurThreadIteration >= pstTask->astThreadContext[nThreadId].nTargetThreadIteration)
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
	uem_bool bTargetIterationReached = FALSE;
	int nTaskIteration = 0;

	result = UCThreadMutex_Lock(pstTask->hMutex);
	ERRIFGOTO(result, _EXIT);

	result = UKTask_GetTaskIteration(pstTask, &nTaskIteration);
	ERRIFGOTO(result, _EXIT_LOCK);
	//UEM_DEBUG_PRINT("Task!!: %s => pstTask->astTaskIteration[%d].nRunInIteration: %d, nCurModeIndex: %d\n", pstTask->pszTaskName, nIndex, pstTask->astTaskIteration[nIndex].nRunInIteration, nCurModeIndex);

	pstTask->nCurRunInIteration++;
	pstTask->astThreadContext[nThreadId].nCurThreadIteration++;

	if(pstTask->astThreadContext[nThreadId].nTargetThreadIteration > 0 &&
		pstTask->astThreadContext[nThreadId].nCurThreadIteration >= pstTask->astThreadContext[nThreadId].nTargetThreadIteration)
	{
		bTargetIterationReached = TRUE;
	}

	if(pstTask->nCurRunInIteration >= nTaskIteration)
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

uem_result UKTask_SetPeriod (IN int nCallerTaskId, IN char *pszTaskName, IN int nValue, IN char *pszTimeUnit)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstTask = NULL;
	STask *pstCallerTask = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(nValue, 0, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif

	result = UKTask_GetTaskFromTaskId(nCallerTaskId, &pstCallerTask);
	ERRIFGOTO(result, _EXIT);

	if(pstCallerTask->enType != TASK_TYPE_CONTROL)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_CONTROL, _EXIT);
	}

	result = UKTask_GetTaskByTaskNameAndCallerTask(pstCallerTask, pszTaskName, &pstTask);
	ERRIFGOTO(result, _EXIT);

	result = UCThreadMutex_Lock(pstTask->hMutex);
	ERRIFGOTO(result, _EXIT);

	pstTask->nPeriod = nValue;
	result = UKTime_ConvertTimeUnit(pszTimeUnit, &(pstTask->enPeriodMetric));
	ERRIFGOTO(result, _EXIT);

	result = UCThreadMutex_Unlock(pstTask->hMutex);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKTask_ChangeMappedCore (IN int nCallerTaskId, IN char *pszTaskName, IN int nNewLocalId)
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

	result = UKCPUTaskManager_ChangeMappedCore(g_hCPUTaskManager, pstTask->nTaskId, nNewLocalId);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKTask_ChangeMappingSet(IN int nCallerTaskId, IN char *pszTaskName, IN const char *pszMappingSet)
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

	result = UKCPUTaskManager_ChangeMappingSet(g_hCPUTaskManager, pstTask->nTaskId, pszMappingSet);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKTask_GetCurrentMappingSet(IN int nCallerTaskId, IN char *pszTaskName, IN int nBufferLen, OUT char **ppszMappingSet)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstTask = NULL;
	STask *pstCallerTask = NULL;

	result = UKTask_GetTaskFromTaskId(nCallerTaskId, &pstCallerTask);
	ERRIFGOTO(result, _EXIT);

	result = UKTask_GetTaskByTaskNameAndCallerTask(pstCallerTask, pszTaskName, &pstTask);
	ERRIFGOTO(result, _EXIT);

	result = UKCPUTaskManager_GetCurrentMappingSet(g_hCPUTaskManager, pstTask->nTaskId, nBufferLen, ppszMappingSet);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}
