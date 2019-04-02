/*
 * UKTaskControl.c
 *
 *  Created on: 2019. 4. 2.
 *      Author: chjej202
 */


#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <uem_enum.h>
#include <uem_data.h>

#include <UKTaskControl.h>

typedef uem_result (*FnCompositeTaskFindCallback)(SCompositeTaskRuntimeInfo *pstRuntimeInfo, void *pUserData);
typedef uem_result (*FnGeneralTaskFindCallback)(SGeneralTaskRuntimeInfo *pstRuntimeInfo, void *pUserData);

static uem_result findSingleTask(STask *pstTargetTask, FnGeneralTaskFindCallback fnCallback, void *pUserData, int nTaskNum, SGeneralTaskRuntimeInfo astRuntimeInfo[])
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstTask = NULL;
	int nLoop = 0;
	uem_bool bFound = FALSE;

	for(nLoop = 0 ; nLoop < nTaskNum ; nLoop++)
	{
		pstTask = astRuntimeInfo[nLoop].pstTask;
		if(pstTargetTask->nTaskId == pstTask->nTaskId)
		{
			result = fnCallback(&(astRuntimeInfo[nLoop]), pUserData);
			ERRIFGOTO(result, _EXIT);
			bFound = TRUE;
			break;
		}
	}

	if(bFound == FALSE)
	{
		ERRASSIGNGOTO(result, ERR_UEM_NOT_FOUND, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result setRunningInCompositeTask(SCompositeTaskRuntimeInfo *pstRuntimeInfo, void *pUserData)
{
	uem_time tCurTime;
	uem_result result = ERR_UEM_UNKNOWN;

	result = UCTime_GetCurTickInMilliSeconds(&tCurTime);
	ERRIFGOTO(result, _EXIT);

	pstRuntimeInfo->bRunning = TRUE;
	pstRuntimeInfo->tNextTime = tCurTime;
	pstRuntimeInfo->nRunCount = 1;

	callHierarchicalInitOrWrapupFunctions(pstRuntimeInfo->pstCompositeTaskSchedule->pstParentTask, TRUE);
_EXIT:
	return result;
}

static uem_result setRunningInGeneralTask(SGeneralTaskRuntimeInfo *pstRuntimeInfo, void *pUserData)
{
	uem_time tCurTime;
	uem_result result = ERR_UEM_UNKNOWN;

	result = UCTime_GetCurTickInMilliSeconds(&tCurTime);
	ERRIFGOTO(result, _EXIT);

	pstRuntimeInfo->bRunning = TRUE;
	pstRuntimeInfo->tNextTime = tCurTime;
	pstRuntimeInfo->nRunCount = 1;

	pstRuntimeInfo->pstTask->stTaskFunctions.fnInit(pstRuntimeInfo->pstTask->nTaskId);
_EXIT:
	return result;
}

static uem_result setStopInCompositeTask(SCompositeTaskRuntimeInfo *pstRuntimeInfo, void *pUserData)
{
	pstRuntimeInfo->bRunning = FALSE;
	callHierarchicalInitOrWrapupFunctions(pstRuntimeInfo->pstCompositeTaskSchedule->pstParentTask, FALSE);

	return ERR_UEM_NOERROR;
}

static uem_result setStopInGeneralTask(SGeneralTaskRuntimeInfo *pstRuntimeInfo, void *pUserData)
{
	pstRuntimeInfo->bRunning = FALSE;
	pstRuntimeInfo->pstTask->stTaskFunctions.fnWrapup();

	return ERR_UEM_NOERROR;
}


static uem_result getStateInCompositeTask(SCompositeTaskRuntimeInfo *pstRuntimeInfo, void *pUserData)
{
	int *pnRunning;
	pnRunning = (int *) pUserData;

	if(pstRuntimeInfo->bRunning == TRUE)
	{
		*pnRunning = *pnRunning + 1;
	}

	return ERR_UEM_NOERROR;
}

static uem_result getStateInGeneralTask(SGeneralTaskRuntimeInfo *pstRuntimeInfo, void *pUserData)
{
	int *pnRunning;
	pnRunning = (int *) pUserData;

	if(pstRuntimeInfo->bRunning == TRUE)
	{
		*pnRunning = *pnRunning + 1;
	}

	return ERR_UEM_NOERROR;
}


static uem_result findCompositeTask(STask *pstTargetTask, FnCompositeTaskFindCallback fnCallback, void *pUserData, int nTaskNum, SCompositeTaskRuntimeInfo astRuntimeInfo[])
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstTask = NULL;
	int nLoop = 0;
	uem_bool bFound = FALSE;

	for(nLoop = 0 ; nLoop < nTaskNum ; nLoop++)
	{
		pstTask = astRuntimeInfo[nLoop].pstCompositeTaskSchedule->pstParentTask;
		if(pstTargetTask->nTaskId == pstTask->nTaskId)
		{
			result = fnCallback(&(astRuntimeInfo[nLoop]), pUserData);
			ERRIFGOTO(result, _EXIT);
			bFound = TRUE;
			break;
		}
	}

	if(bFound == FALSE)
	{
		ERRASSIGNGOTO(result, ERR_UEM_NOT_FOUND, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static uem_bool containsParentTask(STask *pstTask, int nParentTaskId)
{
	uem_bool bFound = FALSE;
	STask *pstCurrentTask = NULL;

	pstCurrentTask = pstTask;
	while(pstCurrentTask != NULL)
	{
		if(pstCurrentTask->nTaskId == nParentTaskId)
		{
			bFound = TRUE;
			break;
		}
		pstCurrentTask = pstCurrentTask->pstParentGraph->pstParentTask;
	}

	return bFound;
}

static uem_result findHierarchicalTask(STask *pstTargetParentTask, FnGeneralTaskFindCallback fnCallback, void *pUserData, int nTaskNum, SGeneralTaskRuntimeInfo astRuntimeInfo[])
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstTask = NULL;
	int nLoop = 0;
	uem_bool bFound = FALSE;

	for(nLoop = 0 ; nLoop < nTaskNum ; nLoop++)
	{
		pstTask = astRuntimeInfo[nLoop].pstTask;
		if(containsParentTask(pstTask, pstTargetParentTask->nTaskId) == TRUE)
		{
			result = fnCallback(&(astRuntimeInfo[nLoop]), pUserData);
			ERRIFGOTO(result, _EXIT);
			bFound = TRUE;
		}
	}

	if(bFound == FALSE)
	{
		ERRASSIGNGOTO(result, ERR_UEM_NOT_FOUND, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static uem_result applyToGeneralTasks(STask *pstTask, FnGeneralTaskFindCallback fnCallback, void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;

	if(pstTask->pstSubGraph != NULL) // a group of general tasks
	{
		// task with subgraph which is not SDF cannot be controlled
		if(pstTask->pstSubGraph->enType == GRAPH_TYPE_PROCESS_NETWORK)
		{
			ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
		}
		else
		{
			result = findHierarchicalTask(pstTask, fnCallback, pUserData, g_nControlTaskNum, g_astControlTaskRuntimeInfo);
			if(result == ERR_UEM_NOT_FOUND)
			{
				// don't care because control task is rarely controlled by control task
				result = ERR_UEM_NOERROR;
			}
			ERRIFGOTO(result, _EXIT);

			// find subgraph general task
			result = findHierarchicalTask(pstTask, fnCallback, pUserData, g_nGeneralTaskNum, g_astGeneralTaskRuntimeInfo);
			ERRIFGOTO(result, _EXIT);
		}
	}
	else // general task
	{
		result = findSingleTask(pstTask, fnCallback, pUserData, g_nGeneralTaskNum, g_astGeneralTaskRuntimeInfo);
		if(result == ERR_UEM_NOT_FOUND)
		{
			result = findSingleTask(pstTask, fnCallback, pUserData, g_nControlTaskNum, g_astControlTaskRuntimeInfo);
		}
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKTaskControl_RunTask(STask *pstTask)
{
	uem_result result = ERR_UEM_UNKNOWN;

	if(pstTask->pstSubGraph != NULL)
	{
		result = UKChannel_ClearChannelInSubgraph(pstTask->nTaskId);
		ERRIFGOTO(result, _EXIT);
	}
	// else is single task, so there is no subgraph

	if(pstTask->bStaticScheduled == TRUE && pstTask->pstSubGraph != NULL) // target task is composite task
	{

		result = findCompositeTask(pstTask, setRunningInCompositeTask, NULL, g_nCompositeTaskNum, g_astCompositeTaskRuntimeInfo);
	}
	else // general task
	{
		result = applyToGeneralTasks(pstTask, setRunningInGeneralTask, NULL);
	}
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKTaskControl_StopTask(STask *pstTask)
{
	uem_result result = ERR_UEM_UNKNOWN;

	if(pstTask->bStaticScheduled == TRUE && pstTask->pstSubGraph != NULL)
	{
		// find composite task
		result = findCompositeTask(pstTask, setStopInCompositeTask, NULL, g_nCompositeTaskNum, g_astCompositeTaskRuntimeInfo);
	}
	else // general task
	{
		result = applyToGeneralTasks(pstTask, setStopInGeneralTask, NULL);
	}
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKTaskControl_StoppingTask(STask *pstTask)
{
	uem_result result = ERR_UEM_UNKNOWN;

	// constrained device does not support stopping task (work as same to stop task)
	result = UKTaskControl_StopTask(pstTask);

	return result;
}

// it only retrieves INTERNAL_STATE_RUN/INTERNAL_STATE_STOP for constrained devices
uem_result UKTaskControl_GetTaskState(STask *pstTask, EInternalTaskState *penTaskState)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nRunning = 0;

	if(pstTask->bStaticScheduled == TRUE && pstTask->pstSubGraph != NULL)
	{
		// find composite task
		result = findCompositeTask(pstTask, getStateInCompositeTask, &nRunning, g_nCompositeTaskNum, g_astCompositeTaskRuntimeInfo);
	}
	else // general task
	{
		result = applyToGeneralTasks(pstTask, getStateInGeneralTask, &nRunning);
	}
	ERRIFGOTO(result, _EXIT);

	if(nRunning > 0)
	{
		*penTaskState = INTERNAL_STATE_RUN;
	}
	else
	{
		*penTaskState = INTERNAL_STATE_STOP;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

