/*
 * UKTaskScheduler.c
 *
 *  Created on: 2018. 9. 5.
 *      Author: chjej202
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <uem_enum.h>
#include <uem_data.h>

#include <UCTime.h>

#include <UKTime.h>
#include <UKTask.h>
#include <UKChannel.h>
#include <UKTaskScheduler.h>


#define MAX_STACK_INDEX (10)

//extern SGeneralTaskRuntimeInfo g_astGeneralTaskRuntimeInfo[];
//extern SCompositeTaskRuntimeInfo g_astCompositeTaskRuntimeInfo[];

//extern int g_nGeneralTaskNum;
//extern int g_nCompositeTaskNum;

typedef uem_result (*FnCompositeTaskFindCallback)(SCompositeTaskRuntimeInfo *pstRuntimeInfo, void *pUserData);
typedef uem_result (*FnGeneralTaskFindCallback)(SGeneralTaskRuntimeInfo *pstRuntimeInfo, void *pUserData);

static void initializeGeneralTasks(int nTaskNum, SGeneralTaskRuntimeInfo astRuntimeInfo[])
{
	int nLoop = 0;
	int nTaskId = 0;

	for(nLoop = 0 ; nLoop < nTaskNum ; nLoop++)
	{
		nTaskId = astRuntimeInfo[nLoop].pstTask->nTaskId;
		astRuntimeInfo[nLoop].pstTask->stTaskFunctions.fnInit(nTaskId);
	}
}

static void callAllInitFunctions()
{
	int nLoop = 0;
	STask *pstTask = NULL;
	for(nLoop = 0 ; nLoop <  g_nTaskIdToTaskNum ; nLoop++)
	{
		pstTask = g_astTaskIdToTask[nLoop].pstTask;

		if(pstTask->stTaskFunctions.fnInit != NULL)
		{
			g_astTaskIdToTask[nLoop].pstTask->stTaskFunctions.fnInit(pstTask->nTaskId);
		}
	}
}


static void callHierarchicalInitOrWrapupFunctions(STask *pstParentTask, uem_bool bCallInit)
{
	STask *pstTask;
	STaskGraph *pstTaskGraph;
	int nLoop = 0;
	int nCurStackIndex = 0;
	int anStackIndex[MAX_STACK_INDEX];
	int nNumOfTasks = 0;
	int nStackIndex = 0;

	for(nLoop = 0; nLoop < MAX_STACK_INDEX ; nLoop++)
	{
		anStackIndex[nLoop] = 0;
	}

	pstTaskGraph = pstParentTask->pstSubGraph;
	nNumOfTasks = pstTaskGraph->nNumOfTasks;
	nStackIndex = anStackIndex[nCurStackIndex];

	while(nCurStackIndex > 0 || anStackIndex[0] < nNumOfTasks)
	{
		if(anStackIndex[nCurStackIndex] >= pstTaskGraph->nNumOfTasks)
		{
			pstTaskGraph = pstTaskGraph->pstParentTask->pstParentGraph;
			anStackIndex[nCurStackIndex] = 0;
			nCurStackIndex--;
		}

		if(pstTaskGraph->astTasks[nStackIndex].pstSubGraph != NULL)
		{
			pstTaskGraph = pstTaskGraph->astTasks[nStackIndex].pstSubGraph;
			anStackIndex[nCurStackIndex]++;
			nCurStackIndex++;
		}
		else
		{
			nStackIndex = anStackIndex[nCurStackIndex];
			// call init function
			pstTask = &(pstTaskGraph->astTasks[nStackIndex]);
			if(bCallInit == TRUE)
			{
				pstTask->stTaskFunctions.fnInit(pstTask->nTaskId);
			}
			else
			{
				pstTask->stTaskFunctions.fnWrapup();
			}

			anStackIndex[nCurStackIndex]++;
		}
	}
}



static void initializeCompositeTasks(int nTaskNum, SCompositeTaskRuntimeInfo astRuntimeInfo[])
{
	int nLoop = 0;
	STask *pstParentTask = NULL;

	for(nLoop = 0 ; nLoop < nTaskNum ; nLoop++)
	{
		pstParentTask = astRuntimeInfo[nLoop].pstCompositeTaskSchedule->pstParentTask;

		if(pstParentTask == NULL)
		{
			callAllInitFunctions();
		}
		else
		{
			callHierarchicalInitOrWrapupFunctions(pstParentTask, TRUE);
		}
	}
}

static void setBaseTimeOfGeneralTasks(unsigned long ulBaseTime, int nTaskNum, SGeneralTaskRuntimeInfo astRuntimeInfo[])
{
	int nLoop = 0;
	for(nLoop = 0 ; nLoop < nTaskNum ; nLoop++)
	{
		astRuntimeInfo[nLoop].tNextTime = ulBaseTime;
	}
}

static void setBaseTimeOfCompositeTasks(unsigned long ulBaseTime, int nTaskNum, SCompositeTaskRuntimeInfo astRuntimeInfo[])
{
	int nLoop = 0;
	for(nLoop = 0 ; nLoop < nTaskNum ; nLoop++)
	{
		astRuntimeInfo[nLoop].tNextTime = ulBaseTime;
	}
}

uem_result UKTaskScheduler_Init()
{
	uem_result result = ERR_UEM_UNKNOWN;
	unsigned long ulBaseTime;

	initializeGeneralTasks(g_nControlTaskNum, g_astControlTaskRuntimeInfo);
	initializeGeneralTasks(g_nGeneralTaskNum, g_astGeneralTaskRuntimeInfo);
	initializeCompositeTasks(g_nCompositeTaskNum, g_astCompositeTaskRuntimeInfo);

	//base time setting
	result = UCTime_GetCurTickInMilliSeconds(&ulBaseTime);
	ERRIFGOTO(result, _EXIT);

	setBaseTimeOfGeneralTasks(ulBaseTime, g_nControlTaskNum, g_astControlTaskRuntimeInfo);
	setBaseTimeOfGeneralTasks(ulBaseTime, g_nGeneralTaskNum, g_astGeneralTaskRuntimeInfo);
	setBaseTimeOfCompositeTasks(ulBaseTime, g_nCompositeTaskNum, g_astCompositeTaskRuntimeInfo);
_EXIT:
	return result;
}


static uem_result handleTimeDrivenTask(SGeneralTaskRuntimeInfo *pstRunTimeInfo)
{
	uem_result result = ERR_UEM_UNKNOWN;
	uem_time tPrevTime;
	int nNextRunCount = 0;
	uem_time tCurTime;
	STask *pstTask = NULL;

	result = UCTime_GetCurTickInMilliSeconds(&tCurTime);
	ERRIFGOTO(result, _EXIT);

	pstTask = pstRunTimeInfo->pstTask;
	if(tCurTime <= pstRunTimeInfo->tNextTime)
	{
		if(pstRunTimeInfo->nRunCount > 0)
		{
			pstTask->stTaskFunctions.fnGo(pstTask->nTaskId);
			pstRunTimeInfo->nRunCount--;
		}
		else
		{
			// do nothing
		}
	}
	else // ulCurTime > g_astGeneralTaskRuntimeInfo[nLoop].ulNextTime
	{
		tPrevTime = pstRunTimeInfo->tNextTime;
		result = UKTime_GetNextTimeByPeriod(tPrevTime, pstTask->nPeriod, pstTask->enPeriodMetric,
						&(pstRunTimeInfo->tNextTime), &nNextRunCount);
		ERRIFGOTO(result, _EXIT);

		pstRunTimeInfo->nRunCount = nNextRunCount;
		pstTask->stTaskFunctions.fnGo(pstTask->nTaskId);
		pstRunTimeInfo->nRunCount--;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static uem_result runGeneralTasks(int nTaskNum, SGeneralTaskRuntimeInfo astRuntimeInfo[])
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstTask = NULL;
	int nLoop = 0;

	for(nLoop = 0 ; nLoop < nTaskNum ; nLoop++)
	{
		if(astRuntimeInfo[nLoop].bRunning == TRUE)
		{
			pstTask = astRuntimeInfo[nLoop].pstTask;
			if(pstTask->enRunCondition == RUN_CONDITION_TIME_DRIVEN)
			{
				result = handleTimeDrivenTask(&(astRuntimeInfo[nLoop]));
				ERRIFGOTO(result, _EXIT);
			}
			else
			{
				//UEM_DEBUG_PRINT("general task: running: %s\n", pstTask->pszTaskName);
				pstTask->stTaskFunctions.fnGo(pstTask->nTaskId);
				if(pstTask->enRunCondition == RUN_CONDITION_CONTROL_DRIVEN) // run once on control-driven task
				{
					astRuntimeInfo->pstTask->stTaskFunctions.fnWrapup();
					astRuntimeInfo[nLoop].bRunning = FALSE;
				}
			}
		}
	}
	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result runCompositeTasks(int nTaskNum, SCompositeTaskRuntimeInfo astRuntimeInfo[])
{
	uem_result result = ERR_UEM_NOERROR;
	int nLoop = 0;
	int nTaskId = 0;
	STask *pstParentTask = NULL;

	for(nLoop = 0 ; nLoop < nTaskNum ; nLoop++)
	{
		if(astRuntimeInfo[nLoop].bRunning == TRUE)
		{
			if(astRuntimeInfo[nLoop].pstCompositeTaskSchedule->pstParentTask != NULL)
			{
				pstParentTask = astRuntimeInfo[nLoop].pstCompositeTaskSchedule->pstParentTask;
				nTaskId = pstParentTask->nTaskId;

				if(pstParentTask->enRunCondition == RUN_CONDITION_CONTROL_DRIVEN)
				{
					// set FALSE to run once
					callHierarchicalInitOrWrapupFunctions(pstParentTask, FALSE);
					astRuntimeInfo[nLoop].bRunning = FALSE;
				}
			}
			else
			{
				nTaskId = INVALID_TASK_ID;
			}
			astRuntimeInfo[nLoop].pstCompositeTaskSchedule->fnCompositeGo(nTaskId);
		}
	}

	return result;
}


uem_result UKTaskScheduler_Run()
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = runGeneralTasks(g_nGeneralTaskNum, g_astGeneralTaskRuntimeInfo);
	ERRIFGOTO(result, _EXIT);
	result = runCompositeTasks(g_nCompositeTaskNum, g_astCompositeTaskRuntimeInfo);
	ERRIFGOTO(result, _EXIT);
	result = runGeneralTasks(g_nControlTaskNum, g_astControlTaskRuntimeInfo);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


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


uem_result UKTaskScheduler_RunTask(STask *pstTask)
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

uem_result UKTaskScheduler_StopTask(STask *pstTask)
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


uem_result UKTaskScheduler_StoppingTask(STask *pstTask)
{
	uem_result result = ERR_UEM_UNKNOWN;

	// constrained device does not support stopping task (work as same to stop task)
	result = UKTaskScheduler_StopTask(pstTask);

	return result;
}

// it only retrieves INTERNAL_STATE_RUN/INTERNAL_STATE_STOP for constrained devices
uem_result UKTaskScheduler_GetTaskState(STask *pstTask, EInternalTaskState *penTaskState)
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



