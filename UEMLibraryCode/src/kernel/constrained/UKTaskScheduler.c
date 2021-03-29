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

#include <UKTaskControl.h>
#include <UKTaskScheduler.h>

#define MAX_STACK_INDEX (10)

//extern SGeneralTaskRuntimeInfo g_astGeneralTaskRuntimeInfo[];
//extern SCompositeTaskRuntimeInfo g_astCompositeTaskRuntimeInfo[];

//extern int g_nGeneralTaskNum;
//extern int g_nCompositeTaskNum;



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
		pstParentTask = astRuntimeInfo[nLoop].pstCompositeTaskSchedule->pstParentTaskGraph->pstParentTask;

		if(pstParentTask == NULL)
		{
			callAllInitFunctions();
		}
		else
		{
			if(pstParentTask->enRunCondition != RUN_CONDITION_CONTROL_DRIVEN)
			{
				callHierarchicalInitOrWrapupFunctions(pstParentTask, TRUE);
			}
			else
			{
				//No init for control-driven task.
			}

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
			if(astRuntimeInfo[nLoop].pstCompositeTaskSchedule->pstParentTaskGraph->pstParentTask != NULL)
			{
				pstParentTask = astRuntimeInfo[nLoop].pstCompositeTaskSchedule->pstParentTaskGraph->pstParentTask;
				nTaskId = pstParentTask->nTaskId;

				if(pstParentTask->enRunCondition == RUN_CONDITION_CONTROL_DRIVEN)
				{
					callHierarchicalInitOrWrapupFunctions(pstParentTask, TRUE);
				}
			}
			else
			{
				nTaskId = INVALID_TASK_ID;
			}
			astRuntimeInfo[nLoop].pstCompositeTaskSchedule->fnCompositeGo(nTaskId);

			if(pstParentTask != NULL && pstParentTask->enRunCondition == RUN_CONDITION_CONTROL_DRIVEN)
			{
				//set FALSE to run once
				callHierarchicalInitOrWrapupFunctions(pstParentTask, FALSE);
				astRuntimeInfo[nLoop].bRunning = FALSE;
			}
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





