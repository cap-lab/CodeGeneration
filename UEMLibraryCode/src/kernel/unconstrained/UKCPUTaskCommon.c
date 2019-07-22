/*
 * UKCPUTaskCommon.c
 *
 *  Created on: 2018. 2. 5.
 *      Author: jej
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <UCTime.h>
#include <UCDynamicStack.h>

#include <uem_data.h>

#include <UKTask.h>
#include <UKTime.h>
#include <UKCPUTaskCommon.h>


#define MIN_SLEEP_DURATION (10)
#define MAX_SLEEP_DURATION (100)


static uem_result checkAndPopStack(HStack hStack, IN OUT STaskGraph **ppstTaskGraph, IN OUT int *pnIndex)
{
	uem_result result = ERR_UEM_UNKNOWN;

	STaskGraph *pstTaskGraph = NULL;
	int nIndex = 0;
	int nStackNum = 0;
	void *pIndex = NULL;

	pstTaskGraph = *ppstTaskGraph;
	nIndex = *pnIndex;

	result = UCDynamicStack_Length(hStack, &nStackNum);
	ERRIFGOTO(result, _EXIT);

	if(nIndex >= pstTaskGraph->nNumOfTasks && nStackNum > 0)
	{
		result = UCDynamicStack_Pop(hStack, &pIndex);
		ERRIFGOTO(result, _EXIT);

#if SIZEOF_VOID_P == 8
		nIndex = (int) ((long long) pIndex);
#else
		nIndex = (int) pIndex;
#endif

		result = UCDynamicStack_Pop(hStack, (void **) &pstTaskGraph);
		ERRIFGOTO(result, _EXIT);

		*ppstTaskGraph = pstTaskGraph;
		*pnIndex = nIndex;
	}
	else
	{
		// do nothing
	}
	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result callFunctionsInHierarchicalTaskGraph(STaskGraph *pstTaskGraph, FnTaskTraverse fnCallback, HStack hStack, void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nCurrentIndex = 0;
	int nStackNum = 0;
	STaskGraph *pstCurrentTaskGraph = NULL;
	STaskGraph *pstNextTaskGraph = NULL;
	STask *pstCurTask = NULL;
	int nNumOfTasks = 0;

	nNumOfTasks = pstTaskGraph->nNumOfTasks;
	pstCurrentTaskGraph = pstTaskGraph;

	while(nCurrentIndex < nNumOfTasks || nStackNum > 0)
	{
		if(pstCurrentTaskGraph->astTasks[nCurrentIndex].pstSubGraph != NULL)
		{
			pstNextTaskGraph = pstCurrentTaskGraph->astTasks[nCurrentIndex].pstSubGraph;
			// the current task has subgraph, skip current task index
			nCurrentIndex++;

			if(nCurrentIndex < pstCurrentTaskGraph->nNumOfTasks)
			{
				result = UCDynamicStack_Push(hStack, pstCurrentTaskGraph);
				ERRIFGOTO(result, _EXIT);
#if SIZEOF_VOID_P == 8
				result = UCDynamicStack_Push(hStack, (void *) (long long) nCurrentIndex);
#else
				result = UCDynamicStack_Push(hStack, (void *) nCurrentIndex);
#endif
				ERRIFGOTO(result, _EXIT);
			}

			// reset values
			pstCurrentTaskGraph = pstNextTaskGraph;
			nCurrentIndex = 0;
			nNumOfTasks = pstCurrentTaskGraph->nNumOfTasks;
		}
		else // does not have internal task
		{
			// call current index's proper callback function
			pstCurTask = &(pstCurrentTaskGraph->astTasks[nCurrentIndex]);

			result = fnCallback(pstCurTask, pUserData);
			ERRIFGOTO(result, _EXIT);

			// proceed index, if all index is proceeded, pop the task graph from stack
			nCurrentIndex++;

			result = checkAndPopStack(hStack, &pstCurrentTaskGraph, &nCurrentIndex);
			ERRIFGOTO(result, _EXIT);

			nNumOfTasks = pstCurrentTaskGraph->nNumOfTasks;
		}

		result = UCDynamicStack_Length(hStack, &nStackNum);
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKCPUTaskCommon_TraverseSubGraphTasks(STask *pstParentTask, FnTaskTraverse fnCallback, void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	HStack hStack = NULL;
	STaskGraph* pstTaskGraph;

	result = UCDynamicStack_Create(&hStack);
	ERRIFGOTO(result, _EXIT);

	if(pstParentTask != NULL)
	{
		pstTaskGraph = pstParentTask->pstSubGraph;
	}
	else // when the whole task graph is a composite task.
	{
		pstTaskGraph = &g_stGraph_top;
	}

	result = callFunctionsInHierarchicalTaskGraph(pstTaskGraph, fnCallback, hStack, pUserData);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	if(hStack != NULL)
	{
		UCDynamicStack_Destroy(&hStack, NULL, NULL);
	}
	return result;
}


uem_result UKCPUTaskCommon_CheckTaskState(ECPUTaskState enOldState, ECPUTaskState enNewState)
{
	uem_result result = ERR_UEM_UNKNOWN;

	if(enNewState == enOldState)
	{
		UEMASSIGNGOTO(result, ERR_UEM_ALREADY_DONE, _EXIT);
	}

	switch(enOldState)
	{
	case TASK_STATE_RUNNING:
		// do nothing
		break;
	case TASK_STATE_SUSPEND:
		// do nothing
		break;
	case TASK_STATE_STOP:
		if(enNewState == TASK_STATE_SUSPEND)
		{
			UEMASSIGNGOTO(result, ERR_UEM_SKIP_THIS, _EXIT);
		}
		break;
	case TASK_STATE_STOPPING:
		if(enNewState != TASK_STATE_STOP)
		{
			ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_CONTROL, _EXIT);
		}
		break;
	default:
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_CONTROL, _EXIT);
		break;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKCPUTaskCommon_HandleTimeDrivenTask(STask *pstCurrentTask, FnUemTaskGo fnGo, IN OUT long long *pllNextTime,
										IN OUT int *pnRunCount, IN OUT int *pnNextMaxRunCount, OUT uem_bool *pbFunctionCalled)
{
	uem_result result = ERR_UEM_UNKNOWN;
	long long llCurTime = 0;
	long long llNextTime = 0;
	int nMaxRunCount = 0;
	int nRunCount = 0;
	uem_bool bFunctionCalled = FALSE;

	llNextTime = *pllNextTime;
	nRunCount = *pnRunCount;
	nMaxRunCount = *pnNextMaxRunCount;

	result = UCTime_GetCurTickInMilliSeconds(&llCurTime);
	ERRIFGOTO(result, _EXIT);
	if(llCurTime <= llNextTime) // time is not passed
	{
		//UEM_DEBUG_PRINT("pstCurrentTask (%s) time in: %I64d %I64d %I64d\n", pstCurrentTask->pszTaskName, llCurTime, llNextTime, llNextTime - llCurTime);
		if(nRunCount < nMaxRunCount) // run count is available
		{
			nRunCount++;
			fnGo(pstCurrentTask->nTaskId);
			bFunctionCalled = TRUE;
		}
		else // all run count is used and time is not passed yet
		{
			// if period is long, sleep by time passing
			if(llNextTime - llCurTime > MAX_SLEEP_DURATION)
			{
				UCTime_Sleep(MAX_SLEEP_DURATION);
			}
			else if(llNextTime - llCurTime > MIN_SLEEP_DURATION) // left time is more than SLEEP_DURATION ms
			{
				UCTime_Sleep(MIN_SLEEP_DURATION);
			}
			else
			{
				// otherwise, busy wait
			}
		}
	}
	else // time is passed, reset
	{
		result = UKTime_GetNextTimeByPeriod(llNextTime, pstCurrentTask->nPeriod, pstCurrentTask->enPeriodMetric,
										&llNextTime, &nMaxRunCount);
		ERRIFGOTO(result, _EXIT);
		nRunCount = 0;
		if(nRunCount < nMaxRunCount) // run count is available
		{
			nRunCount++;
			fnGo(pstCurrentTask->nTaskId);
			bFunctionCalled = TRUE;
		}
	}

	*pllNextTime = llNextTime;
	*pnRunCount = nRunCount;
	*pnNextMaxRunCount = nMaxRunCount;
	*pbFunctionCalled = bFunctionCalled;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}



