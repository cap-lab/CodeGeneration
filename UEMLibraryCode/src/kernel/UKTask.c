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
#include <UKModeTransition.h>

#include <UKCPUTaskManager.h>

typedef void *HTaskHandle;

typedef uem_result (*FnTaskManagerRunTask)(STask *pstTask);
typedef uem_result (*FnTaskManagerStopTask)(STask *pstTask);
typedef uem_result (*FnTaskManagerSuspendTask)(STask *pstTask);
typedef uem_result (*FnTaskManagerResumeTask)(STask *pstTask);
typedef uem_result (*FnTaskManagerCallTask)(STask *pstTask);


typedef struct _STaskAPI {
	HTaskHandle hTaskManagerHandle;
	FnTaskManagerRunTask fnRunTask;
	FnTaskManagerStopTask fnStopTask;
	FnTaskManagerSuspendTask fnSuspendTask;
	FnTaskManagerResumeTask fnResumeTask;
} STaskAPI;


//STaskAPI g_stCPUTaskAPI = {
//	g_hCPUTaskManager, // hTaskManagerHandle
//	UKCPUTaskManager_RunTask, // fnRunTask
//	UKCPUTaskManager_StopTask, // fnStopTask
//	UKCPUTaskManager_SuspendTask, // fnSuspendTask
//	UKCPUTaskManager_ResumeTask, // fnResumeTask
//};


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



static uem_result UKTask_GetTaskInTaskGraphByTaskName(STaskGraph *pstTaskGraph, uem_string strTaskName, STask **ppstTask)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;
	uem_string_struct stCurrentTaskName;

#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(ppstTask, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	IFVARERRASSIGNGOTO(pstTaskGraph, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	IFVARERRASSIGNGOTO(strTaskName, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif

	for(nLoop = 0 ; nLoop < pstTaskGraph->nNumOfTasks ; nLoop++)
	{
		result = UCString_New(&stCurrentTaskName, (char *) pstTaskGraph->astTasks[nLoop].pszTaskName, UEMSTRING_CONST);
		ERRIFGOTO(result, _EXIT);

		if(UCString_IsEqual(&stCurrentTaskName, strTaskName) == TRUE)
		{
			*ppstTask = &(pstTaskGraph->astTasks[nLoop]);
			break;
		}
	}

	if(nLoop == pstTaskGraph->nNumOfTasks)
	{
		ERRASSIGNGOTO(result, ERR_UEM_NO_DATA, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

#define TASK_NAME_DELIMITER "_"

uem_result UKTask_GetTaskByTaskNameAndCallerTask(STask *pstCallerTask, char *pszTaskName, OUT STask **ppstTask)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstTask = NULL;
	uem_string_struct stTargetTaskName;
	char aszCombinedTaskName[MAX_TASK_NAME_LEN];

	if(pstCallerTask->pstParentGraph->pstParentTask != NULL)
	{
		result = UCString_New(&stTargetTaskName, aszCombinedTaskName, MAX_TASK_NAME_LEN);
		ERRIFGOTO(result, _EXIT);

		result = UCString_SetLow(&stTargetTaskName, pstCallerTask->pstParentGraph->pstParentTask->pszTaskName, UEMSTRING_MAX);
		ERRIFGOTO(result, _EXIT);

		result = UCString_AppendLow(&stTargetTaskName, TASK_NAME_DELIMITER, sizeof(TASK_NAME_DELIMITER)-1);
		ERRIFGOTO(result, _EXIT);

		result = UCString_AppendLow(&stTargetTaskName, pszTaskName, UEMSTRING_MAX);
		ERRIFGOTO(result, _EXIT);
	}
	else // pstCallerTask->pstParentGraph->pstParentTask == NULL -> top-level graph
	{
		result = UCString_New(&stTargetTaskName, pszTaskName, UEMSTRING_CONST);
		ERRIFGOTO(result, _EXIT);
	}

	result = UKTask_GetTaskInTaskGraphByTaskName(pstCallerTask->pstParentGraph, &stTargetTaskName, &pstTask);
	ERRIFGOTO(result, _EXIT);

	*ppstTask = pstTask;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
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

	if(pstTask->nTaskFunctionSetNum == 0)
	{
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}

	// CallTask calls the first init/go/wrapup functions
	pstTask->astTaskFunctions[0].fnInit(pstTask->nTaskId);
	pstTask->astTaskFunctions[0].fnGo(pstTask->nTaskId);
	pstTask->astTaskFunctions[0].fnWrapup();

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKTask_GetTaskFromTaskName(char *pszTaskName, STask **ppstTask)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;
	uem_string_struct stTargetTaskName;
	uem_string_struct stCurrentTaskName;

#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(ppstTask, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	IFVARERRASSIGNGOTO(pszTaskName, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif

	result = UCString_New(&stTargetTaskName, pszTaskName, UEMSTRING_CONST);
	ERRIFGOTO(result, _EXIT);


	for(nLoop = 0 ; nLoop <  g_nTaskIdToTaskNum ; nLoop++)
	{
		result = UCString_New(&stCurrentTaskName, (char *) g_astTaskIdToTask[nLoop].pszTaskName, UEMSTRING_CONST);
		ERRIFGOTO(result, _EXIT);

		if(UCString_IsEqual(&stCurrentTaskName, &stTargetTaskName) == TRUE)
		{
			*ppstTask = g_astTaskIdToTask[nLoop].pstTask;
			break;
		}
	}

	if(nLoop == g_nTaskIdToTaskNum)
	{
		ERRASSIGNGOTO(result, ERR_UEM_NO_DATA, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKTask_GetTaskFromTaskId(int nTaskId, STask **ppstTask)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;

#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(ppstTask, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	if(nTaskId < 0) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
#endif

	for(nLoop = 0 ; nLoop <  g_nTaskIdToTaskNum ; nLoop++)
	{
		if(nTaskId == g_astTaskIdToTask[nLoop].nTaskId)
		{
			*ppstTask = g_astTaskIdToTask[nLoop].pstTask;
			break;
		}
	}

	if(nLoop == g_nTaskIdToTaskNum)
	{
		ERRASSIGNGOTO(result, ERR_UEM_NO_DATA, _EXIT);
	}

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

uem_bool UKTask_isParentTask(int nTaskId, int nParentTaskId)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstTask = NULL;
	STask *pstParentTask = NULL;
	uem_bool bIsParentTask = FALSE;

	result = UKTask_GetTaskFromTaskId(nTaskId, &pstTask);
	ERRIFGOTO(result, _EXIT);

	pstParentTask = pstTask->pstParentGraph->pstParentTask;

	while(pstParentTask != NULL)
	{
		if(pstParentTask->nTaskId == nParentTaskId)
		{
			bIsParentTask = TRUE;
			break;
		}

		pstParentTask = pstParentTask->pstParentGraph->pstParentTask;
	}

_EXIT:
	return bIsParentTask;
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
		pstTask->nTargetIteration = nNewIteration;
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

#ifdef ARGUMENT_CHECK
	if(nTargetIteration < 0) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
#endif
	for(nLoop = 0 ; nLoop < g_nTaskIdToTaskNum ; nLoop++)
	{
		pstTask = g_astTaskIdToTask[nLoop].pstTask;
		pstTopParentTask = pstTask;
		while(pstTopParentTask->pstParentGraph->pstParentTask != NULL)
		{
			pstTopParentTask = pstTopParentTask->pstParentGraph->pstParentTask;
		}
		result = UKTask_SetTargetIteration(pstTask, nTargetIteration, pstTopParentTask->nTaskId);
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKTask_CheckIterationRunCount(STask *pstTask, OUT uem_bool *pbTargetIterationReached)
{
	uem_result result = ERR_UEM_UNKNOWN;
	uem_bool bTargetIterationReached = FALSE;

	result = UCThreadMutex_Lock(pstTask->hMutex);
	ERRIFGOTO(result, _EXIT);

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


uem_result UKTask_IncreaseRunCount(STask *pstTask, OUT uem_bool *pbTargetIterationReached)
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




