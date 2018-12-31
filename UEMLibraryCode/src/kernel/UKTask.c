/*
 * UKTask.c
 *
 *  Created on: 2018. 9. 7.
 *      Author: chjej202
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <UCString.h>

#include <uem_data.h>

#define TASK_NAME_DELIMITER "_"

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
		result = UCString_New(&stCurrentTaskName, (char *) g_astTaskIdToTask[nLoop].pstTask->pszTaskName, UEMSTRING_CONST);
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

static uem_result getTaskInTaskGraphByTaskName(STaskGraph *pstTaskGraph, uem_string strTaskName, STask **ppstTask)
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

	result = getTaskInTaskGraphByTaskName(pstCallerTask->pstParentGraph, &stTargetTaskName, &pstTask);
	ERRIFGOTO(result, _EXIT);

	*ppstTask = pstTask;

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
		if(nTaskId == g_astTaskIdToTask[nLoop].pstTask->nTaskId)
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





