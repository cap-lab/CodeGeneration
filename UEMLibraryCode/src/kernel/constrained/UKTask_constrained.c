/*
 * UKTask.c
 *
 *  Created on: 2018. 8. 28.
 *      Author: chjej202
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <uem_data.h>

#include <UKTask.h>
#include <UKTaskScheduler.h>

uem_result UKTask_Initialize()
{
	return ERR_UEM_NOERROR;
}


void UKTask_Finalize()
{

}


uem_result UKTask_RunTask (IN int nCallerTaskId, IN char *pszTaskName)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstCallerTask = NULL;
	STask *pstTask = NULL;

	result = UKTask_GetTaskFromTaskId(nCallerTaskId, &pstCallerTask);
	ERRIFGOTO(result, _EXIT);

	if(pstCallerTask->enType != TASK_TYPE_CONTROL)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_CONTROL, _EXIT);
	}

	result = UKTask_GetTaskByTaskNameAndCallerTask(pstCallerTask, pszTaskName, &pstTask);
	ERRIFGOTO(result, _EXIT);

	result = UKTaskScheduler_RunTask(pstTask);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKTask_StopTask (IN int nCallerTaskId, IN char *pszTaskName, IN uem_bool bDelayedStop)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstCallerTask = NULL;
	STask *pstTask = NULL;

	result = UKTask_GetTaskFromTaskId(nCallerTaskId, &pstCallerTask);
	ERRIFGOTO(result, _EXIT);

	if(pstCallerTask->enType != TASK_TYPE_CONTROL)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_CONTROL, _EXIT);
	}

	result = UKTask_GetTaskByTaskNameAndCallerTask(pstCallerTask, pszTaskName, &pstTask);
	ERRIFGOTO(result, _EXIT);

	result = UKTaskScheduler_StopTask(pstTask);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKTask_CallTask (IN int nCallerTaskId, IN char *pszTaskName)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstCallerTask = NULL;
	STask *pstTask = NULL;

	result = UKTask_GetTaskFromTaskId(nCallerTaskId, &pstCallerTask);
	ERRIFGOTO(result, _EXIT);

	if(pstCallerTask->enType != TASK_TYPE_CONTROL)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_CONTROL, _EXIT);
	}

	result = UKTask_GetTaskByTaskNameAndCallerTask(pstCallerTask, pszTaskName, &pstTask);
	ERRIFGOTO(result, _EXIT);

	if(pstTask->stTaskFunctions.fnInit != NULL && pstTask->stTaskFunctions.fnGo != NULL &&
			pstTask->stTaskFunctions.fnWrapup != NULL)
	{
		pstTask->stTaskFunctions.fnInit(pstTask->nTaskId);
		pstTask->stTaskFunctions.fnGo(pstTask->nTaskId);
		pstTask->stTaskFunctions.fnWrapup();
	}
	else
	{
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKTask_GetTaskState(IN int nCallerTaskId, char *pszTaskName, EInternalTaskState *penTaskState)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstCallerTask = NULL;
	STask *pstTask = NULL;

	result = UKTask_GetTaskFromTaskId(nCallerTaskId, &pstCallerTask);
	ERRIFGOTO(result, _EXIT);

	if(pstCallerTask->enType != TASK_TYPE_CONTROL)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_CONTROL, _EXIT);
	}

	result = UKTask_GetTaskByTaskNameAndCallerTask(pstCallerTask, pszTaskName, &pstTask);
	ERRIFGOTO(result, _EXIT);

	result = UKTaskScheduler_GetTaskState(pstTask, penTaskState);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}



