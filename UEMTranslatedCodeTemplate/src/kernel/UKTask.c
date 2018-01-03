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


uem_result UKTask_RunTask (IN char *pszTaskName)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstTask = NULL;
//	int nLen = 0;
//	int nLoop = 0;
//	STaskGraph *pstTaskGraph = NULL;

	result = UKTask_GetTaskFromTaskName(pszTaskName, &pstTask);
	ERRIFGOTO(result, _EXIT);

	result = UKCPUTaskManager_RunTask(g_hCPUTaskManager, pstTask->nTaskId);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKTask_StopTask (IN char *pszTaskName, IN uem_bool bDelayedStop)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstTask = NULL;

	result = UKTask_GetTaskFromTaskName(pszTaskName, &pstTask);
	ERRIFGOTO(result, _EXIT);

	if(bDelayedStop == FALSE)
	{
		result = UKCPUTaskManager_StopTask(g_hCPUTaskManager, pstTask->nTaskId);
		ERRIFGOTO(result, _EXIT);
	}
	else // bDelayedStop == TRUE
	{
		// TODO: END_TASK implementation
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKTask_SuspendTask (IN char *pszTaskName)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstTask = NULL;

	result = UKTask_GetTaskFromTaskName(pszTaskName, &pstTask);
	ERRIFGOTO(result, _EXIT);

	result = UKCPUTaskManager_SuspendTask(g_hCPUTaskManager, pstTask->nTaskId);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKTask_ResumeTask (IN char *pszTaskName)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstTask = NULL;

	result = UKTask_GetTaskFromTaskName(pszTaskName, &pstTask);
	ERRIFGOTO(result, _EXIT);

	result = UKCPUTaskManager_ResumeTask(g_hCPUTaskManager, pstTask->nTaskId);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKTask_CallTask (IN char *pszTaskName)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstTask = NULL;

	result = UKTask_GetTaskFromTaskName(pszTaskName, &pstTask);
	ERRIFGOTO(result, _EXIT);

	// CallTask calls the first init/go/wrapup functions
	pstTask->astTaskFunctions[0].fnInit(pstTask->nTaskId);
	pstTask->astTaskFunctions[0].fnGo();
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

	result = UCString_New(&stTargetTaskName, pszTaskName, UEMSTRING_MAX);
	ERRIFGOTO(result, _EXIT);


	for(nLoop = 0 ; nLoop <  g_nTaskIdToTaskNum ; nLoop++)
	{
		result = UCString_New(&stCurrentTaskName, g_astTaskIdToTask[nLoop].pszTaskName, UEMSTRING_MAX);
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

