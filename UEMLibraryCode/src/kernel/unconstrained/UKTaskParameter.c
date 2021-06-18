/*
 * UKTaskParameter.c
 *
 *  Created on: 2018. 2. 12.
 *      Author: jej
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>


#include <UCString.h>

#include <uem_data.h>

#include <UKTask.h>
#include <UKCPUTaskManager.h>

static uem_result getTaskParamElement(EParameterType enType, char *pszParamName, STask *pstTask, OUT STaskParameter **ppstParameter)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;
	uem_string_struct stParamName;
	uem_string_struct stTargetParamName;

	result = UCString_New(&stParamName, pszParamName, UEMSTRING_CONST);
	ERRIFGOTO(result, _EXIT);

	for(nLoop = 0; nLoop < pstTask->nTaskParamNum ; nLoop++)
	{
		if(enType == pstTask->astTaskParam[nLoop].enType)
		{
			result = UCString_New(&stTargetParamName, (char *) pstTask->astTaskParam[nLoop].pszParamName, UEMSTRING_CONST);
			ERRIFGOTO(result, _EXIT);

			if(UCString_IsEqual(&stParamName, &stTargetParamName) == TRUE)
			{
				*ppstParameter = &(pstTask->astTaskParam[nLoop]);
				break;
			}
		}
	}

	if(nLoop == pstTask->nTaskParamNum)
	{
		ERRASSIGNGOTO(result, ERR_UEM_NOT_FOUND, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKTaskParameter_GetInteger (IN int nCallerTaskId, IN char *pszTaskName, IN char *pszParamName, OUT int *pnParamVal)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstTask = NULL;
	STaskParameter *pstParam = NULL;
	STask *pstCallerTask = NULL;

	result = UKTask_GetTaskFromTaskId(nCallerTaskId, &pstCallerTask);
	ERRIFGOTO(result, _EXIT);

	result = UKTask_GetTaskByTaskNameAndCallerTask(pstCallerTask, pszTaskName, &pstTask);
	ERRIFGOTO(result, _EXIT);

	if(pstCallerTask->enType != TASK_TYPE_CONTROL && pstTask->nTaskId != nCallerTaskId)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_CONTROL, _EXIT);
	}

	result = getTaskParamElement(PARAMETER_TYPE_INT, pszParamName, pstTask, &pstParam);
	ERRIFGOTO(result, _EXIT);

	result = UCThreadMutex_Lock(pstTask->hMutex);
	ERRIFGOTO(result, _EXIT);

	*pnParamVal = pstParam->uParamValue.nParam;

	result = UCThreadMutex_Unlock(pstTask->hMutex);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKTaskParameter_SetInteger (IN int nCallerTaskId, IN char *pszTaskName, IN char *pszParamName, IN int nParamVal)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstTask = NULL;
	STaskParameter *pstParam = NULL;
	STask *pstCallerTask = NULL;

	result = UKTask_GetTaskFromTaskId(nCallerTaskId, &pstCallerTask);
	ERRIFGOTO(result, _EXIT);

    result = UKTask_GetTaskByTaskNameAndCallerTask(pstCallerTask, pszTaskName, &pstTask);
	ERRIFGOTO(result, _EXIT);

	if(pstCallerTask->enType != TASK_TYPE_CONTROL && pstTask->nTaskId != nCallerTaskId)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_CONTROL, _EXIT);
	}

	result = UKTask_GetTaskByTaskNameAndCallerTask(pstCallerTask, pszTaskName, &pstTask);
	ERRIFGOTO(result, _EXIT);

	result = getTaskParamElement(PARAMETER_TYPE_INT, pszParamName, pstTask, &pstParam);
	ERRIFGOTO(result, _EXIT);

	result = UCThreadMutex_Lock(pstTask->hMutex);
	ERRIFGOTO(result, _EXIT);

	pstParam->uParamValue.nParam = nParamVal;

	result = UCThreadMutex_Unlock(pstTask->hMutex);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKTaskParameter_GetFloat (IN int nCallerTaskId, IN char *pszTaskName, IN char *pszParamName, OUT double *pdbParamVal)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstTask = NULL;
	STaskParameter *pstParam = NULL;
	STask *pstCallerTask = NULL;

	result = UKTask_GetTaskFromTaskId(nCallerTaskId, &pstCallerTask);
	ERRIFGOTO(result, _EXIT);

	result = UKTask_GetTaskByTaskNameAndCallerTask(pstCallerTask, pszTaskName, &pstTask);
	ERRIFGOTO(result, _EXIT);

	if(pstCallerTask->enType != TASK_TYPE_CONTROL && pstTask->nTaskId != nCallerTaskId)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_CONTROL, _EXIT);
	}

	result = getTaskParamElement(PARAMETER_TYPE_DOUBLE, pszParamName, pstTask, &pstParam);
	ERRIFGOTO(result, _EXIT);

	result = UCThreadMutex_Lock(pstTask->hMutex);
	ERRIFGOTO(result, _EXIT);

	*pdbParamVal = pstParam->uParamValue.dbParam;

	result = UCThreadMutex_Unlock(pstTask->hMutex);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKTaskParameter_SetFloat (IN int nCallerTaskId, IN char *pszTaskName, IN char *pszParamName, IN double dbParamVal)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstTask = NULL;
	STaskParameter *pstParam = NULL;
	STask *pstCallerTask = NULL;

	result = UKTask_GetTaskFromTaskId(nCallerTaskId, &pstCallerTask);
	ERRIFGOTO(result, _EXIT);

	result = UKTask_GetTaskByTaskNameAndCallerTask(pstCallerTask, pszTaskName, &pstTask);
	ERRIFGOTO(result, _EXIT);

	if(pstCallerTask->enType != TASK_TYPE_CONTROL && pstTask->nTaskId != nCallerTaskId)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_CONTROL, _EXIT);
	}

	result = UKTask_GetTaskByTaskNameAndCallerTask(pstCallerTask, pszTaskName, &pstTask);
	ERRIFGOTO(result, _EXIT);

	result = getTaskParamElement(PARAMETER_TYPE_DOUBLE, pszParamName, pstTask, &pstParam);
	ERRIFGOTO(result, _EXIT);

	result = UCThreadMutex_Lock(pstTask->hMutex);
	ERRIFGOTO(result, _EXIT);

	pstParam->uParamValue.dbParam = dbParamVal;

	result = UCThreadMutex_Unlock(pstTask->hMutex);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


