/*
 * UFTask.c
 *
 *  Created on: 2017. 8. 12.
 *      Author: jej
 */

#include <uem_common.h>

#include <UKModeTransition.h>
#include <UKTask.h>

#include <UFTask.h>



uem_result UFTask_GetIntegerParameter (IN char *pszTaskName, IN char *pszParamName, OUT int *pnParamVal)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UFTask_SetIntegerParameter (IN char *pszTaskName, IN char *pszParamName, IN int nParamVal)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UFTask_GetFloatParameter (IN char *pszTaskName, IN char *pszParamName, OUT double *pdbParamVal)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UFTask_SetFloatParameter (IN char *pszTaskName, IN char *pszParamName, IN double dbParamVal)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UFTask_SetThroughput (IN char *pszTaskName, IN char *pszValue, IN char *pszUnit)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UFTask_GetState (IN char *pszTaskName, OUT ETaskState *penTaskState)
{
	uem_result result = ERR_UEM_UNKNOWN;
	EInternalTaskState enTaskState;

	result = UKTask_GetTaskState(pszTaskName, &enTaskState);
	ERRIFGOTO(result, _EXIT);

	switch(enTaskState)
	{
	case INTERNAL_STATE_RUN:
		*penTaskState = STATE_RUN;
		break;
	case INTERNAL_STATE_WAIT:
		*penTaskState = STATE_WAIT;
		break;
	case INTERNAL_STATE_END:
		*penTaskState = STATE_END;
		break;
	case INTERNAL_STATE_STOP:
		*penTaskState = STATE_STOP;
		break;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UFTask_GetCurrentModeName (IN char *pszTaskName, OUT char **ppszModeName)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = UKModeTransition_GetCurrentModeName (pszTaskName, ppszModeName);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UFTask_SetModeIntegerParameter (IN char *pszTaskName, IN char *pszParamName, IN int nParamVal)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = UKModeTransition_SetModeIntegerParameter (pszTaskName, pszParamName, nParamVal);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UFTask_UpdateMode (IN char *pszTaskName)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = UKModeTransition_UpdateMode (pszTaskName);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}



