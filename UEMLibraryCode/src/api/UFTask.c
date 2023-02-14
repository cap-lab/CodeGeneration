/*
 * UFTask.c
 *
 *  Created on: 2017. 8. 12.
 *      Author: jej
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#ifndef API_LITE
#include <UKModeTransition.h>
#endif

#include <UKTaskParameter.h>
#include <UKTask.h>

#include <UFTask.h>

uem_result UFTask_GetIntegerParameter (IN int nCallerTaskId, IN char *pszTaskName, IN char *pszParamName, OUT int *pnParamVal)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = UKTaskParameter_GetInteger(nCallerTaskId, pszTaskName, pszParamName, pnParamVal);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UFTask_SetIntegerParameter (IN int nCallerTaskId, IN char *pszTaskName, IN char *pszParamName, IN int nParamVal)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = UKTaskParameter_SetInteger(nCallerTaskId, pszTaskName, pszParamName, nParamVal);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UFTask_GetFloatParameter (IN int nCallerTaskId, IN char *pszTaskName, IN char *pszParamName, OUT double *pdbParamVal)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = UKTaskParameter_GetFloat(nCallerTaskId, pszTaskName, pszParamName, pdbParamVal);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UFTask_SetFloatParameter (IN int nCallerTaskId, IN char *pszTaskName, IN char *pszParamName, IN double dbParamVal)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = UKTaskParameter_SetFloat(nCallerTaskId, pszTaskName, pszParamName, dbParamVal);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UFTask_GetState (IN int nCallerTaskId, IN char *pszTaskName, OUT ETaskState *penTaskState)
{
	uem_result result = ERR_UEM_UNKNOWN;
	EInternalTaskState enTaskState;

	result = UKTask_GetTaskState(nCallerTaskId, pszTaskName, &enTaskState);
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

#ifndef API_LITE
uem_result UFTask_SetThroughput (IN int nCallerTaskId, IN char *pszTaskName, IN char *pszValue, IN char *pszUnit)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = UKTask_SetThroughputConstraint (nCallerTaskId, pszTaskName, pszValue, pszUnit);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UFTask_GetCurrentModeName (IN int nCallerTaskId, IN char *pszTaskName, OUT char **ppszModeName)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = UKModeTransition_GetCurrentModeName (nCallerTaskId, pszTaskName, ppszModeName);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UFTask_SetModeIntegerParameter (IN int nCallerTaskId, IN char *pszTaskName, IN char *pszParamName, IN int nParamVal)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = UKModeTransition_SetModeIntegerParameter (nCallerTaskId, pszTaskName, pszParamName, nParamVal);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UFTask_UpdateMode (IN int nCallerTaskId, IN char *pszTaskName)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = UKModeTransition_UpdateMode (nCallerTaskId, pszTaskName);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UFTask_SetPeriod (IN int nCallerTaskId, IN char *pszTaskName, IN int nValue, IN char *pszTimeUnit)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = UKTask_SetPeriod (nCallerTaskId, pszTaskName, nValue, pszTimeUnit);

	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UFTask_ChangeMappedCore (IN int nCallerTaskId, IN char *pszTaskName, IN int nNewLocalId)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = UKTask_ChangeMappedCore(nCallerTaskId, pszTaskName, nNewLocalId);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UFTask_ChangeMappingSet (IN int nCallerTaskId, IN char *pszTaskName, IN const char *pszMappingSet)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = UKTask_ChangeMappingSet(nCallerTaskId, pszTaskName, pszMappingSet);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UFTask_GetCurrentMappingSet (IN int nCallerTaskId, IN char *pszTaskName, IN int nBufferLen, OUT char **ppszMappingSet)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = UKTask_GetCurrentMappingSet(nCallerTaskId, pszTaskName, nBufferLen, ppszMappingSet);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}
#endif


