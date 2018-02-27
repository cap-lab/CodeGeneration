/*
 * UFTask_deprecated.c
 *
 *  Created on: 2017. 8. 24.
 *      Author: jej
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <UKCPUTaskManager.h>

#include <UFTask_deprecated.h>
#include <UFTask.h>


void SYS_REQ_SET_THROUGHPUT(char *pszTaskName, char *pszValue, char *pszUnit)
{
	UFTask_SetThroughput (pszTaskName, pszValue, pszUnit);
}


long SYS_REQ_GET_PARAM_INT(char *pszTaskName, char *pszParamName)
{
	uem_result result = ERR_UEM_UNKNOWN;
	long lParamVal = 0;
	int nParamVal = 0;

	result = UFTask_GetIntegerParameter (pszTaskName, pszParamName, &nParamVal);
	ERRIFGOTO(result, _EXIT);

	lParamVal = (long) nParamVal;
_EXIT:
	return lParamVal;
}


void SYS_REQ_SET_PARAM_INT(char *pszTaskName, char *pszParamName, long lParamVal)
{
	int nParamVal = 0;
	uem_result result = ERR_UEM_UNKNOWN;

	nParamVal = (int) lParamVal;

	result = UFTask_SetIntegerParameter (pszTaskName, pszParamName, nParamVal);
	ERRIFGOTO(result, _EXIT);
_EXIT:
	return;
}


double SYS_REQ_GET_PARAM_FLOAT(char *pszTaskName, char *pszParamName)
{
	double dbParamVal = 0;
	uem_result result = ERR_UEM_UNKNOWN;

	result = UFTask_GetFloatParameter(pszTaskName, pszParamName, &dbParamVal);
	ERRIFGOTO(result, _EXIT);

_EXIT:
	return dbParamVal;
}


void SYS_REQ_SET_PARAM_FLOAT(char *pszTaskName, char *pszParamName, double dbParamVal)
{
	uem_result result = ERR_UEM_UNKNOWN;
	result = UFTask_SetFloatParameter(pszTaskName, pszParamName, dbParamVal);
	ERRIFGOTO(result, _EXIT);

_EXIT:
	return;
}

// STATE_RUN (0), STATE_STOP (1) STATE_WAIT (2) STATE_END (3)

int SYS_REQ_CHECK_TASK_STATE(char *pszTaskName)
{
	int nTaskState = 1;
	ETaskState enTaskState = STATE_STOP;
	uem_result result = ERR_UEM_UNKNOWN;

	result = UFTask_GetState(pszTaskName, &enTaskState);
	ERRIFGOTO(result, _EXIT);

	switch(enTaskState)
	{
	case STATE_STOP:
		nTaskState = 1;
		break;
	case STATE_WAIT:
		nTaskState = 2;
		break;
	case STATE_RUN:
		nTaskState = 0;
		break;
	case STATE_END:
		nTaskState = 3;
		break;
	default:
		nTaskState = 1;
		break;
	}

_EXIT:
	return nTaskState;
}


void SYS_REQ_EXECUTE_TRANSITION(char *pszTaskName)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = UFTask_UpdateMode(pszTaskName);
	ERRIFGOTO(result, _EXIT);
_EXIT:
	return;
}


void SYS_REQ_SET_MTM_PARAM_INT(char *pszTaskName, char *pszParamName, long lParamVal)
{
	int nParamVal = 0;
	uem_result result = ERR_UEM_UNKNOWN;

	nParamVal = (int) lParamVal;

	result = UFTask_SetModeIntegerParameter (pszTaskName, pszParamName, nParamVal);
	ERRIFGOTO(result, _EXIT);
_EXIT:
	return;
}


char *SYS_REQ_GET_MODE(char *pszTaskName)
{
	char *pszModeName = NULL;
	uem_result result = ERR_UEM_UNKNOWN;

	result = UFTask_GetCurrentModeName (pszTaskName, &pszModeName);
	ERRIFGOTO(result, _EXIT);
_EXIT:
	return pszModeName;
}

