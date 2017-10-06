/*
 * UFTask_deprecated.c
 *
 *  Created on: 2017. 8. 24.
 *      Author: jej
 */


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
	long lParamVal = 0;
	int nParamVal = 0;

	UFTask_GetIntegerParameter (pszTaskName, pszParamName, &nParamVal);

	lParamVal = (long) nParamVal;

	return lParamVal;
}


void SYS_REQ_SET_PARAM_INT(char *pszTaskName, char *pszParamName, long lParamVal)
{
	int nParamVal = 0;

	nParamVal = (int) lParamVal;

	UFTask_SetIntegerParameter (pszTaskName, pszParamName, nParamVal);
}


double SYS_REQ_GET_PARAM_FLOAT(char *pszTaskName, char *pszParamName)
{
	double dbParamVal = 0;

	UFTask_GetFloatParameter(pszTaskName, pszParamName, &dbParamVal);

	return dbParamVal;
}


void SYS_REQ_SET_PARAM_FLOAT(char *pszTaskName, char *pszParamName, double dbParamVal)
{
	UFTask_SetFloatParameter(pszTaskName, pszParamName, dbParamVal);
}

// STATE_RUN (0), STATE_STOP (1) STATE_WAIT (2) STATE_END (3)

int SYS_REQ_CHECK_TASK_STATE(char *pszTaskName)
{
	int nTaskState;
	ETaskState enTaskState = STATE_STOP;

	UFTask_GetState(pszTaskName, &enTaskState);

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
		break;
	}

	return nTaskState;
}


void SYS_REQ_EXECUTE_TRANSITION(char *pszTaskName)
{
	UFTask_UpdateMode(pszTaskName);
}


void SYS_REQ_SET_MTM_PARAM_INT(char *pszTaskName, char *pszParamName, long lParamVal)
{
	int nParamVal = 0;

	nParamVal = (int) lParamVal;

	UFTask_SetModeIntegerParameter (pszTaskName, pszParamName, nParamVal);
}


char *SYS_REQ_GET_MODE(char *pszTaskName)
{
	char *pszModeName = NULL;

	UFTask_GetCurrentModeName (pszTaskName, &pszModeName);

	return pszModeName;
}

