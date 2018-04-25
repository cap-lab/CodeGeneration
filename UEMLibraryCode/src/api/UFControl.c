/*
 * UFControl.c
 *
 *  Created on: 2017. 8. 12.
 *      Author: jej
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif


#include <uem_common.h>

#include <UKTask.h>
#include <UFControl.h>

uem_result UFControl_RunTask (IN int nCallerTaskId, IN char *pszTaskName)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = UKTask_RunTask(nCallerTaskId, pszTaskName);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UFControl_StopTask (IN int nCallerTaskId, IN char *pszTaskName, IN uem_bool bDelayedStop)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = UKTask_StopTask(nCallerTaskId, pszTaskName, bDelayedStop);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UFControl_SuspendTask (IN int nCallerTaskId, IN char *pszTaskName)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = UKTask_SuspendTask(nCallerTaskId, pszTaskName);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UFControl_ResumeTask (IN int nCallerTaskId, IN char *pszTaskName)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = UKTask_ResumeTask(nCallerTaskId, pszTaskName);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UFControl_CallTask (IN int nCallerTaskId, IN char *pszTaskName)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = UKTask_CallTask(nCallerTaskId, pszTaskName);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}



