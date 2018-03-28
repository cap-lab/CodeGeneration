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

uem_result UFControl_RunTask (IN char *pszTaskName)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = UKTask_RunTask(pszTaskName);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UFControl_StopTask (IN char *pszTaskName, IN uem_bool bDelayedStop)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = UKTask_StopTask(pszTaskName, bDelayedStop);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UFControl_SuspendTask (IN char *pszTaskName)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = UKTask_SuspendTask(pszTaskName);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UFControl_ResumeTask (IN char *pszTaskName)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = UKTask_ResumeTask(pszTaskName);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UFControl_CallTask (IN char *pszTaskName)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = UKTask_CallTask(pszTaskName);
	ERRIFGOTO(result, _EXIT);


	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}



