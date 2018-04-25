/*
 * UFTimer.c
 *
 *  Created on: 2017. 8. 12.
 *      Author: jej
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <UFTimer.h>


uem_result UFTimer_GetCurrentTime (IN int nCallerTaskId, OUT long long *pllCurTime)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UFTimer_Set (IN int nCallerTaskId, IN long long llTimeValue, IN char *pszTimeUnit, OUT int *pnTimerId)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UFTimer_GetAlarmed (IN int nCallerTaskId, IN int nTimerId, OUT uem_bool *pbTimerPassed)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UFTimer_Reset (IN int nCallerTaskId, IN int nTimerId)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}



