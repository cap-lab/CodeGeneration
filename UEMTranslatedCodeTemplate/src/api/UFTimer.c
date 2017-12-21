/*
 * UFTimer.c
 *
 *  Created on: 2017. 8. 12.
 *      Author: jej
 */

#include <uem_common.h>

#include <UFTimer.h>


uem_result UFTimer_GetCurrentTime (OUT long long *pllCurTime)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UFTimer_Set (IN long long llTimeValue, IN char *pszTimeUnit, OUT int *pnTimerId)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UFTimer_GetAlarmed (IN int nTimerId, OUT uem_bool *pbTimerPassed)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UFTimer_Reset (IN int nTimerId)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}



