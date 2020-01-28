/*
 * UCTime.c
 *
 *  Created on: 2019. 11. 29.
 *      Author: jej
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <stdlib.h>
#include <windows.h>
#include <time.h>

#include <UCTime.h>

uem_result UCTime_GetCurTimeInMilliSeconds(uem_time *ptTime)
{
	uem_result result = ERR_UEM_UNKNOWN;
    time_t tTimeVal;

#ifdef ARGUMENT_CHECK
    IFVARERRASSIGNGOTO(ptTime, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif

    time(&tTimeVal);

    *ptTime = (uem_time) tTimeVal;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UCTime_GetCurTickInMilliSeconds(uem_time *ptTime)
{
	uem_result result = ERR_UEM_UNKNOWN;
    unsigned long long ullTick = 0;
    int nRet = 0;
#ifdef ARGUMENT_CHECK
    IFVARERRASSIGNGOTO(ptTime, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif

    ullTick = GetTickCount64();

    *ptTime = (uem_time) ullTick;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UCTime_Sleep(int nMillisec)
{
	uem_result result = ERR_UEM_UNKNOWN;
    struct timespec stTime;
    int nError = 0;

#ifdef ARGUMENT_CHECK
    if(nMillisec <= 0)
    {
    	ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
    }
#endif

    Sleep((DWORD) nMillisec);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


