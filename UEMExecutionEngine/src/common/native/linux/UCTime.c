/*
 * UCTime.c
 *
 *  Created on: 2017. 8. 10.
 *      Author: jej
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <stdlib.h>
#include <errno.h>
#ifdef HAVE_MINGW_GETTIMEOFDAY
#include <time.h>
#else
#include <sys/time.h>
#endif

#include <UCTime.h>

uem_result UCTime_GetCurTimeInMilliSeconds(long long *pllTime)
{
	uem_result result = ERR_UEM_UNKNOWN;
    long long llTime = 0;
    struct timeval stTime;
    int nRet = 0;

#ifdef ARGUMENT_CHECK
    IFVARERRASSIGNGOTO(pllTime, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif

#ifdef HAVE_MINGW_GETTIMEOFDAY
    nRet = mingw_gettimeofday(&stTime, NULL);
#else
    nRet = gettimeofday(&stTime, NULL);
#endif
    if(nRet != 0)
    {
        ERRASSIGNGOTO(result, ERR_UEM_INTERNAL_FAIL, _EXIT)
    }

    llTime = (((long long) stTime.tv_sec) * 1000) + (stTime.tv_usec / 1000);

    *pllTime = llTime;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UCTime_GetCurTickInMilliSeconds(long long *pllTime)
{
	uem_result result = ERR_UEM_UNKNOWN;
    long long llTime = 0;
    struct timespec stTime;
    int nRet = 0;
#ifdef ARGUMENT_CHECK
    IFVARERRASSIGNGOTO(pllTime, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif

    nRet = clock_gettime(CLOCK_MONOTONIC, &stTime);
    if(nRet != 0)
    {
        ERRASSIGNGOTO(result, ERR_UEM_INTERNAL_FAIL, _EXIT)
    }

    llTime = (((long long) stTime.tv_sec) * 1000) + (stTime.tv_nsec / 1000000);

    *pllTime = llTime;
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

    stTime.tv_sec = nMillisec/1000;
    stTime.tv_nsec = (nMillisec % 1000) * 1000000;
    nError = nanosleep(&stTime, NULL);
    if(nError != 0) {
    	switch(errno) {
    	case EINVAL:
    		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
    		break;
    	case EINTR:
    		ERRASSIGNGOTO(result, ERR_UEM_INTERRUPT, _EXIT);
    		break;
    	default:
    		ERRASSIGNGOTO(result, ERR_UEM_INTERNAL_FAIL, _EXIT);
    		break;
    	}
    }

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


