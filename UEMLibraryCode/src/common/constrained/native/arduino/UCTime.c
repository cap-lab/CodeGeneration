/*
 * UCTime.c
 *
 *  Created on: 2018. 9. 6.
 *      Author: chjej202
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <Arduino.h>

#include <uem_common.h>

// Function name is same but argument type is different compared to unconstrained target UCTime_GetCurTickInMilliSeconds
uem_result UCTime_GetCurTickInMilliSeconds(uem_time *ptTime)
{
	uem_result result = ERR_UEM_UNKNOWN;
	uem_time tMiiliSec = 0;
#ifdef ARGUMENT_CHECK
    IFVARERRASSIGNGOTO(ptTime, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
    tMiiliSec = millis();

	*ptTime = tMiiliSec;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UCTime_Sleep(int nMillisec)
{
	uem_result result = ERR_UEM_UNKNOWN;
#ifdef ARGUMENT_CHECK
    if(nMillisec < 0)
    {
    	ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
    }
#endif
	delay((unsigned long) nMillisec);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

