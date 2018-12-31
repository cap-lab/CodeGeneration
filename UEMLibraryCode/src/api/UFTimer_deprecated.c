/*
 * UFTimer_deprecated.c
 *
 *  Created on: 2017. 8. 24.
 *      Author: jej
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <UFTimer.h>


unsigned int SYS_REQ_GET_CURRENT_TIME_BASE(int nCallerTaskId)
{
	uem_result result;
	unsigned int unTimeValue = 0;
	long long llCurTime = 0;

	result = UFTimer_GetCurrentTime(nCallerTaskId, &llCurTime);
	ERRIFGOTO(result, _EXIT);

	unTimeValue = (long long) llCurTime;
_EXIT:
	return unTimeValue;
}


int SYS_REQ_SET_TIMER(int nCallerTaskId, unsigned int unTimeValue, char *pszTimeUnit)
{
	uem_result result;
	int nTimerSlotId = INVALID_TIMER_SLOT_ID;
	int nTimeValue;

	nTimeValue = (int) unTimeValue;

	result = UFTimer_SetAlarm (nCallerTaskId, nTimeValue, pszTimeUnit, &nTimerSlotId);
	ERRIFGOTO(result, _EXIT);
_EXIT:
	return nTimerSlotId;
}


int SYS_REQ_GET_TIMER_ALARMED(int nCallerTaskId, int nTimerId)
{
	uem_result result;
	int nTimerAlarmed = -1;
	uem_bool bTimerPassed = FALSE;

	result = UFTimer_GetAlarmed (nCallerTaskId, nTimerId, &bTimerPassed);
	ERRIFGOTO(result, _EXIT);

	if(bTimerPassed == TRUE)
	{
		nTimerAlarmed = 1;
	}
	else
	{
		nTimerAlarmed = 0;
	}
_EXIT:
	return nTimerAlarmed;
}


void SYS_REQ_RESET_TIMER(int nCallerTaskId, int nTimerId)
{
	UFTimer_Reset (nCallerTaskId, nTimerId);
}

