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

#include <uem_data.h>

#include <UFTimer.h>


unsigned int SYS_REQ_GET_CURRENT_TIME_BASE(int nCallerTaskId)
{
	unsigned int unTimeValue;
	long long llCurTime;

	UFTimer_GetCurrentTime(nCallerTaskId, &llCurTime);

	unTimeValue = (long long) llCurTime;

	return unTimeValue;
}


int SYS_REQ_SET_TIMER(int nCallerTaskId, unsigned int nTimeValue, char *pszTimeUnit)
{
	int nTimerSlotId = INVALID_TIMER_SLOT_ID;
	long long llTimeValue;

	llTimeValue = (long long) nTimeValue;

	UFTimer_Set (nCallerTaskId, llTimeValue, pszTimeUnit, &nTimerSlotId);

	return nTimerSlotId;
}


int SYS_REQ_GET_TIMER_ALARMED(int nCallerTaskId, int nTimerId)
{
	int nTimerAlarmed;
	uem_bool bTimerPassed = FALSE;

	UFTimer_GetAlarmed (nCallerTaskId, nTimerId, &bTimerPassed);

	if(bTimerPassed == TRUE)
	{
		nTimerAlarmed = 1;
	}
	else
	{
		nTimerAlarmed = 0;
	}

	return nTimerAlarmed;
}


void SYS_REQ_RESET_TIMER(int nCallerTaskId, int nTimerId)
{
	UFTimer_Reset (nCallerTaskId, nTimerId);
}

