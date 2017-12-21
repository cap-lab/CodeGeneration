/*
 * UFTimer_deprecated.c
 *
 *  Created on: 2017. 8. 24.
 *      Author: jej
 */

#include <uem_common.h>

#include <uem_data.h>

#include <UFTimer.h>


unsigned int SYS_REQ_GET_CURRENT_TIME_BASE()
{
	unsigned int unTimeValue;
	long long llCurTime;

	UFTimer_GetCurrentTime(&llCurTime);

	unTimeValue = (long long) llCurTime;

	return unTimeValue;
}


int SYS_REQ_SET_TIMER(unsigned int nTimeValue, char *pszTimeUnit)
{
	int nTimerSlotId = INVALID_TIMER_SLOT_ID;
	long long llTimeValue;

	llTimeValue = (long long) nTimeValue;

	UFTimer_Set (llTimeValue, pszTimeUnit, &nTimerSlotId);

	return nTimerSlotId;
}


int SYS_REQ_GET_TIMER_ALARMED(int nTimerId)
{
	int nTimerAlarmed;
	uem_bool bTimerPassed = FALSE;

	UFTimer_GetAlarmed (nTimerId, &bTimerPassed);

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


void SYS_REQ_RESET_TIMER(int nTimerId)
{
	UFTimer_Reset (nTimerId);
}

