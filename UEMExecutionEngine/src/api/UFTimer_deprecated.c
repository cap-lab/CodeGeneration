/*
 * UFTimer_deprecated.c
 *
 *  Created on: 2017. 8. 24.
 *      Author: jej
 */


unsigned int SYS_REQ_GET_CURRENT_TIME_BASE()
{
	unsigned int unTimeValue;

	return unTimeValue;
}


int SYS_REQ_SET_TIMER(unsigned int nTimeValue, char *pszTimeUnit)
{
	int nTimerSlotId;

	return nTimerSlotId;
}


int SYS_REQ_GET_TIMER_ALARMED(unsigned int nTimerId)
{
	int nTimerAlarmed;

	return nTimerAlarmed;
}


void SYS_REQ_RESET_TIMER(unsigned int nTimerId)
{

}

