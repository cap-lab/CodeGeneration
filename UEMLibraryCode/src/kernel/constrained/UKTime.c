/*
 * UKTime.c
 *
 *  Created on: 2018. 9. 6.
 *      Author: chjej202
 */


#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <uem_enum.h>

#define MILLISEC_UNIT (1000L)
#define SEC_UNIT (1000L)
#define MINUTE_UNIT (60L)
#define HOUR_UNIT (60L)

uem_result UKTime_GetNextTimeByPeriod(uem_time tPrevTime, int nPeriod, ETimeMetric enPeriodMetric,
											OUT uem_time *ptNextTime, OUT int *pnNextMaxRunCount)
{
	uem_result result = ERR_UEM_UNKNOWN;
	uem_time tPeriod;

	tPeriod = (uem_time) nPeriod;

	switch(enPeriodMetric)
	{
	case TIME_METRIC_COUNT: // currently, same to 1 ms
		*ptNextTime = tPrevTime + 1 * tPeriod;
		*pnNextMaxRunCount = 1;
		break;
	case TIME_METRIC_CYCLE: // currently, same to 1 ms
		*ptNextTime = tPrevTime + 1 * tPeriod;
		*pnNextMaxRunCount = 1;
		break;
	case TIME_METRIC_MICROSEC: // TODO: micro-second time tick is even not correct
		if(tPeriod > 0 && tPeriod < MILLISEC_UNIT)
		{
			*pnNextMaxRunCount = MILLISEC_UNIT/tPeriod;
		}
		else
		{
			*pnNextMaxRunCount = 1;
		}

		if(tPeriod/MILLISEC_UNIT <= 0)
		{
			*ptNextTime = tPrevTime + 1;
		}
		else
		{
			*ptNextTime = tPrevTime + tPeriod/MILLISEC_UNIT;
		}
		break;
	case TIME_METRIC_MILLISEC:
		*ptNextTime = tPrevTime + tPeriod;
		*pnNextMaxRunCount = 1;
		break;
	case TIME_METRIC_SEC:
		*ptNextTime = tPrevTime + SEC_UNIT * tPeriod;
		*pnNextMaxRunCount = 1;
		break;
	case TIME_METRIC_MINUTE:
		*ptNextTime = tPrevTime + SEC_UNIT * MINUTE_UNIT * tPeriod;
		*pnNextMaxRunCount = 1;
		break;
	case TIME_METRIC_HOUR:
		*ptNextTime = tPrevTime + SEC_UNIT * MINUTE_UNIT * HOUR_UNIT * tPeriod;
		*pnNextMaxRunCount = 1;
		break;
	default:
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
		break;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

