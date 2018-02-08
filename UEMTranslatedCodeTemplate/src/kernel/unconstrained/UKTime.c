/*
 * UKTime.c
 *
 *  Created on: 2018. 1. 29.
 *      Author: jej
 */


#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <uem_data.h>

#define SEC_UNIT (1000)
#define MINUTE_UNIT (60)
#define HOUR_UNIT (60)

uem_result UKTime_GetNextTimeByPeriod(long long llPrevTime, int nPeriod, ETimeMetric enPeriodMetric,
											OUT long long *pllNextTime, OUT int *pnNextMaxRunCount)
{
	uem_result result = ERR_UEM_UNKNOWN;

	switch(enPeriodMetric)
	{
	case TIME_METRIC_COUNT: // currently, same to 1 ms
		*pllNextTime = llPrevTime + 1 * nPeriod;
		*pnNextMaxRunCount = 1;
		break;
	case TIME_METRIC_CYCLE: // currently, same to 1 ms
		*pllNextTime = llPrevTime + 1 * nPeriod;
		*pnNextMaxRunCount = 1;
		break;
	case TIME_METRIC_MICROSEC: // TODO: micro-second time tick is even not correct
		if(nPeriod > 0 && nPeriod < SEC_UNIT)
		{
			*pnNextMaxRunCount = SEC_UNIT/nPeriod;
		}
		else
		{
			*pnNextMaxRunCount = 1;
		}

		if(nPeriod/SEC_UNIT <= 0)
		{
			nPeriod = 1;
		}
		*pllNextTime = llPrevTime + 1 * nPeriod/SEC_UNIT;
		break;
	case TIME_METRIC_MILLISEC:
		*pllNextTime = llPrevTime + 1 * nPeriod;
		*pnNextMaxRunCount = 1;
		break;
	case TIME_METRIC_SEC:
		*pllNextTime = llPrevTime + SEC_UNIT * nPeriod;
		*pnNextMaxRunCount = 1;
		break;
	case TIME_METRIC_MINUTE:
		*pllNextTime = llPrevTime + SEC_UNIT * MINUTE_UNIT * nPeriod;
		*pnNextMaxRunCount = 1;
		break;
	case TIME_METRIC_HOUR:
		*pllNextTime = llPrevTime + SEC_UNIT * MINUTE_UNIT * HOUR_UNIT * nPeriod;
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
