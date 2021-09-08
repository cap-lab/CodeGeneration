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

#define MILLISEC_UNIT (1000)
#define SEC_UNIT (1000)
#define MINUTE_UNIT (60)
#define HOUR_UNIT (60)

#define TIMER_UNIT_HOUR "H"
#define TIMER_UNIT_MINUTE "M"
#define TIMER_UNIT_SEC "S"
#define TIMER_UNIT_MILLSEC "MS"
#define TIMER_UNIT_MICROSEC "US"

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

uem_result UKTime_ConvertTimeUnit(char *pszTimeUnit, OUT ETimeMetric *penTimeMetric)
{
	uem_result result = ERR_UEM_UNKNOWN;

	if (UC_memcmp(pszTimeUnit, TIMER_UNIT_SEC, sizeof(TIMER_UNIT_SEC)) == 0)
	{
		*penTimeMetric = TIME_METRIC_SEC;
	}
	else if (UC_memcmp(pszTimeUnit, TIMER_UNIT_MILLSEC, sizeof(TIMER_UNIT_MILLSEC)) == 0)
	{
		*penTimeMetric = TIME_METRIC_MILLISEC;
	}
	else if (UC_memcmp(pszTimeUnit, TIMER_UNIT_MICROSEC, sizeof(TIMER_UNIT_MICROSEC)) == 0)
	{
		*penTimeMetric = TIME_METRIC_MICROSEC;
	}
	else if (UC_memcmp(pszTimeUnit, TIMER_UNIT_HOUR, sizeof(TIMER_UNIT_HOUR)) == 0)
	{
		*penTimeMetric = TIME_METRIC_HOUR;
	}
	else if (UC_memcmp(pszTimeUnit, TIMER_UNIT_MINUTE, sizeof(TIMER_UNIT_MINUTE)) == 0)
	{
		*penTimeMetric = TIME_METRIC_MINUTE;
	}
	else
	{
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKTime_ConvertToMilliSec(long long llTime, ETimeMetric enTimeMetric, OUT long long *pllConvertedtTime)
{
	uem_result result = ERR_UEM_UNKNOWN;

	switch(enTimeMetric)
	{
	case TIME_METRIC_MICROSEC: // TODO: micro-second time tick is even not correct
		*pllConvertedtTime = llTime / 1000 == 0 ? 1 : llTime / 1000;
		break;
	case TIME_METRIC_MILLISEC:
		*pllConvertedtTime = llTime;
		break;
	case TIME_METRIC_SEC:
		*pllConvertedtTime = llTime * SEC_UNIT;
		break;
	case TIME_METRIC_MINUTE:
		*pllConvertedtTime = llTime * SEC_UNIT * MINUTE_UNIT;
		break;
	case TIME_METRIC_HOUR:
		*pllConvertedtTime = llTime * SEC_UNIT * MINUTE_UNIT * HOUR_UNIT;
		break;
	default:
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
		break;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}
