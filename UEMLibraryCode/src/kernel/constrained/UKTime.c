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

uem_result UKTime_GetNextTimeByPeriod(unsigned long ulPrevTime, int nPeriod, ETimeMetric enPeriodMetric,
											OUT unsigned long *pulNextTime, OUT int *pnNextMaxRunCount)
{
	uem_result result = ERR_UEM_UNKNOWN;
	unsigned long ulPeriod;

	ulPeriod = (unsigned long) nPeriod;

	switch(enPeriodMetric)
	{
	case TIME_METRIC_COUNT: // currently, same to 1 ms
		*pulNextTime = ulPrevTime + 1 * ulPeriod;
		*pnNextMaxRunCount = 1;
		break;
	case TIME_METRIC_CYCLE: // currently, same to 1 ms
		*pulNextTime = ulPrevTime + 1 * ulPeriod;
		*pnNextMaxRunCount = 1;
		break;
	case TIME_METRIC_MICROSEC: // TODO: micro-second time tick is even not correct
		if(ulPeriod > 0 && ulPeriod < MILLISEC_UNIT)
		{
			*pnNextMaxRunCount = MILLISEC_UNIT/ulPeriod;
		}
		else
		{
			*pnNextMaxRunCount = 1;
		}

		if(ulPeriod/MILLISEC_UNIT <= 0)
		{
			*pulNextTime = ulPrevTime + 1;
		}
		else
		{
			*pulNextTime = ulPrevTime + ulPeriod/MILLISEC_UNIT;
		}
		break;
	case TIME_METRIC_MILLISEC:
		*pulNextTime = ulPrevTime + ulPeriod;
		*pnNextMaxRunCount = 1;
		break;
	case TIME_METRIC_SEC:
		*pulNextTime = ulPrevTime + SEC_UNIT * ulPeriod;
		*pnNextMaxRunCount = 1;
		break;
	case TIME_METRIC_MINUTE:
		*pulNextTime = ulPrevTime + SEC_UNIT * MINUTE_UNIT * ulPeriod;
		*pnNextMaxRunCount = 1;
		break;
	case TIME_METRIC_HOUR:
		*pulNextTime = ulPrevTime + SEC_UNIT * MINUTE_UNIT * HOUR_UNIT * ulPeriod;
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

