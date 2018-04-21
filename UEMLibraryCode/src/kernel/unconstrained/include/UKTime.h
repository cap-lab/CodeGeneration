/*
 * UKTime.h
 *
 *  Created on: 2018. 1. 29.
 *      Author: jej
 */

#ifndef SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKTIME_H_
#define SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKTIME_H_

#include <uem_common.h>

#include <uem_data.h>

uem_result UKTime_GetNextTimeByPeriod(long long llPrevTime, int nPeriod, ETimeMetric enPeriodMetric,
											OUT long long *pllNextTime, OUT int *pnNextMaxRunCount);
uem_result UKTime_GetProgramExecutionTime(OUT int *pnValue, OUT ETimeMetric *penMetric);

#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKTIME_H_ */
