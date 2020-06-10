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

/**
 * @brief Get next time value by period.
 *
 * This function next time value by period. \n
 * If the period metric is too small, it also provides @a pnNextMaxRunCount to execute multiple times. \n
 * (ex. 1 us period => 1000 run count with 1 ms period)
 *
 * @param llPrevTime previous time value in milliseconds.
 * @param nPeriod period value.
 * @param enPeriodMetric period metric.
 * @param[out] pllNextTime next time value in milliseconds.
 * @param[out] pnNextMaxRunCount run count of current period.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_ILLEGAL_DATA. \n
 *         @ref ERR_UEM_ILLEGAL_DATA is occurred when the period metric is unknown.
 */
uem_result UKTime_GetNextTimeByPeriod(long long llPrevTime, int nPeriod, ETimeMetric enPeriodMetric,
											OUT long long *pllNextTime, OUT int *pnNextMaxRunCount);

/**
 * @brief  Convert the given time as a milli sec unit.
 *
 * This function calculates next runtime with given value and metric. \n
 * The calculation method depends on the input metric unit. \n
 * If task time metric is TIME_METRIC_COUNT or TIME_METRIC_CYCLE, it is calculated as 1ms, \n
 * and the other cases will be calculated according to each unit.
 *
 *
 * @param llTime the given time.
 * @param enTimeMetric the metric for the given time.
 * @param[out] pllConvertedtTime converted time (ms).
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_ILLEGAL_DATA if invalid enPeriodMetric argument comes.
 *
 */
uem_result UKTime_ConvertToMilliSec(long long llTime, ETimeMetric enTimeMetric, OUT long long *pllConvertedtTime);

/**
 * @brief Get program execution time.
 *
 * This function retrieves program execution time.
 *
 * @param[out] pnValue time value.
 * @param[out] penMetric time unit.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKTime_GetProgramExecutionTime(OUT int *pnValue, OUT ETimeMetric *penMetric);

#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKTIME_H_ */
