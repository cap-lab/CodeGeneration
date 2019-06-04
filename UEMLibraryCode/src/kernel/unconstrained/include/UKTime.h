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
 * @brief
 *
 * This function
 *
 * @param llPrevTime
 * @param nPeriod
 * @param enPeriodMetric
 * @param[out] pllNextTime
 * @param[out] pnNextMaxRunCount
 *
 * @return
 */
uem_result UKTime_GetNextTimeByPeriod(long long llPrevTime, int nPeriod, ETimeMetric enPeriodMetric,
											OUT long long *pllNextTime, OUT int *pnNextMaxRunCount);

/**
 * @brief
 *
 * This function
 *
 * @param[out] pnValue
 * @param[out] penMetric
 *
 * @return
 */
uem_result UKTime_GetProgramExecutionTime(OUT int *pnValue, OUT ETimeMetric *penMetric);

#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKTIME_H_ */
