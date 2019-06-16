/*
 * UKTime.h
 *
 *  Created on: 2018. 9. 6.
 *      Author: chjej202
 */

#ifndef SRC_KERNEL_CONSTRAINED_INCLUDE_UKTIME_H_
#define SRC_KERNEL_CONSTRAINED_INCLUDE_UKTIME_H_

#include <uem_common.h>

#include <uem_enum.h>


#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief  Calculate next runtime.
 *
 * This function calculates next runtime with given value and metric. \n
 * The calculation method depends on the input metric unit. \n
 * If task time metric is TIME_METRIC_COUNT or TIME_METRIC_CYCLE, it is calculated as 1ms, \n
 * and the other cases will be calculated according to each unit.
 *
 *
 * @param tPrevTime previous run time.
 * @param nPeriod task run period.
 * @param enPeriodMetric period metric.
 * @param[out] ptNextTime next run time.
 * @param[out] pnNextMaxRunCount available run count until next run time.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_ILLEGAL_DATA if invalid enPeriodMetric argument comes.
 *
 */
uem_result UKTime_GetNextTimeByPeriod(uem_time tPrevTime, int nPeriod, ETimeMetric enPeriodMetric,
											OUT uem_time *ptNextTime, OUT int *pnNextMaxRunCount);

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_CONSTRAINED_INCLUDE_UKTIME_H_ */
