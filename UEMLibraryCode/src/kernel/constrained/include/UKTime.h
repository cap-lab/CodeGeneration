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

uem_result UKTime_GetNextTimeByPeriod(uem_time tPrevTime, int nPeriod, ETimeMetric enPeriodMetric,
											OUT uem_time *ptNextTime, OUT int *pnNextMaxRunCount);

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_CONSTRAINED_INCLUDE_UKTIME_H_ */
