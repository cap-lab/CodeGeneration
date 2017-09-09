/*
 * UCTime.h
 *
 *  Created on: 2017. 8. 10.
 *      Author: jej
 */

#ifndef SRC_COMMON_INCLUDE_UCTIME_H_
#define SRC_COMMON_INCLUDE_UCTIME_H_

#include "uem_common.h";

#ifdef __cplusplus
extern "C"
{
#endif


uem_result UCTime_GetCurTimeInMilliSeconds(long long *pllTime);
uem_result UCTime_GetCurTickInMilliSeconds(long long *pllTime);
uem_result UCTime_Sleep(int nMillisec);

#ifdef __cplusplus
}
#endif

#endif /* SRC_COMMON_INCLUDE_UCTIME_H_ */
