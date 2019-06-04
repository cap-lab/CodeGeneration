/*
 * UCTime.h
 *
 *  Created on: 2018. 9. 6.
 *      Author: chjej202
 */

#ifndef SRC_COMMON_CONSTRAINED_INCLUDE_UCTIME_H_
#define SRC_COMMON_CONSTRAINED_INCLUDE_UCTIME_H_

#include <uem_common.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief
 *
 * This function
 *
 * @param ptTime
 *
 * @return
 */
uem_result UCTime_GetCurTickInMilliSeconds(uem_time *ptTime);

/**
 * @brief
 *
 * This function
 *
 * @param nMillisec
 *
 * @return
 */
uem_result UCTime_Sleep(int nMillisec);

#ifdef __cplusplus
}
#endif

#endif /* SRC_COMMON_CONSTRAINED_INCLUDE_UCTIME_H_ */
