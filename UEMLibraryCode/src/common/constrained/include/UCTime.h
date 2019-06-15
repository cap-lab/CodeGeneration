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
 * @brief Get current tick value in milliseconds.
 *
 * This function retrieves a current tick (uptime of a device) in millisecond time unit.
 *
 * @param[out] ptTime current tick in milliseconds.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UCTime_GetCurTickInMilliSeconds(OUT uem_time *ptTime);

/**
 * @brief Sleep current thread.
 *
 * This function sleeps the current thread for a specific amount of time.
 *
 * @param nMillisec the amount of time to sleep in milliseconds.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UCTime_Sleep(int nMillisec);

#ifdef __cplusplus
}
#endif

#endif /* SRC_COMMON_CONSTRAINED_INCLUDE_UCTIME_H_ */
