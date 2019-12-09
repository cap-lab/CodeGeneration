/*
 * UCTime.h
 *
 *  Created on: 2017. 8. 10.
 *      Author: jej
 */

#ifndef SRC_COMMON_UNCONSTRAINED_INCLUDE_UCTIME_H_
#define SRC_COMMON_UNCONSTRAINED_INCLUDE_UCTIME_H_

#include "uem_common.h"

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief Get current UNIX time in milliseconds
 *
 * This function retrieves a current UNIX time in milliseconds time unit. \n
 * This function can be used for time measurement.
 *
 * @param[out] ptTime time in milliseconds.
 *
 * @return  @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *          Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_INTERNAL_FAIL. \n
 *          @ref ERR_UEM_INTERNAL_FAIL can be occurred when the internal time-measuring operation is failed.
 */
uem_result UCTime_GetCurTimeInMilliSeconds(OUT uem_time *ptTime);

/**
 * @brief Get current tick value in milliseconds.
 *
 * This function retrieves a current tick (uptime of a device) in millisecond time unit. \n
 * This function can be used for time measurement.
 *
 * @param[out] ptTime  current tick in milliseconds.
 *
 * @return  @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *          Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_INTERNAL_FAIL. \n
 *          @ref ERR_UEM_INTERNAL_FAIL can be occurred when the internal tick-measuring operation is failed.
 */
uem_result UCTime_GetCurTickInMilliSeconds(OUT uem_time *ptTime);

/**
 * @brief Sleep current thread.
 *
 * This function sleeps the current thread for a specific amount of time.
 *
 * @param nMillisec the amount of time to sleep in milliseconds.
 *
 * @return  @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *          Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_INTERRUPT, @ref ERR_UEM_INTERNAL_FAIL. \n
 *          @ref ERR_UEM_INTERRUPT can be occurred if the internal sleep operation meets the interrupt during sleep.
 */
uem_result UCTime_Sleep(int nMillisec);

#ifdef __cplusplus
}
#endif

#endif /* SRC_COMMON_UNCONSTRAINED_INCLUDE_UCTIME_H_ */
