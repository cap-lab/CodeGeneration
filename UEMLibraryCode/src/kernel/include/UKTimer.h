/*
 * UKTimer.h
 *
 *  Created on: 2018. 5. 1.
 *      Author: jej
 */

#ifndef SRC_KERNEL_INCLUDE_UKTIMER_H_
#define SRC_KERNEL_INCLUDE_UKTIMER_H_

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
 * @param nCallerTaskId
 * @param nTimeValue
 * @param pszTimeUnit
 * @param[out] pnTimerId
 *
 * @return
 */
uem_result UKTimer_SetAlarm (IN int nCallerTaskId, IN int nTimeValue, IN char *pszTimeUnit, OUT int *pnTimerId);

/**
 * @brief
 *
 * This function
 *
 * @param nCallerTaskId
 * @param nTimerId
 * @param[out] pbTimerPassed
 *
 * @return
 */
uem_result UKTimer_GetAlarmed (IN int nCallerTaskId, IN int nTimerId, OUT uem_bool *pbTimerPassed);

/**
 * @brief
 *
 * This function
 *
 * @param nCallerTaskId
 * @param nTimerId
 *
 * @return
 */
uem_result UKTimer_Reset (IN int nCallerTaskId, IN int nTimerId);

#ifdef __cplusplus
}
#endif


#endif /* SRC_KERNEL_INCLUDE_UKTIMER_H_ */
