/*
 * UFTimer.h
 *
 *  Created on: 2017. 8. 11.
 *      Author: jej
 */

#ifndef SRC_API_INCLUDE_UFTIMER_H_
#define SRC_API_INCLUDE_UFTIMER_H_

#include <uem_common.h>

#ifdef __cplusplus
extern "C"
{
#endif


uem_result UFTimer_GetCurrentTime (IN int nCallerTaskId, OUT long long *pllCurTime);
uem_result UFTimer_SetAlarm (IN int nCallerTaskId, IN int nTimeValue, IN char *pszTimeUnit, OUT int *pnTimerId);
uem_result UFTimer_GetAlarmed (IN int nCallerTaskId, IN int nTimerId, OUT uem_bool *pbTimerPassed);
uem_result UFTimer_Reset (IN int nCallerTaskId, IN int nTimerId);

#ifdef __cplusplus
}
#endif


#endif /* SRC_API_INCLUDE_UFTIMER_H_ */
