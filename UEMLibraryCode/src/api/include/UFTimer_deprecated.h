/*
 * UFTimer_deprecated.h
 *
 *  Created on: 2017. 8. 13.
 *      Author: jej
 */

#ifndef SRC_API_INCLUDE_UFTIMER_DEPRECATED_H_
#define SRC_API_INCLUDE_UFTIMER_DEPRECATED_H_

#ifdef __cplusplus
extern "C"
{
#endif

unsigned int SYS_REQ_GET_CURRENT_TIME_BASE(int nCallerTaskId);
int SYS_REQ_SET_TIMER(int nCallerTaskId, unsigned int nTimeValue, char *pszTimeUnit);
int SYS_REQ_GET_TIMER_ALARMED(int nCallerTaskId, unsigned int nTimerId);
void SYS_REQ_RESET_TIMER(int nCallerTaskId, unsigned int nTimerId);

#ifdef __cplusplus
}
#endif

#endif /* SRC_API_INCLUDE_UFTIMER_DEPRECATED_H_ */
