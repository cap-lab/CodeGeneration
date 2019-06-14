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

/**
 * @brief (Deprecated) Return the current time base.
 *
 * time is expressed in usec unit, and can be expressed up to 100 seconds.
 *
 * @param nCallerTaskId id of caller task.
 *
 * @return timebase value.
 */
unsigned int SYS_REQ_GET_CURRENT_TIME_BASE(int nCallerTaskId);

/**
 * @brief (Deprecated) Set a timer based on current time.
 *
 * @param nCallerTaskId id of caller task.
 * @param nTimeValue timer value.
 * @param pszTimeUnit timer unit. \n
 * can choose between "S", "MS" ,"US"(default is US).
 *
 * @return timer id.
 */
int SYS_REQ_SET_TIMER(int nCallerTaskId, unsigned int nTimeValue, char *pszTimeUnit);

/**
 * @brief (Deprecated) Check if the set timer expires.
 *
 * If the time elapses, the timer is re-initialized.
 *
 * @param nCallerTaskId id of caller task.
 * @param nTimerId timer id.
 *
 * @return value indicating whether time is passed or not. \n
 * 1 indicates time passed, 0 indicates time not passed.
 */
int SYS_REQ_GET_TIMER_ALARMED(int nCallerTaskId, unsigned int nTimerId);

/**
 * @brief (Deprecated) re-initialized the timer.
 *
 * @param nCallerTaskId id of caller task.
 * @param nTimerId timer id.
 *
 * @return
 */
void SYS_REQ_RESET_TIMER(int nCallerTaskId, unsigned int nTimerId);

#ifdef __cplusplus
}
#endif

#endif /* SRC_API_INCLUDE_UFTIMER_DEPRECATED_H_ */
