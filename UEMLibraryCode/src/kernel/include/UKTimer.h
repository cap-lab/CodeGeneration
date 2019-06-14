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
 * @brief Set a timer based on current time.
 *
 * could be only used in Control task. \n
 * nTimeValue should be greater than 0.
 *
 * This function sets alarm and passes empty timer slot id. \n
 * If there is no empty slot when looking for an empty timer slot, \n
 * passes a slot id that has been checked to be out of time by GetAlarm function.\n
 * If there is no such slot, passes the slot id that has already elapsed. \n
 * If there is no such slot, it returns @ref ERR_UEM_UNAVAILABLE_DATA.
 *
 * @param nCallerTaskId id of caller task.
 * @param nTimeValue timer value.
 * @param pszTimeUnit timer unit.
 * @param[out] pnTimerId  timer id.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_INVALID_PARAM for invalid time unit, timer id, timer value, caller task id. \n
 * @ref ERR_UEM_NO_DATA if task name does not match to any task. \n
 * @ref ERR_UEM_ILLEGAL_CONTROL if caller task is not a Control task. \n
 * @ref ERR_UEM_UNAVAILABLE_DATA for invalid timer slot id. \n
 * (unconstrained kernel code) @ref ERR_UEM_INTERNAL_FAIL if clock_gettime function defined in pthread_time.h failed.
 */
uem_result UKTimer_SetAlarm (IN int nCallerTaskId, IN int nTimeValue, IN char *pszTimeUnit, OUT int *pnTimerId);

/**
 * @brief Check if the set timer expires.
 *
 * This function check if timer corresponding to nTimerid is expired.
 * If the time has passed, the timer is re-initialized.
 *
 * @param nCallerTaskId id of caller task.
 * @param nTimerId timer id.
 * @param[out] pbTimerPassed value indicating whether time is passed or not. \n
 * 1 indicates time passed, 0 indicates time not passed.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_INVALID_PARAM for invalid caller task id, invalid pbTimerPassed pointer, or \n
 * if nTimerId less then 0 or nTimerId greater then or equal to g_nTimerSlotNum, or \n
 * corresponding time value is INVALID_TIME_VALUE. \n
 * @ref ERR_UEM_ILLEGAL_CONTROL if caller task is not a Control task. \n
 * @ref ERR_UEM_NO_DATA if task name does not match to any task.
 */
uem_result UKTimer_GetAlarmed (IN int nCallerTaskId, IN int nTimerId, OUT uem_bool *pbTimerPassed);

/**
 * @brief Re-initialize the timer.
 *
 * This function make the timer slot corresponding to nTimerId empty.
 *
 * @param nCallerTaskId id of caller task.
 * @param nTimerId timer id.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_INVALID_PARAM for invalid caller task id, or \n
 * if nTimerId less then 0 or nTimerId greater then or equal to g_nTimerSlotNum, or \n
 * corresponding time value is INVALID_TIME_VALUE. \n
 * @ref ERR_UEM_ILLEGAL_CONTROL if caller task is not a Control task. \n
 * @ref ERR_UEM_ILLEGAL_DATA if timer id does not match to any task timer slot.
 */
uem_result UKTimer_Reset (IN int nCallerTaskId, IN int nTimerId);

#ifdef __cplusplus
}
#endif


#endif /* SRC_KERNEL_INCLUDE_UKTIMER_H_ */
