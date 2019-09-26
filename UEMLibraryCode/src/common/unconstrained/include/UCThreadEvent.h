/*
 * UCThread.h
 *
 *  Created on: 2017. 8. 5.
 *      Author: jej
 */

#ifndef SRC_COMMON_UNCONSTRAINED_INCLUDE_UCTHREADEVENT_H_
#define SRC_COMMON_UNCONSTRAINED_INCLUDE_UCTHREADEVENT_H_

#include "uem_common.h"

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct _SThreadEvent *HThreadEvent;

extern HThreadEvent *g_ahStaticThreadEvents;

/**
 * @brief Create an event.
 *
 * This function creates an event which can performs a synchronization among threads. \n
 * The event is basically working as a one-to-many (single event setter-multiple event handler) \n
 * thread synchronization. \n
 * The behavior is similar to SetEvent() and WaitForSingleObject() used in Windows. \n
 * If SetEvent is triggered before WaitForSingleObject is called, the event handler \n
 * which calls WaitForSingleObject is not blocked and handles the event.
 *
 * @param phEvent [out] an event handle to be created.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_OUT_OF_MEMORY, @ref ERR_UEM_INTERNAL_FAIL.
 *
 */
uem_result UCThreadEvent_Create(OUT HThreadEvent *phEvent);

/**
 * @brief Set an event.
 *
 * This function sets an event which wakes up all the threads waiting event.
 *
 * @param hEvent an event handle.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INTERNAL_FAIL.
 */
uem_result UCThreadEvent_SetEvent(HThreadEvent hEvent);

/**
 * @brief Wait an event infinitely.
 *
 * This function waits an event infinitely. A thread is blocked inside this function until an event is set.
 *
 * @param hEvent an event handle.
 *
 * @return @ref ERR_UEM_NOERROR is returned when the event is received. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_MUTEX_ERROR.
 */
uem_result UCThreadEvent_WaitEvent(HThreadEvent hEvent);

/**
 * @brief Wait an event in specific time.
 *
 * This function waits an event until @a llSleepTimeMs milliseconds.
 * If the time is exceeded, this function returns @ref ERR_UEM_TIME_EXPIRED.
 *
 * @param hEvent an event handle.
 * @param llSleepTimeMs maximum time to be blocked.
 *
 * @return @ref ERR_UEM_NOERROR is returned when the event is received. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_MUTEX_ERROR, \n
 *         @ref ERR_UEM_TIME_EXPIRED.
 *         @ref ERR_UEM_TIME_EXPIRED is occurred when the event is not received until timeout.
 */
uem_result UCThreadEvent_WaitTimeEvent(HThreadEvent hEvent, long long llSleepTimeMs);

/**
 * @brief Destroy an event.
 *
 * This function destroys an event.
 *
 * @param[in,out] phEvent an event handle to be destroyed.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_MUTEX_ERROR.
 */
uem_result UCThreadEvent_Destroy(IN OUT HThreadEvent *phEvent);


uem_result UCThreadEvent_ClearEvent(HThreadEvent hEvent);

#ifdef __cplusplus
}
#endif

#endif /* SRC_COMMON_UNCONSTRAINED_INCLUDE_UCTHREADEVENT_H_ */
