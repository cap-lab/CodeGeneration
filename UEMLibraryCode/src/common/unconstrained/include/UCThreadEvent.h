/*
 * UCThread.h
 *
 *  Created on: 2017. 8. 5.
 *      Author: jej
 */

#ifndef SRC_COMMON_INCLUDE_UCTHREADEVENT_H_
#define SRC_COMMON_INCLUDE_UCTHREADEVENT_H_

#include "uem_common.h"

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct _SThreadEvent *HThreadEvent;

extern HThreadEvent *g_ahStaticThreadEvents;

/**
 * @brief
 *
 * This function
 *
 * @param phEvent [out]
 *
 * @return
 */
uem_result UCThreadEvent_Create(OUT HThreadEvent *phEvent);

/**
 * @brief
 *
 * This function
 *
 * @param hEvent
 *
 * @return
 */
uem_result UCThreadEvent_SetEvent(HThreadEvent hEvent);

/**
 * @brief
 *
 * This function
 *
 * @param hEvent
 *
 * @return
 */
uem_result UCThreadEvent_WaitEvent(HThreadEvent hEvent);

/**
 * @brief
 *
 * This function
 *
 * @param hEvent
 * @param llSleepTimeMs
 *
 * @return
 */
uem_result UCThreadEvent_WaitTimeEvent(HThreadEvent hEvent, long long llSleepTimeMs);

/**
 * @brief
 *
 * This function
 *
 * @param[in,out] phEvent
 *
 * @return
 */
uem_result UCThreadEvent_Destroy(IN OUT HThreadEvent *phEvent);

#ifdef __cplusplus
}
#endif

#endif /* SRC_COMMON_INCLUDE_UCTHREADEVENT_H_ */
