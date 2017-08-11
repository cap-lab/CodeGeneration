/*
 * UCThread.h
 *
 *  Created on: 2017. 8. 5.
 *      Author: jej
 */

#ifndef SRC_COMMON_INCLUDE_UCTHREADEVENT_H_
#define SRC_COMMON_INCLUDE_UCTHREADEVENT_H_

#include "uem_common.h";

#ifdef __cplusplus
extern "C"
{
#endif

typedef void *HThreadEvent;

extern HThreadEvent *g_ahStaticThreadEvents;

uem_result UCThreadEvent_Create(HThreadEvent *phEvent);
uem_result UCThreadEvent_SetEvent(HThreadEvent hEvent);
uem_result UCThreadEvent_WaitEvent(HThreadEvent hEvent);
uem_result UCThreadEvent_WaitTimeEvent(HThreadEvent hEvent, long long llSleepTimeMs);
uem_result UCThreadEvent_Destroy(HThreadEvent *phEvent);

#ifdef __cplusplus
}
#endif

#endif /* SRC_COMMON_INCLUDE_UCTHREADEVENT_H_ */
