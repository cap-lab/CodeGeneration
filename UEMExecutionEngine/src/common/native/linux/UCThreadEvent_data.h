/*
 * UCThreadEvent_data.h
 *
 *  Created on: 2017. 8. 11.
 *      Author: jej
 */

#ifndef SRC_COMMON_NATIVE_LINUX_UCTHREADEVENT_DATA_H_
#define SRC_COMMON_NATIVE_LINUX_UCTHREADEVENT_DATA_H_

#include <uem_common.h>

#include <UCThreadEvent.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct _SThreadEvent {
	EUemModuleId enId;
	uem_bool bIsSet;
	uem_bool bIsStatic;
	pthread_mutex_t hCond;
	pthread_cond_t hMutex;
} SThreadEvent;

SThreadEvent thread_events_data[] = {
	{ID_UEM_THREAD_EVENT, FALSE, TRUE, PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER},
};

HThreadEvent thread_events[] = {
	&thread_events_data[0],
};

HThreadEvent *g_ahStaticThreadEvents = thread_events;

#ifdef __cplusplus
}
#endif

#endif /* SRC_COMMON_NATIVE_LINUX_UCTHREADEVENT_DATA_H_ */
