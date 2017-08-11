/*
 * UCThreadMutex_data.h
 *
 *  Created on: 2017. 8. 11.
 *      Author: jej
 */

#ifndef SRC_COMMON_NATIVE_LINUX_UCTHREADMUTEX_DATA_H_
#define SRC_COMMON_NATIVE_LINUX_UCTHREADMUTEX_DATA_H_

#include <uem_common.h>

#include <UCThreadMutex.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct _SThreadMutex {
	EUemModuleId enId;
	uem_bool bInMutex;
	uem_bool bIsStatic;
    pthread_mutex_t hMutex;
} SThreadMutex;


static SThreadMutex thread_mutexes_data[] = {
	{ID_UEM_THREAD_MUTEX, FALSE, TRUE, PTHREAD_MUTEX_INITIALIZER},
};

static HThreadMutex thread_mutexes[] = {
	&thread_mutexes_data[0],
};

HThreadMutex *g_ahStaticThreadMutexes = thread_mutexes;

#ifdef __cplusplus
}
#endif

#endif /* SRC_COMMON_NATIVE_LINUX_UCTHREADMUTEX_DATA_H_ */
