/*
 * UCThread.h
 *
 *  Created on: 2017. 8. 10.
 *      Author: jej
 */

#ifndef SRC_COMMON_INCLUDE_UCTHREAD_H_
#define SRC_COMMON_INCLUDE_UCTHREAD_H_

#include "uem_common.h";

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct _SThread {
	int enId;
	pthread_t hNativeThread;
} SThread;

typedef SThread *HThread;

typedef void * (*FnNativeThread)(void *pData);

uem_result UCThread_Create(FnNativeThread fnThreadRoutine, void *pUserData, HThread *phThread);
uem_result UCThread_Destroy(HThread *phThread);

#ifdef __cplusplus
}
#endif

#endif /* SRC_COMMON_INCLUDE_UCTHREAD_H_ */
