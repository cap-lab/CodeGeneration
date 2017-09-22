/*
 * UCThread.c
 *
 *  Created on: 2017. 8. 10.
 *      Author: jej
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif


#include <pthread.h>

#include <UCThread.h>

typedef struct _SThread {
	EUemModuleId enId;
	pthread_t hNativeThread;
} SThread;

uem_result UCThread_Create(FnNativeThread fnThreadRoutine, void *pUserData, HThread *phThread)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UCThread_Destroy(HThread *phThread)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

