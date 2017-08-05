/*
 * UCThreadEvent.c
 *
 *  Created on: 2017. 8. 5.
 *      Author: jej
 */

#include <pthread.h>

#include <UCThreadEvent.h>

typedef struct _SThreadEvent {
	int id;
	uem_bool bIsSet;
	pthread_mutex_t hCond;
	pthread_cond_t hMutex;
} SThreadEvent;


typedef SThreadEvent *HThreadEvent;

uem_result UCThreadEvent_Create(HThreadEvent *phEvent)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UCThreadEvent_SetEvent(HThreadEvent hEvent)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UCThreadEvent_Create(HThreadEvent *phEvent)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UCThreadEvent_WaitTimeEvent(HThreadEvent hEvent, long long llSleepTimeMs)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UCThreadEvent_Destroy(HThreadEvent *phEvent)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


