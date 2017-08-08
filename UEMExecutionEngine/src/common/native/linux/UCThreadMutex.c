/*
 * UCThreadMutex.c
 *
 *  Created on: 2017. 8. 5.
 *      Author: jej
 */

#include <pthread.h>

#include <UCThreadMutex.h>

typedef struct _SThreadMutex {
	int enId;
	uem_bool bInMutex;
    pthread_mutex_t hMutex;
} SThreadMutex;


typedef SThreadMutex *HThreadMutex;


uem_result UCThreadMutex_Create(HThreadMutex *phMutex)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UCThreadMutex_Lock(HThreadMutex hMutex)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UCThreadMutex_Unlock(HThreadMutex hMutex)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UCThreadMutex_Destroy(HThreadMutex *phMutex)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


