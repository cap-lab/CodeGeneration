/*
 * UCThread.c
 *
 *  Created on: 2017. 8. 10.
 *      Author: jej
 */

#include <pthread.h>

#include <UCThread.h>


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

