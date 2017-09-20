/*
 * CPUThreadPool.c
 *
 *  Created on: 2017. 9. 19.
 *      Author: jej
 */

#include <uem_common.h>

#include <uem_data.h>

#include "CPUThreadPool.h"


uem_result CPUThreadPool_Create(OUT HCPUThreadPool *phCPUThreadPool)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result CPUThreadPool_RegisterTask(HCPUThreadPool hCPUThreadPool, STask *pstTask, int nCPUId)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result CPUThreadPool_RegisterCompositeTask(HCPUThreadPool hCPUThreadPool, SScheduledTasks *pstScheduledTasks, int nCPUId)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result CPUThreadPool_Destroy(IN OUT HCPUThreadPool *phCPUThreadPool)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


