/*
 * CPUThreadPool.c
 *
 *  Created on: 2017. 9. 19.
 *      Author: jej
 */



#include <uem_common.h>

#include <uem_data.h>

#include <UKCPUThreadPool.h>


uem_result UKCPUThreadPool_Create(OUT HCPUThreadPool *phCPUThreadPool)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKCPUThreadPool_RegisterTask(HCPUThreadPool hCPUThreadPool, STask *pstTask, int nCPUId)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKCPUThreadPool_RegisterCompositeTask(HCPUThreadPool hCPUThreadPool, SScheduledTasks *pstScheduledTasks, int nCPUId)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKCPUThreadPool_Destroy(IN OUT HCPUThreadPool *phCPUThreadPool)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


