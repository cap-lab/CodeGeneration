/*
 * UKCPUGeneralTaskManager.c
 *
 *  Created on: 2018. 1. 16.
 *      Author: jej
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

typedef struct _SCPUGeneralTaskManager *HCPUGeneralTaskManager;

uem_result UKCPUGeneralTaskManager_Create(IN OUT HCPUGeneralTaskManager *phManager)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKCPUGeneralTaskManager_RegisterTask(HCPUGeneralTaskManager hManager, SMappedGeneralTaskInfo *pstMappedTask)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKCPUGeneralTaskManager_CreateThread(HCPUGeneralTaskManager hManager, STask *pstTargetTask)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKCPUGeneralTaskManager_ChangeState(HCPUGeneralTaskManager hManager, STask *pstTargetTask, , EInternalTaskState enTaskState)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKCPUGeneralTaskManager_ActivateThread(HCPUGeneralTaskManager hManager, STask *pstTargetTask)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKCPUGeneralTaskManager_GetTaskState(HCPUGeneralTaskManager hManager, STask *pstTargetTask, EInternalTaskState *penTaskState)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKCPUGeneralTaskManager_DestroyThread(HCPUGeneralTaskManager hManager, STask *pstTargetTask)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}



uem_result UKCPUGeneralTaskManager_Destroy(IN OUT HCPUGeneralTaskManager *phManager)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}
