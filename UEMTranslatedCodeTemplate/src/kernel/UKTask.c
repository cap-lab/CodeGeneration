/*
 * UKTask.c
 *
 *  Created on: 2017. 9. 2.
 *      Author: jej
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <uem_data.h>

#include <UKTask.h>


uem_result UKTask_GetTaskFromTaskId(int nTaskId, STask **ppstTask)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;

#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(ppstTask, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	if(nTaskId < 0) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
#endif

	for(nLoop = 0 ; nLoop <  g_nTaskIdToTaskNum ; nLoop++)
	{
		if(nTaskId == g_astTaskIdToTask[nLoop].nTaskId)
		{
			*ppstTask = g_astTaskIdToTask[nLoop].pstTask;
			break;
		}
	}

	if(nLoop == g_nTaskIdToTaskNum)
	{
		ERRASSIGNGOTO(result, ERR_UEM_NO_DATA, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKTask_TraverseAllTasks(FnTaskTraverse fnCallback, void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;

	for(nLoop = 0 ; nLoop <  g_nTaskIdToTaskNum ; nLoop++)
	{
		result = fnCallback(g_astTaskIdToTask[nLoop].pstTask, pUserData);
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

