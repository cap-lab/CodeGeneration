/*
 * UKLoop.c
 *
 *  Created on: 2018. 8. 30.
 *      Author: DG-SHIN
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <uem_data.h>

#include <UCString.h>

#include <UKLoop.h>
#include <UKTask_internal.h>


static uem_result getParentLoopTaskByCallerTask(STask *pstCallerTask, OUT STask **ppstTask)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstTask = NULL;
	STask *pstParentTask = NULL;

	pstParentTask = pstCallerTask;

	while(pstParentTask != NULL)
	{
		if(pstParentTask->pstLoopInfo != NULL)
		{
			*ppstTask = pstParentTask;
			break;
		}

		if(pstParentTask->pstParentGraph->pstParentTask != NULL)
		{
			pstParentTask = pstParentTask->pstParentGraph->pstParentTask;
		}
		else // pstCallerTask->pstParentGraph->pstParentTask == NULL -> top-level graph
		{
			ERRASSIGNGOTO(result, ERR_UEM_NO_DATA, _EXIT);
		}
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKLoop_GetLoopTaskIteration(IN int nCallerTaskId, IN int nTaskThreadId, OUT int *pnTaskIteration)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstTask = NULL;
	STask *pstCallerTask = NULL;
	int nIndex = 0;

	result = UKTask_GetTaskFromTaskId(nCallerTaskId, &pstCallerTask);
	ERRIFGOTO(result, _EXIT);

	result = getParentLoopTaskByCallerTask(pstCallerTask, &pstTask);
	ERRIFGOTO(result, _EXIT);

	*pnTaskIteration = pstCallerTask->astThreadContext[nTaskThreadId].nCurRunIndex % pstTask->pstLoopInfo->nLoopCount;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKLoop_StopNextIteration(IN int nCallerTaskId)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstCallerTask = NULL;
	STask *pstParentTask = NULL;

	result = UKTask_GetTaskFromTaskId(nCallerTaskId, &pstCallerTask);
	ERRIFGOTO(result, _EXIT);

	IFVARERRASSIGNGOTO(pstCallerTask->pstParentGraph->pstParentTask, NULL, result, ERR_UEM_ILLEGAL_DATA, _EXIT);
	IFVARERRASSIGNGOTO(pstCallerTask->pstParentGraph->pstParentTask->pstLoopInfo, NULL, result, ERR_UEM_ILLEGAL_DATA, _EXIT);

	pstParentTask = pstCallerTask->pstParentGraph->pstParentTask;

	if (pstParentTask->pstLoopInfo->enType != LOOP_TYPE_CONVERGENT) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}
	if (pstParentTask->pstLoopInfo->nDesignatedTaskId != pstCallerTask->nTaskId) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}

	pstParentTask->pstLoopInfo->bDesignatedTaskState = TRUE;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


