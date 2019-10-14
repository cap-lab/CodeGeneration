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

#include <UKLoopModelController.h>


static uem_result getParentLoopTaskGraphByCallerTask(STask *pstCallerTask, OUT STaskGraph **ppstTaskGraph)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STaskGraph *pstParentTaskGraph = NULL;

	pstParentTaskGraph = pstCallerTask->pstParentGraph;

	while(pstParentTaskGraph->pstParentTask != NULL)
	{
		if(pstParentTaskGraph->enControllerType == CONTROLLER_TYPE_DYNAMIC_CONVERGENT_LOOP ||
			pstParentTaskGraph->enControllerType == CONTROLLER_TYPE_STATIC_CONVERGENT_LOOP ||
			pstParentTaskGraph->enControllerType == CONTROLLER_TYPE_DYNAMIC_DATA_LOOP ||
			pstParentTaskGraph->enControllerType == CONTROLLER_TYPE_STATIC_DATA_LOOP)
		{
			*ppstTaskGraph = pstParentTaskGraph;
			break;
		}

		if(pstParentTaskGraph->pstParentTask != NULL)
		{
			pstParentTaskGraph = pstParentTaskGraph->pstParentTask->pstParentGraph;
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
	STaskGraph *pstTaskGraph = NULL;
	STask *pstCallerTask = NULL;
	SLoopController *pstLoopController = NULL;

	result = UKTask_GetTaskFromTaskId(nCallerTaskId, &pstCallerTask);
	ERRIFGOTO(result, _EXIT);

	result = getParentLoopTaskGraphByCallerTask(pstCallerTask, &pstTaskGraph);
	ERRIFGOTO(result, _EXIT);

	pstLoopController = (SLoopController *)pstTaskGraph->pController;

	*pnTaskIteration = pstCallerTask->astThreadContext[nTaskThreadId].nCurRunIndex % pstLoopController->pstLoopInfo->nLoopCount;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKLoop_StopNextIteration(IN int nCallerTaskId)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstCallerTask = NULL;
	STaskGraph *pstParentTaskGraph = NULL;
	SLoopController *pstLoopController = NULL;

	result = UKTask_GetTaskFromTaskId(nCallerTaskId, &pstCallerTask);
	ERRIFGOTO(result, _EXIT);

	IFVARERRASSIGNGOTO(pstCallerTask->pstParentGraph->pstParentTask, NULL, result, ERR_UEM_ILLEGAL_DATA, _EXIT);

	pstParentTaskGraph = pstCallerTask->pstParentGraph;

	if(pstParentTaskGraph->enControllerType != CONTROLLER_TYPE_DYNAMIC_CONVERGENT_LOOP &&
	pstParentTaskGraph->enControllerType != CONTROLLER_TYPE_STATIC_CONVERGENT_LOOP)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
	}

	pstLoopController =  (SLoopController *) pstParentTaskGraph->pController;

	if (pstLoopController->pstLoopInfo->nDesignatedTaskId != pstCallerTask->nTaskId) {
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_CONTROL, _EXIT);
	}

	pstLoopController->pstLoopInfo->bDesignatedTaskState = TRUE;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


