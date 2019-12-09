/*
 * UKModelController.c
 *
 *  Created on: 2019. 9. 30.
 *      Author: jej
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <uem_data.h>
#include <UCThreadMutex.h>

#include <UKModelController.h>
#include <UKCPUTaskCommon.h>


uem_result UKModelController_GetTopLevelGraph(STaskGraph *pstLeafTaskGraph, OUT STaskGraph **ppstGraph)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STaskGraph *pstHighestGraph = NULL;
	STaskGraph *pstTaskGraph = NULL;
	SModelControllerCommon *pstCommon = NULL;

	pstTaskGraph = pstLeafTaskGraph;

	while(pstTaskGraph != NULL)
	{
		switch(pstTaskGraph->enControllerType)
		{
		case CONTROLLER_TYPE_VOID:
			// skip
			break;
		case CONTROLLER_TYPE_CONTROL_TASK_INCLUDED:
		case CONTROLLER_TYPE_STATIC_MODE_TRANSITION:
		case CONTROLLER_TYPE_STATIC_CONVERGENT_LOOP:
		case CONTROLLER_TYPE_STATIC_DATA_LOOP:
			pstCommon = (SModelControllerCommon *) pstTaskGraph->pController;
			IFVARERRASSIGNGOTO(pstCommon, NULL, result, ERR_UEM_ILLEGAL_DATA, _EXIT);
			pstHighestGraph = pstTaskGraph;
			break;
		case CONTROLLER_TYPE_DYNAMIC_MODE_TRANSITION:
		case CONTROLLER_TYPE_DYNAMIC_CONVERGENT_LOOP:
		case CONTROLLER_TYPE_DYNAMIC_DATA_LOOP:
			pstCommon = (SModelControllerCommon *) pstTaskGraph->pController;
			IFVARERRASSIGNGOTO(pstCommon, NULL, result, ERR_UEM_ILLEGAL_DATA, _EXIT);
			pstHighestGraph = pstTaskGraph;
			break;
		}

		if(pstTaskGraph->pstParentTask != NULL)
		{
			pstTaskGraph = pstTaskGraph->pstParentTask->pstParentGraph;
		}
		else
		{
			pstTaskGraph = NULL;
		}
	}

	*ppstGraph = pstHighestGraph;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKModelController_GetTopLevelLockHandle(STaskGraph *pstLeafTaskGraph, OUT HThreadMutex *phMutex)
{
	uem_result result = ERR_UEM_UNKNOWN;
	HThreadMutex hCurrentHighestLock = NULL;
	STaskGraph *pstTaskGraph = NULL;
	SModelControllerCommon *pstCommon = NULL;

	pstTaskGraph = pstLeafTaskGraph;

	while(pstTaskGraph != NULL)
	{
		switch(pstTaskGraph->enControllerType)
		{
		case CONTROLLER_TYPE_VOID:
			// skip
			break;
		case CONTROLLER_TYPE_CONTROL_TASK_INCLUDED:
		case CONTROLLER_TYPE_STATIC_MODE_TRANSITION:
		case CONTROLLER_TYPE_STATIC_CONVERGENT_LOOP:
		case CONTROLLER_TYPE_STATIC_DATA_LOOP:
			pstCommon = (SModelControllerCommon *) pstTaskGraph->pController;
			IFVARERRASSIGNGOTO(pstCommon, NULL, result, ERR_UEM_ILLEGAL_DATA, _EXIT);
			hCurrentHighestLock = pstCommon->hMutex;
			break;
		case CONTROLLER_TYPE_DYNAMIC_MODE_TRANSITION:
		case CONTROLLER_TYPE_DYNAMIC_CONVERGENT_LOOP:
		case CONTROLLER_TYPE_DYNAMIC_DATA_LOOP:
			pstCommon = (SModelControllerCommon *) pstTaskGraph->pController;
			IFVARERRASSIGNGOTO(pstCommon, NULL, result, ERR_UEM_ILLEGAL_DATA, _EXIT);
			hCurrentHighestLock = pstCommon->hMutex;
			break;
		}

		if(pstTaskGraph->pstParentTask != NULL)
		{
			pstTaskGraph = pstTaskGraph->pstParentTask->pstParentGraph;
		}
		else
		{
			pstTaskGraph = NULL;
		}
	}

	*phMutex = hCurrentHighestLock;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result callClearFunction(STask *pstTask, void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STaskGraph *pstCurrentTaskGraph = NULL;
	SModelControllerCommon *pstCommon = NULL;

	if(pstTask->pstSubGraph != NULL)
	{
		pstCurrentTaskGraph = pstTask->pstSubGraph;

		if(pstCurrentTaskGraph->pController != NULL)
		{
			pstCommon = (SModelControllerCommon *) pstCurrentTaskGraph->pController;

			if(pstCommon->pstFunctionSet != NULL && pstCommon->pstFunctionSet->fnClear != NULL)
			{
				result = pstCommon->pstFunctionSet->fnClear(pstCurrentTaskGraph);
				ERRIFGOTO(result, _EXIT);

				pstCommon->nCurrentIteration = 0;
			}
		}
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKModelController_CallSubGraphClearFunctions(STaskGraph *pstTaskGraph)
{
	uem_result result = ERR_UEM_UNKNOWN;
	HThreadMutex hTaskGraphLock = NULL;

	result = UKModelController_GetTopLevelLockHandle(pstTaskGraph, &hTaskGraphLock);
	ERRIFGOTO(result, _EXIT);

	if(hTaskGraphLock != NULL)
	{
		result = UCThreadMutex_Lock(hTaskGraphLock);
		ERRIFGOTO(result, _EXIT);
	}

	result = UKCPUTaskCommon_TraverseSubGraphTasks(pstTaskGraph->pstParentTask, callClearFunction, NULL);
	ERRIFGOTO(result, _EXIT_LOCK);

	result = callClearFunction(pstTaskGraph->pstParentTask, NULL);
	ERRIFGOTO(result, _EXIT_LOCK);

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
	if(hTaskGraphLock != NULL)
	{
		UCThreadMutex_Unlock(hTaskGraphLock);
	}
_EXIT:
	return result;
}


uem_result UKModelController_TraverseAndCallFunctions(STaskGraph *pstLeafTaskGraph, HThreadMutex hTaskGraphLock,
													FnTraverseModelControllerFunctions fnFunction, void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STaskGraph *pstTaskGraph = NULL;
	SModelControllerCommon *pstCommon = NULL;

	pstTaskGraph = pstLeafTaskGraph;

	if(hTaskGraphLock != NULL)
	{
		result = UCThreadMutex_Lock(hTaskGraphLock);
		ERRIFGOTO(result, _EXIT);
	}

	while(pstTaskGraph != NULL)
	{
		pstCommon = (SModelControllerCommon *) pstTaskGraph->pController;
		if(pstCommon != NULL)
		{
			result = fnFunction(pstTaskGraph, pstTaskGraph->enControllerType, pstCommon->pstFunctionSet, pUserData);
			ERRIFGOTO(result, _EXIT);
		}

		if(pstTaskGraph->pstParentTask != NULL)
		{
			pstTaskGraph = pstTaskGraph->pstParentTask->pstParentGraph;
		}
		else
		{
			pstTaskGraph = NULL;
		}
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	if(hTaskGraphLock != NULL)
	{
		UCThreadMutex_Unlock(hTaskGraphLock);
	}
	return result;
}



