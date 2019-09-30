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
			ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_CONTROL, _EXIT);
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
		if(pstCommon != NULL && pstCommon->pstFunctionSet != NULL)
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



