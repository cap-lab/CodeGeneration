/*
 * UKModeTransition.c
 *
 *  Created on: 2017. 8. 28.
 *      Author: jej
 */

#include <uem_common.h>

#include <UCString.h>

#include <uem_data.h>

#include <UKTask.h>


static int findIndexByIteration(SModeTransitionMachine *pstModeTransition, int nCurrentIteration, OUT int *pnHistoryEnd)
{
	int nLoop = 0;
	int nHistoryEnd;
	int nCheckNum = 0;
	int nIndex = INVALID_ARRAY_INDEX;

	nHistoryEnd = pstModeTransition->nCurHistoryStartIndex + pstModeTransition->nCurHistoryLen - 1;

	if(nHistoryEnd >= MODE_TRANSITION_ARRAY_SIZE)
	{
		nHistoryEnd = nHistoryEnd - MODE_TRANSITION_ARRAY_SIZE;
	}

	for(nLoop = nHistoryEnd; nCheckNum < pstModeTransition->nCurHistoryLen ; nLoop--)
	{
		//printf("pstModeTransition->astModeTransition[%d]: mode: %d, iteration: %d, nCurrentIteration: %d\n", nLoop, pstModeTransition->astModeTransition[nLoop].nModeIndex, pstModeTransition->astModeTransition[nLoop].nIteration, nCurrentIteration);
		if(pstModeTransition->astModeTransition[nLoop].nIteration <= nCurrentIteration)
		{
			nIndex = nLoop;
			break;
		}

		if(nLoop <= 0)
		{
			nLoop = MODE_TRANSITION_ARRAY_SIZE;
		}
		nCheckNum++;
	}

	if(pnHistoryEnd != NULL)
	{
		*pnHistoryEnd = nHistoryEnd;
	}

	return nIndex;
}


uem_result UKModeTransition_GetCurrentModeIndexByIteration(SModeTransitionMachine *pstModeTransition, int nCurrentIteration, OUT int *pnModeIndex)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nHistoryEnd;
	int nIndex = 0;

	nIndex = findIndexByIteration(pstModeTransition, nCurrentIteration, &nHistoryEnd);

	if(nIndex == INVALID_ARRAY_INDEX && nCurrentIteration != 0)
	{
		printf("aaa nHistoryEnd: %d, pstModeTransition->nCurHistoryStartIndex: %d\n", nHistoryEnd, pstModeTransition->nCurHistoryStartIndex);
		printf("aaa nCurrentIteration: %d\n", nCurrentIteration);
		UEMASSIGNGOTO(result, ERR_UEM_NOT_FOUND, _EXIT);
	}
	else if(nIndex == INVALID_ARRAY_INDEX && nCurrentIteration == 0)
	{
		*pnModeIndex = 0;
	}
	else
	{
		*pnModeIndex = pstModeTransition->astModeTransition[nIndex].nModeIndex;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKModeTransition_GetNextModeStartIndexByIteration(SModeTransitionMachine *pstModeTransition, int nCurrentIteration, OUT int *pnModeIndex, OUT int *pnStartIteration)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nHistoryEnd;
	int nIndex = 0;

	nIndex = findIndexByIteration(pstModeTransition, nCurrentIteration, &nHistoryEnd);

	if(nIndex == INVALID_ARRAY_INDEX && nCurrentIteration != 0)
	{
		printf("aaa nHistoryEnd: %d, pstModeTransition->nCurHistoryStartIndex: %d\n", nHistoryEnd, pstModeTransition->nCurHistoryStartIndex);
		printf("aaa nCurrentIteration: %d\n", nCurrentIteration);
		UEMASSIGNGOTO(result, ERR_UEM_NOT_FOUND, _EXIT);
	}
	else if(nIndex == nHistoryEnd)
	{
		// no next mode start index
		UEMASSIGNGOTO(result, ERR_UEM_NO_DATA, _EXIT);
	}

	nIndex++;
	if(nIndex >= MODE_TRANSITION_ARRAY_SIZE)
	{
		nIndex = 0;
	}

	if(pnStartIteration != NULL)
	{
		*pnStartIteration = pstModeTransition->astModeTransition[nIndex].nIteration;
	}
	if(pnModeIndex != NULL)
	{
		*pnModeIndex = pstModeTransition->astModeTransition[nIndex].nModeIndex;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKModeTransition_GetCurrentModeName (IN int nCallerTaskId, IN char *pszTaskName, OUT char **ppszModeName)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstTask = NULL;
	int nCurModeIndex = INVALID_ARRAY_INDEX;
	int nCurrentIteration;
	/*STask *pstCallerTask = NULL;

	result = UKTask_GetTaskFromTaskId(nCallerTaskId, &pstCallerTask);
	ERRIFGOTO(result, _EXIT);

	result = UKTask_GetTaskByTaskNameAndCallerTask(pstCallerTask, pszTaskName, &pstTask);
	ERRIFGOTO(result, _EXIT);*/

	result = UKTask_GetTaskFromTaskName(pszTaskName, &pstTask);
	ERRIFGOTO(result, _EXIT);

	nCurrentIteration = pstTask->nCurIteration;

	if(pstTask->pstMTMInfo == NULL)
	{
		IFVARERRASSIGNGOTO(pstTask->pstParentGraph->pstParentTask, NULL, result, ERR_UEM_ILLEGAL_DATA, _EXIT);
		IFVARERRASSIGNGOTO(pstTask->pstParentGraph->pstParentTask->pstMTMInfo, NULL, result, ERR_UEM_ILLEGAL_DATA, _EXIT);

		pstTask = pstTask->pstParentGraph->pstParentTask;
	}

	result = UCThreadMutex_Lock(pstTask->hMutex);
	ERRIFGOTO(result, _EXIT);

	if(pstTask->bStaticScheduled == TRUE)
	{
		nCurModeIndex = pstTask->pstMTMInfo->nCurModeIndex;
	}
	else
	{
		result = UKModeTransition_GetCurrentModeIndexByIteration(pstTask->pstMTMInfo, nCurrentIteration, &nCurModeIndex);
		if(result == ERR_UEM_NOT_FOUND && nCurrentIteration == 0)
		{
			nCurModeIndex = pstTask->pstMTMInfo->nCurModeIndex;
			result = ERR_UEM_NOERROR;
		}
		// ignore error check here to unlock the lock
	}

	UCThreadMutex_Unlock(pstTask->hMutex);
	ERRIFGOTO(result, _EXIT);

	*ppszModeName = pstTask->pstMTMInfo->astModeMap[nCurModeIndex].pszModeName;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

int UKModeTransition_GetModeIndexByModeId(SModeTransitionMachine *pstModeTransition, int nModeId)
{
	int nLoop = 0;
	int nModeIndex = INVALID_MODE_ID;
	int nModeLen = pstModeTransition->nNumOfModes;

	for(nLoop = 0 ; nLoop < nModeLen ; nLoop++)
	{
		if(pstModeTransition->astModeMap[nLoop].nModeId == nModeId)
		{
			nModeIndex = nLoop;
			break;
		}
	}

	return nModeIndex;
}


int UKModeTransition_GetVariableIndexByName(SModeTransitionMachine *pstModeTransition, char *pszVariableName)
{
	int nLoop = 0;
	int nVariableIndex = INVALID_MODE_ID;
	int nVariableLen = pstModeTransition->nNumOfIntVariables;
	uem_string_struct strVariableName;
	uem_string_struct strTargetVariableName;
	uem_result result = ERR_UEM_UNKNOWN;

	result = UCString_New(&strVariableName, pszVariableName, UEMSTRING_MAX);
	ERRIFGOTO(result, _EXIT);

	for(nLoop = 0 ; nLoop < nVariableLen ; nLoop++)
	{
		result = UCString_New(&strTargetVariableName, pstModeTransition->astVarIntMap[nLoop].pszVariableName, UEMSTRING_MAX);
		ERRIFGOTO(result, _EXIT);

		if(UCString_IsEqual(&strTargetVariableName, &strVariableName) == TRUE)
		{
			nVariableIndex = nLoop;
			break;
		}
	}
_EXIT:
	return nVariableIndex;
}

EModeState UKModeTransition_GetModeState(int nTaskId)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstTask = NULL;
	EModeState enModeState = MODE_STATE_TRANSITING;

	result = UKTask_GetTaskFromTaskId(nTaskId, &pstTask);
	ERRIFGOTO(result, _EXIT);

	IFVARERRASSIGNGOTO(pstTask->pstMTMInfo, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	result = UCThreadMutex_Lock(pstTask->hMutex);
	ERRIFGOTO(result, _EXIT);

	enModeState = pstTask->pstMTMInfo->enModeState;

	result = UCThreadMutex_Unlock(pstTask->hMutex);
	ERRIFGOTO(result, _EXIT);
_EXIT:
	return enModeState;
}


EModeState UKModeTransition_GetModeStateInternal(SModeTransitionMachine *pstModeTransition)
{
	EModeState enModeState = MODE_STATE_TRANSITING;

	enModeState = pstModeTransition->enModeState;

	return enModeState;
}


uem_result UKModeTransition_Clear(SModeTransitionMachine *pstModeTransition)
{
	uem_result result = ERR_UEM_UNKNOWN;

	pstModeTransition->nCurHistoryStartIndex = 0;
	pstModeTransition->nCurModeIndex = 0;
	pstModeTransition->enModeState = MODE_STATE_TRANSITING;
	pstModeTransition->nCurHistoryLen = 0;
	pstModeTransition->nCurrentIteration = 0;
	pstModeTransition->nNextModeIndex = 0;

	result = ERR_UEM_NOERROR;

	return result;
}


EModeState UKModeTransition_UpdateModeStateInternal(SModeTransitionMachine *pstModeTransition, EModeState enModeState, int nIteration)
{
	int nHistoryEnd;
	// MODE_STATE_TRANSITING => MODE_STATE_NORMAL
	if(pstModeTransition->enModeState == MODE_STATE_TRANSITING && enModeState == MODE_STATE_NORMAL)
	{
		if(pstModeTransition->nCurModeIndex != pstModeTransition->nNextModeIndex || nIteration == 0)
		{
			pstModeTransition->nCurModeIndex = pstModeTransition->nNextModeIndex;

			if(pstModeTransition->nCurHistoryLen < MODE_TRANSITION_ARRAY_SIZE)
			{
				pstModeTransition->nCurHistoryLen++;
			}
			else
			{
				pstModeTransition->nCurHistoryStartIndex++;
				if(pstModeTransition->nCurHistoryStartIndex >= MODE_TRANSITION_ARRAY_SIZE)
				{
					pstModeTransition->nCurHistoryStartIndex = 0;
				}
			}

			nHistoryEnd = pstModeTransition->nCurHistoryStartIndex + pstModeTransition->nCurHistoryLen - 1;
			if(nHistoryEnd >= MODE_TRANSITION_ARRAY_SIZE)
			{
				nHistoryEnd -= MODE_TRANSITION_ARRAY_SIZE;
			}

			pstModeTransition->astModeTransition[nHistoryEnd].nIteration = nIteration;
			pstModeTransition->astModeTransition[nHistoryEnd].nModeIndex = pstModeTransition->nCurModeIndex;
		}
	}

	pstModeTransition->enModeState = enModeState;

	return enModeState;
}


uem_result UKModeTransition_SetModeIntegerParameter (IN int nCallerTaskId, IN char *pszTaskName, IN char *pszParamName, IN int nParamVal)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstTask = NULL;
	int nLoop = 0;
	int nLen = 0;
	uem_string_struct strTargetParamName;
	uem_string_struct strParamName;
	/*STask *pstCallerTask = NULL;

	result = UKTask_GetTaskFromTaskId(nCallerTaskId, &pstCallerTask);
	ERRIFGOTO(result, _EXIT);

	result = UKTask_GetTaskByTaskNameAndCallerTask(pstCallerTask, pszTaskName, &pstTask);
	ERRIFGOTO(result, _EXIT);*/

	result = UKTask_GetTaskFromTaskName(pszTaskName, &pstTask);
	ERRIFGOTO(result, _EXIT);

	if(pstTask->pstMTMInfo == NULL)
	{
		IFVARERRASSIGNGOTO(pstTask->pstParentGraph->pstParentTask, NULL, result, ERR_UEM_ILLEGAL_DATA, _EXIT);
		IFVARERRASSIGNGOTO(pstTask->pstParentGraph->pstParentTask->pstMTMInfo, NULL, result, ERR_UEM_ILLEGAL_DATA, _EXIT);

		pstTask = pstTask->pstParentGraph->pstParentTask;
	}

	nLen = pstTask->pstMTMInfo->nNumOfIntVariables;

	result = UCString_New(&strTargetParamName, pszParamName, UEMSTRING_MAX);
	ERRIFGOTO(result, _EXIT);

	for(nLoop = 0 ; nLoop < nLen; nLoop++)
	{
		result = UCString_New(&strParamName, pstTask->pstMTMInfo->astVarIntMap[nLoop].pszVariableName, UEMSTRING_MAX);
		ERRIFGOTO(result, _EXIT);

		if(UCString_IsEqual(&strTargetParamName, &strParamName) == TRUE)
		{
			result = UCThreadMutex_Lock(pstTask->hMutex);
			ERRIFGOTO(result, _EXIT);

			pstTask->pstMTMInfo->astVarIntMap[nLoop].nValue = nParamVal;

			result = UCThreadMutex_Unlock(pstTask->hMutex);
			ERRIFGOTO(result, _EXIT);
			break;
		}
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKModeTransition_UpdateMode (IN int nCallerTaskId, IN char *pszTaskName)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstTask = NULL;
	/*STask *pstCallerTask = NULL;

	result = UKTask_GetTaskFromTaskId(nCallerTaskId, &pstCallerTask);
	ERRIFGOTO(result, _EXIT);

	result = UKTask_GetTaskByTaskNameAndCallerTask(pstCallerTask, pszTaskName, &pstTask);
	ERRIFGOTO(result, _EXIT);*/

	result = UKTask_GetTaskFromTaskName(pszTaskName, &pstTask);
	ERRIFGOTO(result, _EXIT);

	if(pstTask->pstMTMInfo == NULL)
	{
		IFVARERRASSIGNGOTO(pstTask->pstParentGraph->pstParentTask, NULL, result, ERR_UEM_ILLEGAL_DATA, _EXIT);
		IFVARERRASSIGNGOTO(pstTask->pstParentGraph->pstParentTask->pstMTMInfo, NULL, result, ERR_UEM_ILLEGAL_DATA, _EXIT);

		pstTask = pstTask->pstParentGraph->pstParentTask;
	}

	result = UCThreadMutex_Lock(pstTask->hMutex);
	ERRIFGOTO(result, _EXIT);

	pstTask->pstMTMInfo->fnTransition(pstTask->pstMTMInfo);

	result = UCThreadMutex_Unlock(pstTask->hMutex);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

