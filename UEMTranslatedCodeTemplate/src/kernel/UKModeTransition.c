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

uem_result UKModeTransition_GetCurrentModeName (IN char *pszTaskName, OUT char **ppszModeName)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstTask = NULL;
	int nCurModeIndex = 0;

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

	nCurModeIndex = pstTask->pstMTMInfo->nCurModeIndex;

	result = UCThreadMutex_Unlock(pstTask->hMutex);
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


uem_result UKModeTransition_SetModeIntegerParameter (IN char *pszTaskName, IN char *pszParamName, IN int nParamVal)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstTask = NULL;
	int nLoop = 0;
	int nLen = 0;
	uem_string_struct strTargetParamName;
	uem_string_struct strParamName;

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


uem_result UKModeTransition_UpdateMode (IN char *pszTaskName)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstTask = NULL;

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


