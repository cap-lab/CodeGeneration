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

uem_result UKTask_GetCurrentModeName (IN char *pszTaskName, OUT char **ppszModeName)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstTask = NULL;
	int nCurModeIndex = 0;

	result = UKTask_GetTaskFromTaskName(pszTaskName, &pstTask);
	ERRIFGOTO(result, _EXIT);

	IFVARERRASSIGNGOTO(pstTask->pstMTMInfo, NULL, result, ERR_UEM_ILLEGAL_DATA, _EXIT);

	nCurModeIndex = pstTask->pstMTMInfo->nCurModeIndex;

	*ppszModeName = pstTask->pstMTMInfo->astModeMap[nCurModeIndex].pszModeName;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKTask_SetModeIntegerParameter (IN char *pszTaskName, IN char *pszParamName, IN int nParamVal)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstTask = NULL;
	int nLoop = 0;
	int nLen = 0;
	uem_string_struct strTargetParamName;
	uem_string_struct strParamName;

	result = UKTask_GetTaskFromTaskName(pszTaskName, &pstTask);
	ERRIFGOTO(result, _EXIT);

	IFVARERRASSIGNGOTO(pstTask->pstMTMInfo, NULL, result, ERR_UEM_ILLEGAL_DATA, _EXIT);

	nLen = ARRAYLEN(pstTask->pstMTMInfo->astVarIntMap);

	result = UCString_New(&strTargetParamName, pszParamName, UEMSTRING_MAX);
	ERRIFGOTO(result, _EXIT);

	for(nLoop = 0 ; nLoop < nLen; nLoop++)
	{
		result = UCString_New(&strParamName, pstTask->pstMTMInfo->astVarIntMap[nLoop].pszVariableName, UEMSTRING_MAX);
		ERRIFGOTO(result, _EXIT);

		if(UCString_IsEqual(&strTargetParamName, &strParamName) == TRUE)
		{
			pstTask->pstMTMInfo->astVarIntMap[nLoop].nValue = nParamVal;
			break;
		}
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKTask_UpdateMode (IN char *pszTaskName)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstTask = NULL;

	result = UKTask_GetTaskFromTaskName(pszTaskName, &pstTask);
	ERRIFGOTO(result, _EXIT);

	IFVARERRASSIGNGOTO(pstTask->pstMTMInfo, NULL, result, ERR_UEM_ILLEGAL_DATA, _EXIT);

	pstTask->pstMTMInfo->fnTransition();

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


