/*
 * UKProcessor.c
 *
 *  Created on: 2018. 1. 1.
 *      Author: jej
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_data.h>

uem_result UKProcessor_IsCPUByProcessorId(int nProcessorId, uem_bool *pbIsCPU)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;
	uem_bool bIsCPU = FALSE;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(pbIsCPU, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	if(nProcessorId < 0)
	{
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
#endif
	for(nLoop = 0 ; nLoop < g_nProcessorInfoNum ; nLoop++)
	{
		if(g_astProcessorInfo[nLoop].nProcessorId == nProcessorId)
		{
			bIsCPU =  g_astProcessorInfo[nLoop].bIsCPU;
			break;
		}
	}

	*pbIsCPU = bIsCPU;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKProcessor_GetGPUProcessorId(IN int nProcessorId, OUT int *pnGPUProcessorId)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;
	int nGPUProcessorId = 0;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(pnGPUProcessorId, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	if(nProcessorId < 0)
	{
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
#endif
	for(nLoop = 0 ; nLoop < g_nProcessorInfoNum ; nLoop++)
	{
		if(nLoop == nProcessorId)
		{
			break;
		}
		if(g_astProcessorInfo[nLoop].bIsCPU == FALSE)
		{
			nGPUProcessorId++;
		}
	}

	*pnGPUProcessorId = nGPUProcessorId;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}
