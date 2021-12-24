/*
 * UKHostMemorySystem.c
 *
 *  Created on: 2018. 4. 5.
 *      Author: jej
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <UCBasic.h>
#include <UCAlloc.h>
#include <UCThread.h>

uem_result UKHostSystem_CreateMemory(int nSize, int nProcessorId, OUT void **ppMemory)
{
	uem_result result = ERR_UEM_UNKNOWN;

	IFVARERRASSIGNGOTO(ppMemory , NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	*ppMemory = UCAlloc_malloc(nSize);
	ERRMEMGOTO(*ppMemory, result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKHostSystem_CopyToMemory(IN void *pMemory, IN void *pSource, int nCopySize)
{
	UC_memcpy(pMemory, pSource, nCopySize);

	return ERR_UEM_NOERROR;
}


uem_result UKHostSystem_CopyInMemory(IN void *pInMemoryDst, IN void *pInMemorySrc, int nCopySize)
{
	UC_memcpy(pInMemoryDst, pInMemorySrc, nCopySize);

	return ERR_UEM_NOERROR;
}


uem_result UKHostSystem_CopyFromMemory(IN void *pDestination, IN void *pMemory, int nCopySize)
{
	UC_memcpy(pDestination, pMemory, nCopySize);

	return ERR_UEM_NOERROR;
}


uem_result UKHostSystem_DestroyMemory(IN OUT void **ppMemory)
{
	uem_result result = ERR_UEM_UNKNOWN;

	IFVARERRASSIGNGOTO(ppMemory , NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	IFVARERRASSIGNGOTO(*ppMemory , NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	UCAlloc_free(*ppMemory);

	*ppMemory = NULL;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKHostSystem_MapCPU(HThread hThread, int nProcessorId, int nLocalId)
{
	uem_result result = ERR_UEM_UNKNOWN;

	if (nLocalId != MAPPING_NOT_SPECIFIED) {
		result = UCThread_SetMappedCPU(hThread, nLocalId);
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKHostSystem_MapPriority(HThread hThread, int nScheduler, int nPriority)
{
	uem_result result = ERR_UEM_UNKNOWN;

	if(nPriority != PRIORITY_NOT_SPECIFIED) {
		result = UCThread_SetPriority(hThread, nScheduler, nPriority);
		ERRIFGOTO(result, _EXIT);	
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}
