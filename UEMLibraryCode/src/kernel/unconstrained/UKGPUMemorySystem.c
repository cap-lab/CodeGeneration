/*
 * UKGPUMemorySystem.c
 *
 *  Created on: 2018. 4. 5.
 *      Author: jej
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <UCGPUMemory.h>

uem_result UKGPUMemorySystem_CreateMemory(int nSize, OUT void **ppMemory)
{
	uem_result result = ERR_UEM_UNKNOWN;

	IFVARERRASSIGNGOTO(ppMemory , NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	result = UCGPUMemory_Malloc(ppMemory, nSize);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKGPUMemorySystem_CreateHostAllocMemory(int nSize, OUT void **ppMemory)
{
	uem_result result = ERR_UEM_UNKNOWN;

	IFVARERRASSIGNGOTO(ppMemory , NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	result = UCGPUMemory_HostAlloc(ppMemory, nSize, MEMORY_PROPERTY_PORTABLE);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}



uem_result UKGPUMemorySystem_CopyHostToDeviceMemory(IN void *pDest, IN void *pSrc, int nCopySize)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = UCGPUMemory_Memcpy(pDest, pSrc, nCopySize, MEMCPY_KIND_HOST_TO_DEVICE);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKGPUMemorySystem_CopyDeviceToHostMemory(IN void *pDest, IN void *pSrc, int nCopySize)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = UCGPUMemory_Memcpy(pDest, pSrc, nCopySize, MEMCPY_KIND_DEVICE_TO_HOST);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKGPUMemorySystem_CopyDeviceToDeviceMemory(IN void *pDest, IN void *pSrc, int nCopySize)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = UCGPUMemory_Memcpy(pDest, pSrc, nCopySize, MEMCPY_KIND_DEVICE_TO_DEVICE);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKGPUMemorySystem_DestroyHostAllocMemory(IN OUT void **ppMemory)
{
	uem_result result = ERR_UEM_UNKNOWN;

	IFVARERRASSIGNGOTO(ppMemory , NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	IFVARERRASSIGNGOTO(*ppMemory , NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	// insert your API here
	UCGPUMemory_FreeHost(*ppMemory);

	*ppMemory = NULL;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}



uem_result UKGPUMemorySystem_DestroyMemory(IN OUT void **ppMemory)
{
	uem_result result = ERR_UEM_UNKNOWN;

	IFVARERRASSIGNGOTO(ppMemory , NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	IFVARERRASSIGNGOTO(*ppMemory , NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	// insert your API here

	*ppMemory = NULL;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


