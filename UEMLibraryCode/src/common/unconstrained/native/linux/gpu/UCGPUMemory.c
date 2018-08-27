/*
 * UCCUDABasic.c
 *
 *  Created on: 2018. 4. 2.
 *      Author: DG-SHIN
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <stdlib.h>
#include <string.h>

#include "cuda.h"
#include "cuda_runtime.h"

#include <uem_common.h>

#include <UCGPUMemory.h>

uem_result convertCUDAErrorToUEMError(cudaError_t error)
{
	uem_result result = ERR_UEM_UNKNOWN;

	switch(error){
	// cudaMalloc, cudaHostAlloc, cudaFreeHost,  cudaMemcpy
	case cudaErrorInvalidValue :
		result = ERR_UEM_CUDA_INVALID_VALUE;
		break;
	// cudaMalloc, cudaHostAlloc
	case cudaErrorMemoryAllocation :
		result = ERR_UEM_CUDA_MEMORY_ALLOCATION;
		break;
	// cudaFree
	case cudaErrorInvalidDevicePointer :
		result = ERR_UEM_CUDA_INVALID_DEVICE_POINTER;
		break;
	// cudaFree, cudaFreeHost
	case cudaErrorInitializationError :
		result = ERR_UEM_CUDA_INITIALIZATION;
		break;
	// cudaMemcpy
	case cudaErrorInvalidMemcpyDirection :
		result = ERR_UEM_CUDA_INVALID_MEMCPY_DIRECTION;
		break;
	// cudaSucess
	case cudaSuccess:
		result = ERR_UEM_NOERROR;
		break;
	default :
		// ERR_UEM_UNKNOWN
		break;
	}

	return result;
}


uem_result UCGPUMemory_Malloc(void **ppMemory, int nSize)
{
	cudaError_t error;

	error = cudaMalloc(ppMemory,nSize);

	return convertCUDAErrorToUEMError(error);
}


uem_result UCGPUMemory_HostAlloc(void **ppMemory, int nSize, EMemoryProperty flags)
{
	cudaError_t error;

	error = cudaHostAlloc(ppMemory,nSize,flags);

	return convertCUDAErrorToUEMError(error);
}


uem_result UCGPUMemory_Free(void *pMemory)
{
	cudaError_t error;

	error = cudaFree(pMemory);

	return convertCUDAErrorToUEMError(error);
}


uem_result UCGPUMemory_FreeHost(void *pMemory)
{
	cudaError_t error;

	error = cudaFreeHost(pMemory);

	return convertCUDAErrorToUEMError(error);
}


uem_result UCGPUMemory_Memcpy(void *pDest, const void *pSrc, int nSize, EMemcpyKind flags)
{
	cudaError_t error;

	switch (flags){
	case MEMCPY_KIND_HOST_TO_HOST:
		error = cudaMemcpy(pDest, pSrc, nSize,cudaMemcpyHostToHost);
		break;
	case MEMCPY_KIND_HOST_TO_DEVICE:
		error = cudaMemcpy(pDest, pSrc, nSize,cudaMemcpyHostToDevice);
		break;
	case MEMCPY_KIND_DEVICE_TO_HOST:
		error = cudaMemcpy(pDest, pSrc, nSize,cudaMemcpyDeviceToHost);
		break;
	case MEMCPY_KIND_DEVICE_TO_DEVICE:
		error = cudaMemcpy(pDest, pSrc, nSize,cudaMemcpyDeviceToDevice);
		break;
	case MEMCPY_KIND_DEFAULT:
		error = cudaMemcpy(pDest, pSrc, nSize,cudaMemcpyDefault);
		break;
	default:
		//error;
		break;
	}

	return convertCUDAErrorToUEMError(error);
}
