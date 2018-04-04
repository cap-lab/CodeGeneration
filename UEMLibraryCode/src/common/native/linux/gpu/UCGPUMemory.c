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

#include <UCBasic.h>

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
	default :
		break;
	}

	return result;
}


uem_result UCGPUMemory_Malloc(void **pMemory, int nSize)
{
	cudaError_t error;

	error = cudaMalloc((void**)&pMemory,nSize);

	return convertCUDAErrorToUEMError(error);
}


uem_result UCGPUMemory_HostAlloc(void **pMemory, int nSize, EMemoryProperty flags)
{
	cudaError_t error;

	cudaHostAlloc((void**)&pMemory,nSize,flags);

	return convertCUDAErrorToUEMError(error);

}


uem_result UCGPUMemory_Free(void *pMemory)
{
	cudaError_t error;

	cudaFree(pMemory);

	return convertCUDAErrorToUEMError(error);
}


uem_result UCGPUMemory_FreeHost(void *pMemory)
{
	cudaError_t error;

	cudaFreeHost(pMemory);

	return convertCUDAErrorToUEMError(error);
}


uem_result UCGPUMemory_Memcpy(void *pDest, const void *pSrc, int nSize, EMemcpyKind flags)
{
	cudaError_t error;

	switch (flags){
	case MEMCPY_KIND_HOST_TO_HOST:
		cudaMemcpy(pDest, pSrc, nSize,cudaMemcpyHostToHost);
		break;
	case MEMCPY_KIND_HOST_TO_DEVICE:
		cudaMemcpy(pDest, pSrc, nSize,cudaMemcpyHostToDevice);
		break;
	case MEMCPY_KIND_DEVICE_TO_HOST:
		cudaMemcpy(pDest, pSrc, nSize,cudaMemcpyDeviceToHost);
		break;
	case MEMCPY_KIND_DEVICE_TO_DEVICE:
		cudaMemcpy(pDest, pSrc, nSize,cudaMemcpyDeviceToDevice);
		break;
	case MEMCPY_KIND_DEFAULT:
		cudaMemcpy(pDest, pSrc, nSize,cudaMemcpyDefault);
		break;
	default:
		//error;
		break;
	}

	return convertCUDAErrorToUEMError(error);
}
