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

void *UCGPUMemory_Malloc(int nSize)
{
	void *pMemory = NULL;

	cudaMalloc((void**)&pMemory,nSize);

	return pMemory;
}


void *UCGPUMemory_HostAlloc(int nSize, unsigned int flags)
{
	void *pMemory = NULL;

	cudaHostAlloc((void**)&pMemory,nSize,flags);

	return pMemory;
}

void UCGPUMemory_Free(void *pMem)
{
	cudaFree(pMem);
}

void UCGPUMemory_FreeHost(void *pMem)
{
	cudaFreeHost(pMem);
}


void *UCGPUMemory_Memcpy(void *pDest, const void *pSrc, int nSize, unsigned int flags)
{
	void *pMemory = NULL;

	switch (flags){
	case 0:
		pMemory = cudaMemcpy(pDest, pSrc, nSize,cudaMemcpyHostToHost);
		break;
	case 1:
		pMemory = cudaMemcpy(pDest, pSrc, nSize,cudaMemcpyHostToDevice);
		break;
	case 2:
		pMemory = cudaMemcpy(pDest, pSrc, nSize,cudaMemcpyDeviceToHost);
		break;
	case 3:
		pMemory = cudaMemcpy(pDest, pSrc, nSize,cudaMemcpyDeviceToDevice);
		break;
	case 4:
		pMemory = cudaMemcpy(pDest, pSrc, nSize,cudaMemcpyDefault);
		break;
	default:
		//error;
		break;
	}

	return pMemory;
}
