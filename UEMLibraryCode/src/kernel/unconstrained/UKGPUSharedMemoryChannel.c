/*
 * UKGPUSharedMemoryChannel.c
 *
 *  Created on: 2018. 4. 2.
 *      Author: DG-SHIN
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include "../../common/include/gpu/UCGPUMemory.h"
#include <UCBasic.h>
#include <UCThreadMutex.h>

#include <uem_data.h>

#include <UKTask.h>

uem_result UKGPUSharedMemoryChannel_Clear(SChannel *pstChannel)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SGPUSharedMemoryChannel *pstGPUSharedMemroyChannel = NULL;

	pstGPUSharedMemroyChannel = (SGPUSharedMemoryChannel *) pstChannel->pChannelStruct;

	// information clear
	// pDataStart => clear (pBuffer)
	// nDataLen => clear (0)
	pstGPUSharedMemroyChannel->pDataStart = pstGPUSharedMemroyChannel->pBuffer;
	pstGPUSharedMemroyChannel->pDataEnd = pstGPUSharedMemroyChannel->pBuffer;
	pstGPUSharedMemroyChannel->nDataLen = 0;
	pstGPUSharedMemroyChannel->nReadReferenceCount = 0;
	pstGPUSharedMemroyChannel->nWriteReferenceCount = 0;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKGPUSharedMemoryChannel_Initialize(SChannel *pstChannel)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SGPUSharedMemoryChannel *pstGPUSharedMemroyChannel = NULL;

	pstGPUSharedMemroyChannel = (SGPUSharedMemoryChannel *) pstChannel->pChannelStruct;

	// initialize buffer
	// If not set, initialize those things, // pBuffer => is NULL => alloc
	if(pstGPUSharedMemroyChannel->pBuffer == NULL)
	{
		switch(pstChannel->enType){
		case COMMUNICATION_TYPE_CPU_GPU:
			pstGPUSharedMemroyChannel->pBuffer = UC_malloc(pstChannel->nBufSize);
			break;
		case COMMUNICATION_TYPE_GPU_CPU:
			pstGPUSharedMemroyChannel->pBuffer = UC_malloc(pstChannel->nBufSize);
			break;
		case COMMUNICATION_TYPE_GPU_GPU:
			pstGPUSharedMemroyChannel->pBuffer = UC_cudaMalloc(pstChannel->nBufSize);
			break;
		case COMMUNICATION_TYPE_GPU_GPU_DIFFERENT:
			pstGPUSharedMemroyChannel->pBuffer = UC_cudaHostAlloc(pstChannel->nBufSize, 1); //flag is 'cudaHostAllocPortable'
			break;
		default :
			break;
		}
		ERRMEMGOTO(pstGPUSharedMemroyChannel->pBuffer, result, _EXIT);
	}

	// hMutex => initialize/create
	result = UCThreadMutex_Create(&(pstGPUSharedMemroyChannel->hMutex));
	ERRIFGOTO(result, _EXIT);

	// hEvent => initialize/create
	result = UCThreadEvent_Create(&(pstGPUSharedMemroyChannel->hReadEvent));
	ERRIFGOTO(result, _EXIT);
	result = UCThreadEvent_Create(&(pstGPUSharedMemroyChannel->hWriteEvent));
	ERRIFGOTO(result, _EXIT);

	result = UKGPUSharedMemoryChannel_Clear(pstChannel);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result copyAndMovePointerFromRoundedQueue(unsigned char *pDest, SChannel *pstChannel, int nDataToRead)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nSegmentLen = 0;
	int nRemainderLen = 0;
	SGPUSharedMemoryChannel *pstGPUSharedMemroyChannel = NULL;

	pstGPUSharedMemroyChannel = (SGPUSharedMemoryChannel *) pstChannel->pChannelStruct;


	// data is divided into two segments because the start point of the chunk is close to the end of buffer range
	if(pstGPUSharedMemroyChannel->pDataStart + nDataToRead > pstGPUSharedMemroyChannel->pBuffer + pstChannel->nBufSize)
	{
		nSegmentLen = pstGPUSharedMemroyChannel->pBuffer + pstChannel->nBufSize - pstGPUSharedMemroyChannel->pDataStart;
		nRemainderLen = nDataToRead - nSegmentLen;

		switch (pstChannel->enType) {
		case COMMUNICATION_TYPE_CPU_GPU:
		case COMMUNICATION_TYPE_GPU_CPU:
			UC_memcpy(pDest, pstGPUSharedMemroyChannel->pDataStart, nSegmentLen);
			UC_memcpy(pDest + nSegmentLen, pstGPUSharedMemroyChannel->pBuffer, nRemainderLen);
			break;
		case COMMUNICATION_TYPE_GPU_GPU:
			//The flag for cudamemcpy is DeviceToDevice.
			UC_cudaMemcpy(pDest, pstGPUSharedMemroyChannel->pDataStart, nSegmentLen, 3);
			UC_cudaMemcpy(pDest + nSegmentLen, pstGPUSharedMemroyChannel->pDataStart, nRemainderLen, 3);
			break;
		case COMMUNICATION_TYPE_GPU_GPU_DIFFERENT:
			//The flag for cudamemcpy is HostToDevice.
			UC_cudaMemcpy(pDest, pstGPUSharedMemroyChannel->pDataStart, nSegmentLen, 1);
			UC_cudaMemcpy(pDest + nSegmentLen, pstGPUSharedMemroyChannel->pDataStart, nRemainderLen, 1);
			break;
		default:
			break;
		}

		pstGPUSharedMemroyChannel->pDataStart = pstGPUSharedMemroyChannel->pBuffer + nRemainderLen;
	}
	else
	{
		switch(pstChannel->enType){
		case COMMUNICATION_TYPE_CPU_GPU:
		case COMMUNICATION_TYPE_GPU_CPU:
			UC_memcpy(pDest, pstGPUSharedMemroyChannel->pDataStart, nDataToRead);
			break;
		case COMMUNICATION_TYPE_GPU_GPU:
			//The flag for cudamemcpy is DeviceToDevice.
			UC_cudaMemcpy(pDest, pstGPUSharedMemroyChannel->pDataStart, nDataToRead,3);
			break;
		case COMMUNICATION_TYPE_GPU_GPU_DIFFERENT:
			//The flag for cudamemcpy is HostToDevice.
			UC_cudaMemcpy(pDest, pstGPUSharedMemroyChannel->pDataStart, nDataToRead,1);
			break;
		default :
			break;
		}

		if(pstGPUSharedMemroyChannel->pDataStart + nDataToRead == pstGPUSharedMemroyChannel->pBuffer + pstChannel->nBufSize)
		{
			pstGPUSharedMemroyChannel->pDataStart = pstGPUSharedMemroyChannel->pBuffer;
		}
		else
		{
			pstGPUSharedMemroyChannel->pDataStart += nDataToRead;
		}
	}

	pstGPUSharedMemroyChannel->nDataLen -= nDataToRead;

	result = ERR_UEM_NOERROR;

	return result;
}


static uem_result readFromGeneralQueue(SChannel *pstChannel, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SGPUSharedMemoryChannel *pstGPUSharedMemroyChannel = NULL;

	pstGPUSharedMemroyChannel = (SGPUSharedMemoryChannel *) pstChannel->pChannelStruct;

	result = UCThreadMutex_Lock(pstGPUSharedMemroyChannel->hMutex);
	ERRIFGOTO(result, _EXIT);

	pstGPUSharedMemroyChannel->nReadReferenceCount++;

	// TODO: Error check out exit logic needed
	while(pstGPUSharedMemroyChannel->nDataLen < nDataToRead)
	{
		if(pstGPUSharedMemroyChannel->bReadExit == TRUE)
		{
			UEMASSIGNGOTO(result, ERR_UEM_SUSPEND, _EXIT_LOCK);
		}

		result = UCThreadMutex_Unlock(pstGPUSharedMemroyChannel->hMutex);
		ERRIFGOTO(result, _EXIT);

		result = UCThreadEvent_WaitEvent(pstGPUSharedMemroyChannel->hReadEvent);
		ERRIFGOTO(result, _EXIT);

		result = UCThreadMutex_Lock(pstGPUSharedMemroyChannel->hMutex);
		ERRIFGOTO(result, _EXIT);
	}

	result = copyAndMovePointerFromRoundedQueue(pBuffer, pstChannel, nDataToRead);
	ERRIFGOTO(result, _EXIT_LOCK);

	*pnDataRead = nDataToRead;

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
pstGPUSharedMemroyChannel->nReadReferenceCount--;

	if(pstGPUSharedMemroyChannel->nWriteReferenceCount > 0)
	{
		UCThreadEvent_SetEvent(pstGPUSharedMemroyChannel->hWriteEvent);
	}
	UCThreadMutex_Unlock(pstGPUSharedMemroyChannel->hMutex);
_EXIT:
	return result;
}

uem_result UKGPUSharedMemoryChannel_ReadFromQueue(SChannel *pstChannel, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = readFromGeneralQueue(pstChannel, pBuffer, nDataToRead, nChunkIndex, pnDataRead);
	ERRIFGOTO(result, _EXIT);

	// to preserve, ERR_UEM_SUSPEND, do not set ERR_UEM_NOERROR here
_EXIT:
	return result;
}


static uem_result copyAndMovePointerToRoundedQueue(SChannel *pstChannel, unsigned char *pSrc, int nDataToWrite, int nChunkIndex)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nSegmentLen = 0;
	int nRemainderLen = 0;
	SGPUSharedMemoryChannel *pstGPUSharedMemroyChannel = NULL;

	pstGPUSharedMemroyChannel = (SGPUSharedMemoryChannel *) pstChannel->pChannelStruct;

	// Chunk is divided into two segments because the start point of the chunk is close to the end of buffer range
	if(pstGPUSharedMemroyChannel->pDataEnd + nDataToWrite > pstGPUSharedMemroyChannel->pBuffer + pstChannel->nBufSize)
	{
		nSegmentLen = pstGPUSharedMemroyChannel->pBuffer + pstChannel->nBufSize - pstGPUSharedMemroyChannel->pDataEnd;
		nRemainderLen = nDataToWrite - nSegmentLen;


		switch(pstChannel->enType){
		case COMMUNICATION_TYPE_CPU_GPU:
		case COMMUNICATION_TYPE_GPU_CPU:
			UC_memcpy(pstGPUSharedMemroyChannel->pDataEnd, pSrc, nSegmentLen);
			UC_memcpy(pstGPUSharedMemroyChannel->pBuffer, pSrc + nSegmentLen, nRemainderLen);
			break;
		case COMMUNICATION_TYPE_GPU_GPU:
			//The flag of cudamemcpy is DeviceToDevice.
			UC_cudaMemcpy(pstGPUSharedMemroyChannel->pDataEnd, pSrc, nSegmentLen, 3);
			UC_cudaMemcpy(pstGPUSharedMemroyChannel->pBuffer, pSrc + nSegmentLen, nRemainderLen, 3);
			break;
		case COMMUNICATION_TYPE_GPU_GPU_DIFFERENT:
			//The flag of cudamemcpy is DeviceToHost.
			break;
		default :
			break;
		}

		pstGPUSharedMemroyChannel->pDataEnd = pstGPUSharedMemroyChannel->pBuffer + nDataToWrite - nSegmentLen;
	}
	else
	{
		switch(pstChannel->enType){
		case COMMUNICATION_TYPE_CPU_GPU:
		case COMMUNICATION_TYPE_GPU_CPU:
			UC_memcpy(pstGPUSharedMemroyChannel->pDataEnd, pSrc, nDataToWrite);
			break;
		case COMMUNICATION_TYPE_GPU_GPU:
			//The flag for cudamemcpy is DeviceToDevice.
			UC_cudaMemcpy(pstGPUSharedMemroyChannel->pDataEnd, pSrc, nDataToWrite, 3);
			break;
		case COMMUNICATION_TYPE_GPU_GPU_DIFFERENT:
			//The flag for cudamemcpy is DeviceToHost.
			UC_cudaMemcpy(pstGPUSharedMemroyChannel->pDataEnd, pSrc, nDataToWrite, 2);
			break;
		default :
			break;
		}

		if(pstGPUSharedMemroyChannel->pDataEnd == pstGPUSharedMemroyChannel->pBuffer + pstChannel->nBufSize)
		{
			pstGPUSharedMemroyChannel->pDataEnd = pstGPUSharedMemroyChannel->pBuffer;
		}
		else
		{
			pstGPUSharedMemroyChannel->pDataEnd += nDataToWrite;
		}
	}

	pstGPUSharedMemroyChannel->nDataLen += nDataToWrite;

	result = ERR_UEM_NOERROR;

	return result;
}


static uem_result writeToGeneralQueue(SChannel *pstChannel, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nCurrentSampleRateIndex = 0;
	int nExpectedConsumeSize = 0;
	SGPUSharedMemoryChannel *pstGPUSharedMemroyChannel = NULL;

	pstGPUSharedMemroyChannel = (SGPUSharedMemoryChannel *) pstChannel->pChannelStruct;

	result = UCThreadMutex_Lock(pstGPUSharedMemroyChannel->hMutex);
	ERRIFGOTO(result, _EXIT);

	pstGPUSharedMemroyChannel->nWriteReferenceCount++;

	// TODO: Error check out exit logic needed
	while(pstGPUSharedMemroyChannel->nDataLen + nDataToWrite > pstChannel->nBufSize)
	{
		if(pstGPUSharedMemroyChannel->bWriteExit == TRUE)
		{
			UEMASSIGNGOTO(result, ERR_UEM_SUSPEND, _EXIT_LOCK);
		}

		result = UCThreadMutex_Unlock(pstGPUSharedMemroyChannel->hMutex);
		ERRIFGOTO(result, _EXIT);

		result = UCThreadEvent_WaitEvent(pstGPUSharedMemroyChannel->hWriteEvent);
		ERRIFGOTO(result, _EXIT);

		result = UCThreadMutex_Lock(pstGPUSharedMemroyChannel->hMutex);
		ERRIFGOTO(result, _EXIT);
	}

	result = copyAndMovePointerToRoundedQueue(pstChannel, pBuffer, nDataToWrite, nChunkIndex);
	ERRIFGOTO(result, _EXIT_LOCK);

	*pnDataWritten = nDataToWrite;

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
pstGPUSharedMemroyChannel->nWriteReferenceCount--;

	if(pstGPUSharedMemroyChannel->nReadReferenceCount > 0)
	{
		UCThreadEvent_SetEvent(pstGPUSharedMemroyChannel->hReadEvent);
	}
	UCThreadMutex_Unlock(pstGPUSharedMemroyChannel->hMutex);
_EXIT:
	return result;
}


uem_result UKGPUSharedMemoryChannel_WriteToQueue (SChannel *pstChannel, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = writeToGeneralQueue(pstChannel, pBuffer, nDataToWrite, nChunkIndex, pnDataWritten);
	ERRIFGOTO(result, _EXIT);

	// to preserve, ERR_UEM_SUSPEND, do not set ERR_UEM_NOERROR here
_EXIT:
	return result;
}


uem_result UKGPUSharedMemoryChannel_GetNumOfAvailableData (SChannel *pstChannel, IN int nChunkIndex, OUT int *pnDataNum)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SGPUSharedMemoryChannel *pstGPUSharedMemroyChannel = NULL;

	pstGPUSharedMemroyChannel = (SGPUSharedMemoryChannel *) pstChannel->pChannelStruct;

	result = UCThreadMutex_Lock(pstGPUSharedMemroyChannel->hMutex);
	ERRIFGOTO(result, _EXIT);

	*pnDataNum = pstGPUSharedMemroyChannel->nDataLen;

	result = ERR_UEM_NOERROR;

	UCThreadMutex_Unlock(pstGPUSharedMemroyChannel->hMutex);
_EXIT:
	return result;
}


uem_result UKGPUSharedMemoryChannel_Finalize(SChannel *pstChannel)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SGPUSharedMemoryChannel *pstGPUSharedMemroyChannel = NULL;

	pstGPUSharedMemroyChannel = (SGPUSharedMemoryChannel *) pstChannel->pChannelStruct;

	// ignore error
	UCThreadMutex_Destroy(&(pstGPUSharedMemroyChannel->hMutex));

	// ignore error
	UCThreadEvent_Destroy(&(pstGPUSharedMemroyChannel->hReadEvent));
	UCThreadEvent_Destroy(&(pstGPUSharedMemroyChannel->hWriteEvent));

	result = ERR_UEM_NOERROR;

	return result;
}
