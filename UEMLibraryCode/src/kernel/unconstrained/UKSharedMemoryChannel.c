/*
 * UKSharedMemoryChannel.c
 *
 *  Created on: 2017. 11. 9.
 *      Author: jej
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <UCBasic.h>
#include <UCThreadMutex.h>

#include <uem_data.h>

#include <UKTask.h>

static uem_bool isLocatedInsideLoopTask(SPort *pstPort)
{
	uem_bool bInLoopTask = FALSE;
	SPort *pstCurrentPort = NULL;
	STask *pstTask = NULL;
	uem_result result = ERR_UEM_UNKNOWN;

	pstCurrentPort = pstPort;
	while(pstCurrentPort != NULL)
	{
		result = UKTask_GetTaskFromTaskId(pstCurrentPort->nTaskId, &pstTask);
		ERRIFGOTO(result, _EXIT);

		while(pstTask != NULL)
		{
			if(pstTask->pstLoopInfo != NULL)
			{
				bInLoopTask = TRUE;
				break;
			}
			pstTask = pstTask->pstParentGraph->pstParentTask;
		}

		pstCurrentPort = pstCurrentPort->pstSubGraphPort;
	}

_EXIT:
	return bInLoopTask;
}

static uem_result setChunkNumAndLen(SPort *pstPort, SChunkInfo *pstChunkInfo)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nCurrentSampleRateIndex = 0;
	SPort *pstMostInnerPort = NULL;
	int nOuterMostSampleRate = 0;
	STask *pstCurTask = NULL;
	int nLoop = 0;

	nCurrentSampleRateIndex = pstPort->nCurrentSampleRateIndex;

	nOuterMostSampleRate = pstPort->astSampleRates[nCurrentSampleRateIndex].nSampleRate;

	pstMostInnerPort = pstPort;
	while(pstMostInnerPort->pstSubGraphPort != NULL)
	{
		pstMostInnerPort = pstMostInnerPort->pstSubGraphPort;
	}

	if(pstPort != pstMostInnerPort)
	{
		nCurrentSampleRateIndex = pstMostInnerPort->nCurrentSampleRateIndex;

		pstChunkInfo->nChunkNum = nOuterMostSampleRate / pstMostInnerPort->astSampleRates[nCurrentSampleRateIndex].nSampleRate;
		pstChunkInfo->nChunkLen = pstMostInnerPort->astSampleRates[nCurrentSampleRateIndex].nSampleRate * pstMostInnerPort->nSampleSize;
	}
	else
	{
		result = UKTask_GetTaskFromTaskId(pstPort->nTaskId, &pstCurTask);
		ERRIFGOTO(result, _EXIT);

		if(pstCurTask->pstLoopInfo == NULL) // general task
		{
			pstChunkInfo->nChunkNum = 1;
			pstChunkInfo->nChunkLen = nOuterMostSampleRate * pstPort->nSampleSize;

		}
		else if(pstCurTask->pstLoopInfo->enType == LOOP_TYPE_DATA &&
			pstPort->astSampleRates[nCurrentSampleRateIndex].nMaxAvailableDataNum == 1)
		{
			pstChunkInfo->nChunkNum = nOuterMostSampleRate / (nOuterMostSampleRate / pstCurTask->pstLoopInfo->nLoopCount);
			pstChunkInfo->nChunkLen = (nOuterMostSampleRate * pstPort->nSampleSize) / pstChunkInfo->nChunkNum;
		}
		else // broadcasting or convergent
		{
			pstChunkInfo->nChunkNum = 1;
			pstChunkInfo->nChunkLen = nOuterMostSampleRate * pstPort->nSampleSize;
		}
	}

	// clear chunk information
	for(nLoop = 0 ; nLoop < pstChunkInfo->nChunkNum ; nLoop++)
	{
		pstChunkInfo->astChunk[nLoop].pDataStart = NULL;
		pstChunkInfo->astChunk[nLoop].pDataEnd = NULL;
		pstChunkInfo->astChunk[nLoop].nChunkDataLen = 0;
		pstChunkInfo->astChunk[nLoop].pChunkStart = NULL;
		pstChunkInfo->astChunk[nLoop].nAvailableDataNum = 0;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKSharedMemoryChannel_Clear(SChannel *pstChannel)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;
	SSharedMemoryChannel *pstSharedMemroyChannel = NULL;

	pstSharedMemroyChannel = (SSharedMemoryChannel *) pstChannel->pChannelStruct;

	// information clear
	// pDataStart => clear (pBuffer)
	// nDataLen => clear (0)
	pstSharedMemroyChannel->pDataStart = pstSharedMemroyChannel->pBuffer;
	pstSharedMemroyChannel->pDataEnd = pstSharedMemroyChannel->pBuffer;
	pstSharedMemroyChannel->nDataLen = 0;
	pstSharedMemroyChannel->nReadReferenceCount = 0;
	pstSharedMemroyChannel->nWriteReferenceCount = 0;

	// the chunk num and chunk length is dependent to sample rate of mode transition
	result = setChunkNumAndLen(&(pstChannel->stInputPort), &(pstSharedMemroyChannel->stInputPortChunk));
	ERRIFGOTO(result, _EXIT);

	result = setChunkNumAndLen(&(pstChannel->stOutputPort), &(pstSharedMemroyChannel->stOutputPortChunk));
	ERRIFGOTO(result, _EXIT);

	pstSharedMemroyChannel->nWrittenOutputChunkNum = CHUNK_NUM_NOT_INITIALIZED;

	pstSharedMemroyChannel->pstAvailableInputChunkHead = NULL;
	pstSharedMemroyChannel->pstAvailableInputChunkTail = NULL;

	for(nLoop = 0 ; nLoop < pstSharedMemroyChannel->nMaxChunkNum ; nLoop++)
	{
		pstSharedMemroyChannel->astAvailableInputChunkList[nLoop].pstPrev = NULL;
		pstSharedMemroyChannel->astAvailableInputChunkList[nLoop].pstNext = NULL;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKSharedMemoryChannel_Initialize(SChannel *pstChannel)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSharedMemoryChannel *pstSharedMemroyChannel = NULL;

	pstSharedMemroyChannel = (SSharedMemoryChannel *) pstChannel->pChannelStruct;

	// initialize buffer
	// If not set, initialize those things, // pBuffer => is NULL => alloc
	if(pstSharedMemroyChannel->pBuffer == NULL)
	{
		pstSharedMemroyChannel->pBuffer = UC_malloc(pstChannel->nBufSize);
		ERRMEMGOTO(pstSharedMemroyChannel->pBuffer, result, _EXIT);
	}

	// hMutex => initialize/create
	result = UCThreadMutex_Create(&(pstSharedMemroyChannel->hMutex));
	ERRIFGOTO(result, _EXIT);

	// hEvent => initialize/create
	result = UCThreadEvent_Create(&(pstSharedMemroyChannel->hReadEvent));
	ERRIFGOTO(result, _EXIT);
	result = UCThreadEvent_Create(&(pstSharedMemroyChannel->hWriteEvent));
	ERRIFGOTO(result, _EXIT);

	result = UKSharedMemoryChannel_Clear(pstChannel);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static uem_result removeChunkFromAvailableChunkList(SChannel *pstChannel, int nChunkIndex)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SAvailableChunk *pstAvailableChunk = NULL;
	SAvailableChunk *pstPreviousChunk = NULL;
	SAvailableChunk *pstNextChunk = NULL;
	SSharedMemoryChannel *pstSharedMemroyChannel = NULL;

	pstSharedMemroyChannel = (SSharedMemoryChannel *) pstChannel->pChannelStruct;

	pstAvailableChunk = pstSharedMemroyChannel->pstAvailableInputChunkHead;

	while(pstAvailableChunk != NULL)
	{
		if(pstAvailableChunk->nChunkIndex == nChunkIndex)
		{
			pstPreviousChunk = pstAvailableChunk->pstPrev;
			pstNextChunk = pstAvailableChunk->pstNext;

			if(pstPreviousChunk != NULL)
			{
				pstPreviousChunk->pstNext = pstNextChunk;
			}
			else // pstPreviousChunk == NULL => pstAvailableChunk is a head chunk
			{
				pstSharedMemroyChannel->pstAvailableInputChunkHead = pstAvailableChunk->pstNext;
			}

			if(pstNextChunk != NULL)
			{
				pstNextChunk->pstPrev = pstPreviousChunk;
			}
			else // pstNextChunk == NULL => pstAvailableChunk is a tail chunk
			{
				pstSharedMemroyChannel->pstAvailableInputChunkTail = pstAvailableChunk->pstPrev;
			}

			pstAvailableChunk->pstPrev = NULL;
			pstAvailableChunk->pstNext = NULL;
			break;
		}
		pstAvailableChunk = pstAvailableChunk->pstNext;
	}

	if(pstAvailableChunk == NULL)
	{
		ERRASSIGNGOTO(result, ERR_UEM_NO_DATA, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static uem_result copyAndMovePointerFromRoundedQueue(unsigned char *pDest, SChannel *pstChannel, int nDataToRead)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nSegmentLen = 0;
	int nRemainderLen = 0;
	SSharedMemoryChannel *pstSharedMemroyChannel = NULL;

	pstSharedMemroyChannel = (SSharedMemoryChannel *) pstChannel->pChannelStruct;

	// data is divided into two segments because the start point of the chunk is close to the end of buffer range
	if(pstSharedMemroyChannel->pDataStart + nDataToRead > pstSharedMemroyChannel->pBuffer + pstChannel->nBufSize)
	{
		nSegmentLen = pstSharedMemroyChannel->pBuffer + pstChannel->nBufSize - pstSharedMemroyChannel->pDataStart;
		nRemainderLen = nDataToRead - nSegmentLen;

		UC_memcpy(pDest, pstSharedMemroyChannel->pDataStart, nSegmentLen);
		UC_memcpy(pDest + nSegmentLen, pstSharedMemroyChannel->pBuffer, nRemainderLen);

		pstSharedMemroyChannel->pDataStart = pstSharedMemroyChannel->pBuffer + nRemainderLen;
	}
	else
	{
		UC_memcpy(pDest, pstSharedMemroyChannel->pDataStart, nDataToRead);

		if(pstSharedMemroyChannel->pDataStart + nDataToRead == pstSharedMemroyChannel->pBuffer + pstChannel->nBufSize)
		{
			pstSharedMemroyChannel->pDataStart = pstSharedMemroyChannel->pBuffer;
		}
		else
		{
			pstSharedMemroyChannel->pDataStart += nDataToRead;
		}
	}

	pstSharedMemroyChannel->nDataLen -= nDataToRead;

	result = ERR_UEM_NOERROR;

	return result;
}

static uem_result copyToRoundedChunk(SChannel *pstChannel, unsigned char *pSrc, int nDataToWrite, int nChunkIndex)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SChunk *pstDestChunk = NULL;
	int nSegmentLen = 0;
	int nRemainderLen = 0;
	SSharedMemoryChannel *pstSharedMemroyChannel = NULL;

	pstSharedMemroyChannel = (SSharedMemoryChannel *) pstChannel->pChannelStruct;

	pstDestChunk = &(pstSharedMemroyChannel->stOutputPortChunk.astChunk[nChunkIndex]);

	// Chunk is divided into two segments because the start point of the chunk is close to the end of buffer range
	if(pstDestChunk->pDataEnd + nDataToWrite > pstSharedMemroyChannel->pBuffer + pstChannel->nBufSize)
	{
		nSegmentLen = pstSharedMemroyChannel->pBuffer + pstChannel->nBufSize - pstDestChunk->pDataEnd;
		nRemainderLen = nDataToWrite - nSegmentLen;

		UC_memcpy(pstDestChunk->pDataEnd, pSrc, nSegmentLen);
		UC_memcpy(pstSharedMemroyChannel->pBuffer, pSrc + nSegmentLen, nRemainderLen);

		pstDestChunk->pDataEnd = pstSharedMemroyChannel->pBuffer + nDataToWrite - nSegmentLen;
	}
	else
	{
		UC_memcpy(pstDestChunk->pDataEnd, pSrc, nDataToWrite);

		if(pstDestChunk->pDataEnd == pstSharedMemroyChannel->pBuffer + pstChannel->nBufSize)
		{
			pstDestChunk->pDataEnd = pstSharedMemroyChannel->pBuffer;
		}
		else
		{
			pstDestChunk->pDataEnd += nDataToWrite;
		}
	}

	result = ERR_UEM_NOERROR;

	return result;
}


static uem_result copyAndMovePointerToRoundedQueue(SChannel *pstChannel, unsigned char *pSrc, int nDataToWrite, int nChunkIndex)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nSegmentLen = 0;
	int nRemainderLen = 0;
	SSharedMemoryChannel *pstSharedMemroyChannel = NULL;

	pstSharedMemroyChannel = (SSharedMemoryChannel *) pstChannel->pChannelStruct;

	// Chunk is divided into two segments because the start point of the chunk is close to the end of buffer range
	if(pstSharedMemroyChannel->pDataEnd + nDataToWrite > pstSharedMemroyChannel->pBuffer + pstChannel->nBufSize)
	{
		nSegmentLen = pstSharedMemroyChannel->pBuffer + pstChannel->nBufSize - pstSharedMemroyChannel->pDataEnd;
		nRemainderLen = nDataToWrite - nSegmentLen;

		UC_memcpy(pstSharedMemroyChannel->pDataEnd, pSrc, nSegmentLen);
		UC_memcpy(pstSharedMemroyChannel->pBuffer, pSrc + nSegmentLen, nRemainderLen);

		pstSharedMemroyChannel->pDataEnd = pstSharedMemroyChannel->pBuffer + nDataToWrite - nSegmentLen;
	}
	else
	{
		UC_memcpy(pstSharedMemroyChannel->pDataEnd, pSrc, nDataToWrite);

		if(pstSharedMemroyChannel->pDataEnd == pstSharedMemroyChannel->pBuffer + pstChannel->nBufSize)
		{
			pstSharedMemroyChannel->pDataEnd = pstSharedMemroyChannel->pBuffer;
		}
		else
		{
			pstSharedMemroyChannel->pDataEnd += nDataToWrite;
		}
	}

	pstSharedMemroyChannel->nDataLen += nDataToWrite;

	result = ERR_UEM_NOERROR;

	return result;
}


static uem_result copyFromRoundedChunk(unsigned char *pDest, SChannel *pstChannel, int nDataToRead, int nChunkIndex)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SChunk *pstSrcChunk = NULL;
	int nSegmentLen = 0;
	int nRemainderLen = 0;
	SSharedMemoryChannel *pstSharedMemroyChannel = NULL;

	pstSharedMemroyChannel = (SSharedMemoryChannel *) pstChannel->pChannelStruct;

	pstSrcChunk = &(pstSharedMemroyChannel->stInputPortChunk.astChunk[nChunkIndex]);

	// Chunk is divided into two segments because the start point of the chunk is close to the end of buffer range
	if(pstSrcChunk->pDataStart + nDataToRead > pstSharedMemroyChannel->pBuffer + pstChannel->nBufSize)
	{
		nSegmentLen = pstSharedMemroyChannel->pBuffer + pstChannel->nBufSize - pstSrcChunk->pDataStart;
		nRemainderLen = nDataToRead - nSegmentLen;

		UC_memcpy(pDest, pstSrcChunk->pDataStart, nSegmentLen);
		UC_memcpy(pDest + nSegmentLen, pstSharedMemroyChannel->pBuffer, nRemainderLen);
	}
	else
	{
		UC_memcpy(pDest, pstSrcChunk->pDataStart, nDataToRead);
	}

	result = ERR_UEM_NOERROR;

	return result;
}

static uem_result getMaximumAvailableNum(SPort *pstPort, int *pnMaxAvailableNum)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nCurrentSampleRateIndex = 0;
	int nMaxAvailableDataNum = 0;
	SPort *pstCurrentPort = NULL;

	nCurrentSampleRateIndex = pstPort->nCurrentSampleRateIndex;

	nMaxAvailableDataNum = pstPort->astSampleRates[nCurrentSampleRateIndex].nMaxAvailableDataNum;
	pstCurrentPort = pstPort->pstSubGraphPort;
	while(pstCurrentPort != NULL)
	{
		nCurrentSampleRateIndex = pstCurrentPort->nCurrentSampleRateIndex;
		nMaxAvailableDataNum = nMaxAvailableDataNum * pstCurrentPort->astSampleRates[nCurrentSampleRateIndex].nMaxAvailableDataNum;

		pstCurrentPort = pstCurrentPort->pstSubGraphPort;
	}

	*pnMaxAvailableNum = nMaxAvailableDataNum;

	result = ERR_UEM_NOERROR;

	return result;
}


static uem_result setInputChunks(SChannel *pstChannel)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;
	int nMaxAvailableNum = 0;
	SSharedMemoryChannel *pstSharedMemroyChannel = NULL;

	pstSharedMemroyChannel = (SSharedMemoryChannel *) pstChannel->pChannelStruct;

	for(nLoop = 0 ; nLoop < pstSharedMemroyChannel->stInputPortChunk.nChunkNum ; nLoop++)
	{
		result = getMaximumAvailableNum(&(pstChannel->stInputPort), &nMaxAvailableNum);
		ERRIFGOTO(result, _EXIT);
		pstSharedMemroyChannel->stInputPortChunk.astChunk[nLoop].nAvailableDataNum = nMaxAvailableNum;
		pstSharedMemroyChannel->stInputPortChunk.astChunk[nLoop].nChunkDataLen = pstSharedMemroyChannel->stInputPortChunk.nChunkLen;
		pstSharedMemroyChannel->stInputPortChunk.astChunk[nLoop].pDataStart = pstSharedMemroyChannel->pDataStart + pstSharedMemroyChannel->stInputPortChunk.nChunkLen * nLoop;
		pstSharedMemroyChannel->stInputPortChunk.astChunk[nLoop].pDataEnd = NULL;
		pstSharedMemroyChannel->stInputPortChunk.astChunk[nLoop].pChunkStart = NULL;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result makeAvailableInputChunkList(SChannel *pstChannel)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nChunkNum = 0;
	int nLoop = 0;
	SSharedMemoryChannel *pstSharedMemroyChannel = NULL;

	pstSharedMemroyChannel = (SSharedMemoryChannel *) pstChannel->pChannelStruct;

	nChunkNum = pstSharedMemroyChannel->stInputPortChunk.nChunkNum;

	for(nLoop = 0; nLoop < nChunkNum ; nLoop++)
	{
		if(nLoop == 0)
		{
			pstSharedMemroyChannel->pstAvailableInputChunkHead = &(pstSharedMemroyChannel->astAvailableInputChunkList[nLoop]);
			pstSharedMemroyChannel->astAvailableInputChunkList[nLoop].pstPrev = NULL;
		}
		else
		{
			pstSharedMemroyChannel->astAvailableInputChunkList[nLoop].pstPrev = &(pstSharedMemroyChannel->astAvailableInputChunkList[nLoop-1]);
		}

		if(nLoop == nChunkNum - 1)
		{
			pstSharedMemroyChannel->pstAvailableInputChunkTail = &(pstSharedMemroyChannel->astAvailableInputChunkList[nLoop]);
			pstSharedMemroyChannel->astAvailableInputChunkList[nLoop].pstNext = NULL;
		}
		else
		{
			pstSharedMemroyChannel->astAvailableInputChunkList[nLoop].pstNext = &(pstSharedMemroyChannel->astAvailableInputChunkList[nLoop+1]);
		}
	}

	//printf("available chunk create: %d : %d : %p : %d\n", pstChannel->nChannelIndex, nChunkNum, pstChannel->pstAvailableInputChunkHead, pstChannel->pstAvailableInputChunkHead->nChunkIndex);

	result = setInputChunks(pstChannel);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static uem_result moveDataPointerOfArrayQueue(SChannel *pstChannel)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSharedMemoryChannel *pstSharedMemroyChannel = NULL;

	pstSharedMemroyChannel = (SSharedMemoryChannel *) pstChannel->pChannelStruct;

	pstSharedMemroyChannel->nDataLen = pstSharedMemroyChannel->nDataLen - (pstSharedMemroyChannel->stInputPortChunk.nChunkNum * pstSharedMemroyChannel->stInputPortChunk.nChunkLen);

	// if the next pDataStart needs to be moved to the front pointer
	if(pstSharedMemroyChannel->pBuffer + pstChannel->nBufSize < pstSharedMemroyChannel->pDataStart + pstSharedMemroyChannel->stInputPortChunk.nChunkNum * pstSharedMemroyChannel->stInputPortChunk.nChunkLen)
	{
		pstSharedMemroyChannel->pDataStart = pstSharedMemroyChannel->pBuffer +
					(pstSharedMemroyChannel->stInputPortChunk.nChunkNum * pstSharedMemroyChannel->stInputPortChunk.nChunkLen) -
					(pstSharedMemroyChannel->pBuffer + pstChannel->nBufSize - pstSharedMemroyChannel->pDataStart);
	}
	else
	{
		pstSharedMemroyChannel->pDataStart = pstSharedMemroyChannel->pDataStart + pstSharedMemroyChannel->stInputPortChunk.nChunkNum * pstSharedMemroyChannel->stInputPortChunk.nChunkLen;

		if(pstSharedMemroyChannel->pDataStart == pstSharedMemroyChannel->pBuffer + pstChannel->nBufSize)
		{
			pstSharedMemroyChannel->pDataStart = pstSharedMemroyChannel->pBuffer;
		}
	}

	result = ERR_UEM_NOERROR;

	return result;
}


static uem_result readFromArrayQueue(SChannel *pstChannel, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SChunk *pstTargetChunk = NULL;
	SSharedMemoryChannel *pstSharedMemroyChannel = NULL;

	pstSharedMemroyChannel = (SSharedMemoryChannel *) pstChannel->pChannelStruct;

	result = UCThreadMutex_Lock(pstSharedMemroyChannel->hMutex);
	ERRIFGOTO(result, _EXIT);

	pstSharedMemroyChannel->nReadReferenceCount++;

	while(pstSharedMemroyChannel->stInputPortChunk.astChunk[nChunkIndex].nAvailableDataNum == 0)
	{
		if(pstSharedMemroyChannel->bReadExit == TRUE)
		{
			UEMASSIGNGOTO(result, ERR_UEM_SUSPEND, _EXIT_LOCK);
		}

		result = UCThreadMutex_Unlock(pstSharedMemroyChannel->hMutex);
		ERRIFGOTO(result, _EXIT);

		result = UCThreadEvent_WaitEvent(pstSharedMemroyChannel->hReadEvent);
		ERRIFGOTO(result, _EXIT);

		result = UCThreadMutex_Lock(pstSharedMemroyChannel->hMutex);
		ERRIFGOTO(result, _EXIT);
	}

	pstTargetChunk = &(pstSharedMemroyChannel->stInputPortChunk.astChunk[nChunkIndex]);

	if(nDataToRead != pstTargetChunk->nChunkDataLen)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT_LOCK);
	}

	result = copyFromRoundedChunk(pBuffer, pstChannel, nDataToRead, nChunkIndex);
	ERRIFGOTO(result, _EXIT_LOCK);

	pstTargetChunk->nAvailableDataNum--;

	if(pstTargetChunk->nAvailableDataNum == 0)
	{
		pstSharedMemroyChannel->stInputPortChunk.astChunk[nChunkIndex].nChunkDataLen = 0;

		result = removeChunkFromAvailableChunkList(pstChannel, nChunkIndex);
		ERRIFGOTO(result, _EXIT_LOCK);

		// All chunk list removed?
		if(pstSharedMemroyChannel->pstAvailableInputChunkHead == NULL)
		{
			result = moveDataPointerOfArrayQueue(pstChannel);
			ERRIFGOTO(result, _EXIT_LOCK);

			if(pstSharedMemroyChannel->nDataLen > 0)
			{
				int nCurrentSampleRateIndex = 0;
				int nExpectedConsumeSize = 0;

				nCurrentSampleRateIndex = pstChannel->stInputPort.nCurrentSampleRateIndex;
				nExpectedConsumeSize = pstChannel->stInputPort.astSampleRates[nCurrentSampleRateIndex].nSampleRate * pstChannel->stInputPort.nSampleSize;

				if(nExpectedConsumeSize <= pstSharedMemroyChannel->nDataLen)
				{
					result = makeAvailableInputChunkList(pstChannel);
					ERRIFGOTO(result, _EXIT_LOCK);
				}
			}
		}
	}

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
	pstSharedMemroyChannel->nReadReferenceCount--;

	if(pstSharedMemroyChannel->nWriteReferenceCount > 0)
	{
		UCThreadEvent_SetEvent(pstSharedMemroyChannel->hWriteEvent);
	}

	if(pstSharedMemroyChannel->nReadReferenceCount > 0 &&
			pstSharedMemroyChannel->pstAvailableInputChunkHead != NULL)
	{
		UCThreadEvent_SetEvent(pstSharedMemroyChannel->hReadEvent);
	}

	UCThreadMutex_Unlock(pstSharedMemroyChannel->hMutex);
_EXIT:
	return result;
}

static uem_result readFromGeneralQueue(SChannel *pstChannel, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSharedMemoryChannel *pstSharedMemroyChannel = NULL;

	pstSharedMemroyChannel = (SSharedMemoryChannel *) pstChannel->pChannelStruct;

	result = UCThreadMutex_Lock(pstSharedMemroyChannel->hMutex);
	ERRIFGOTO(result, _EXIT);

	pstSharedMemroyChannel->nReadReferenceCount++;

	// TODO: Error check out exit logic needed
	while(pstSharedMemroyChannel->nDataLen < nDataToRead)
	{
		if(pstSharedMemroyChannel->bReadExit == TRUE)
		{
			UEMASSIGNGOTO(result, ERR_UEM_SUSPEND, _EXIT_LOCK);
		}

		result = UCThreadMutex_Unlock(pstSharedMemroyChannel->hMutex);
		ERRIFGOTO(result, _EXIT);

		result = UCThreadEvent_WaitEvent(pstSharedMemroyChannel->hReadEvent);
		ERRIFGOTO(result, _EXIT);

		result = UCThreadMutex_Lock(pstSharedMemroyChannel->hMutex);
		ERRIFGOTO(result, _EXIT);
	}

	if(pstChannel->enChannelType == CHANNEL_TYPE_OUTPUT_ARRAY)
	{
		int nNewChunkIndex = 0 ;
		nNewChunkIndex = (pstSharedMemroyChannel->stOutputPortChunk.nChunkLen * pstSharedMemroyChannel->stOutputPortChunk.nChunkNum -
						MIN(pstSharedMemroyChannel->nDataLen, pstSharedMemroyChannel->stOutputPortChunk.nChunkLen * pstSharedMemroyChannel->stOutputPortChunk.nChunkNum))/pstSharedMemroyChannel->stOutputPortChunk.nChunkLen;

		nChunkIndex = nNewChunkIndex;
	}

	result = copyAndMovePointerFromRoundedQueue(pBuffer, pstChannel, nDataToRead);
	ERRIFGOTO(result, _EXIT_LOCK);

	if(pstChannel->enChannelType == CHANNEL_TYPE_OUTPUT_ARRAY)
	{
		int nLoop = 0;
		int nTotalDataRead = 0;
		int nDataRead = 0;

		for(nLoop = nChunkIndex ; nLoop < pstSharedMemroyChannel->stOutputPortChunk.nChunkNum ; nLoop++)
		{
			if(pstSharedMemroyChannel->stOutputPortChunk.astChunk[nLoop].nChunkDataLen < nDataToRead)
			{
				nDataRead = pstSharedMemroyChannel->stOutputPortChunk.astChunk[nLoop].nChunkDataLen;
			}
			else // pstChannel->stOutputPortChunk.astChunk[nLoop].nChunkDataLen >= nDataToRead
			{
				nDataRead = nDataToRead;
			}

			pstSharedMemroyChannel->stOutputPortChunk.astChunk[nLoop].nChunkDataLen -= nDataRead;
			nTotalDataRead += nDataRead;

			if(nTotalDataRead >= nDataToRead)
			{
				break;
			}
		}
	}

	*pnDataRead = nDataToRead;

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
pstSharedMemroyChannel->nReadReferenceCount--;

	if(pstSharedMemroyChannel->nWriteReferenceCount > 0)
	{
		UCThreadEvent_SetEvent(pstSharedMemroyChannel->hWriteEvent);
	}
	UCThreadMutex_Unlock(pstSharedMemroyChannel->hMutex);
_EXIT:
	return result;
}


uem_result UKSharedMemoryChannel_ReadFromQueue(SChannel *pstChannel, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead)
{
	uem_result result = ERR_UEM_UNKNOWN;

	// array channel
	if(pstChannel->enChannelType == CHANNEL_TYPE_INPUT_ARRAY || pstChannel->enChannelType == CHANNEL_TYPE_FULL_ARRAY)
	{
		result = readFromArrayQueue(pstChannel, pBuffer, nDataToRead, nChunkIndex, pnDataRead);
		ERRIFGOTO(result, _EXIT);
	}
	else // general channel
	{
		result = readFromGeneralQueue(pstChannel, pBuffer, nDataToRead, nChunkIndex, pnDataRead);
		ERRIFGOTO(result, _EXIT);
	}

	// to preserve, ERR_UEM_SUSPEND, do not set ERR_UEM_NOERROR here
_EXIT:
	return result;
}

uem_result UKSharedMemoryChannel_ReadFromBuffer(SChannel *pstChannel, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSharedMemoryChannel *pstSharedMemroyChannel = NULL;

	pstSharedMemroyChannel = (SSharedMemoryChannel *) pstChannel->pChannelStruct;

	result = UCThreadMutex_Lock(pstSharedMemroyChannel->hMutex);
	ERRIFGOTO(result, _EXIT);

	if(pstSharedMemroyChannel->nDataLen < nDataToRead)
	{
		if(pstSharedMemroyChannel->nDataLen > 0)
		{
			UC_memcpy(pBuffer, pstSharedMemroyChannel->pDataStart, pstSharedMemroyChannel->nDataLen);
		}

		*pnDataRead = pstSharedMemroyChannel->nDataLen;
	}
	else // pstChannel->nDataLen >= nDataToRead
	{
		UC_memcpy(pBuffer, pstSharedMemroyChannel->pDataStart, nDataToRead);

		*pnDataRead = nDataToRead;
	}

	result = ERR_UEM_NOERROR;

	UCThreadMutex_Unlock(pstSharedMemroyChannel->hMutex);
_EXIT:
	return result;
}

static uem_result writeToGeneralQueue(SChannel *pstChannel, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nCurrentSampleRateIndex = 0;
	int nExpectedConsumeSize = 0;
	SSharedMemoryChannel *pstSharedMemroyChannel = NULL;

	pstSharedMemroyChannel = (SSharedMemoryChannel *) pstChannel->pChannelStruct;

	result = UCThreadMutex_Lock(pstSharedMemroyChannel->hMutex);
	ERRIFGOTO(result, _EXIT);

	pstSharedMemroyChannel->nWriteReferenceCount++;

	// TODO: Error check out exit logic needed
	while(pstSharedMemroyChannel->nDataLen + nDataToWrite > pstChannel->nBufSize)
	{
		if(pstSharedMemroyChannel->bWriteExit == TRUE)
		{
			UEMASSIGNGOTO(result, ERR_UEM_SUSPEND, _EXIT_LOCK);
		}

		result = UCThreadMutex_Unlock(pstSharedMemroyChannel->hMutex);
		ERRIFGOTO(result, _EXIT);

		result = UCThreadEvent_WaitEvent(pstSharedMemroyChannel->hWriteEvent);
		ERRIFGOTO(result, _EXIT);

		result = UCThreadMutex_Lock(pstSharedMemroyChannel->hMutex);
		ERRIFGOTO(result, _EXIT);
	}

	result = copyAndMovePointerToRoundedQueue(pstChannel, pBuffer, nDataToWrite, nChunkIndex);
	ERRIFGOTO(result, _EXIT_LOCK);

	if(pstChannel->enChannelType == CHANNEL_TYPE_INPUT_ARRAY && pstSharedMemroyChannel->pstAvailableInputChunkHead == NULL)
	{
		nCurrentSampleRateIndex = pstChannel->stInputPort.nCurrentSampleRateIndex;
		nExpectedConsumeSize = pstChannel->stInputPort.astSampleRates[nCurrentSampleRateIndex].nSampleRate * pstChannel->stInputPort.nSampleSize;

		if(nExpectedConsumeSize <= pstSharedMemroyChannel->nDataLen)
		{
			result = makeAvailableInputChunkList(pstChannel);
			ERRIFGOTO(result, _EXIT_LOCK);
		}
	}

	*pnDataWritten = nDataToWrite;

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
	pstSharedMemroyChannel->nWriteReferenceCount--;

	if(pstSharedMemroyChannel->nReadReferenceCount > 0)
	{
		UCThreadEvent_SetEvent(pstSharedMemroyChannel->hReadEvent);
	}
	UCThreadMutex_Unlock(pstSharedMemroyChannel->hMutex);
_EXIT:
	return result;
}


static uem_result clearOutputChunkInfo(SChannel *pstChannel)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;
	void *pNextChunkDataEnd = NULL;
	int nSegmentLen = 0;
	SSharedMemoryChannel *pstSharedMemroyChannel = NULL;

	pstSharedMemroyChannel = (SSharedMemoryChannel *) pstChannel->pChannelStruct;

	if(pstSharedMemroyChannel->pDataEnd == pstSharedMemroyChannel->pBuffer + pstChannel->nBufSize)
	{
		pstSharedMemroyChannel->pDataEnd = pstSharedMemroyChannel->pBuffer;
	}

	for(nLoop = 0 ; nLoop < pstSharedMemroyChannel->stOutputPortChunk.nChunkNum ; nLoop++)
	{
		// if(pstChannel->stOutputPortChunk.astChunk[nLoop].pDataEnd )
		pNextChunkDataEnd = pstSharedMemroyChannel->pDataEnd + pstSharedMemroyChannel->stOutputPortChunk.nChunkLen * nLoop;

		if(pNextChunkDataEnd >= pstSharedMemroyChannel->pBuffer + pstChannel->nBufSize)
		{
			nSegmentLen = pNextChunkDataEnd - (pstSharedMemroyChannel->pBuffer + pstChannel->nBufSize);
			pstSharedMemroyChannel->stOutputPortChunk.astChunk[nLoop].pDataEnd = pstSharedMemroyChannel->pBuffer + nSegmentLen;
		}
		else
		{
			pstSharedMemroyChannel->stOutputPortChunk.astChunk[nLoop].pDataEnd = pNextChunkDataEnd;
		}

		// maximum available number is not needed for output chunk
		pstSharedMemroyChannel->stOutputPortChunk.astChunk[nLoop].nAvailableDataNum = 1;
	}

	pstSharedMemroyChannel->nWrittenOutputChunkNum = 0;

	result = ERR_UEM_NOERROR;
//_EXIT:
	return result;
}



static uem_result writeToArrayQueue(SChannel *pstChannel, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nCurrentSampleRateIndex = 0;
	int nCurrentReadSampleRateIndex = 0;
	int nExpectedProduceSize = 0;
	int nExpectedConsumeSize = 0;
	SSharedMemoryChannel *pstSharedMemroyChannel = NULL;

	pstSharedMemroyChannel = (SSharedMemoryChannel *) pstChannel->pChannelStruct;

	result = UCThreadMutex_Lock(pstSharedMemroyChannel->hMutex);
	ERRIFGOTO(result, _EXIT);

	pstSharedMemroyChannel->nWriteReferenceCount++;

	nCurrentSampleRateIndex = pstChannel->stOutputPort.nCurrentSampleRateIndex;
	nExpectedProduceSize = pstChannel->stOutputPort.astSampleRates[nCurrentSampleRateIndex].nSampleRate * pstChannel->stOutputPort.nSampleSize;

	// TODO: Error check out exit logic needed
	while(pstSharedMemroyChannel->nDataLen + nExpectedProduceSize > pstChannel->nBufSize || // nBuffer is full or
		(pstSharedMemroyChannel->nWrittenOutputChunkNum > 0 && pstSharedMemroyChannel->nWrittenOutputChunkNum < pstSharedMemroyChannel->stOutputPortChunk.nChunkNum &&
				pstSharedMemroyChannel->stOutputPortChunk.astChunk[nChunkIndex].nChunkDataLen > 0)) // current chunk index is already filled with data
	{
		if(pstSharedMemroyChannel->bWriteExit == TRUE)
		{
			UEMASSIGNGOTO(result, ERR_UEM_SUSPEND, _EXIT_LOCK);
		}

		result = UCThreadMutex_Unlock(pstSharedMemroyChannel->hMutex);
		ERRIFGOTO(result, _EXIT);

		result = UCThreadEvent_WaitEvent(pstSharedMemroyChannel->hWriteEvent);
		ERRIFGOTO(result, _EXIT);

		result = UCThreadMutex_Lock(pstSharedMemroyChannel->hMutex);
		ERRIFGOTO(result, _EXIT);
	}

	// Chunk needs to be initialized
	if(pstSharedMemroyChannel->nWrittenOutputChunkNum < 0 || pstSharedMemroyChannel->nWrittenOutputChunkNum >= pstSharedMemroyChannel->stOutputPortChunk.nChunkNum)
	{
		result = clearOutputChunkInfo(pstChannel);
		ERRIFGOTO(result, _EXIT_LOCK);
	}

	result = copyToRoundedChunk(pstChannel, pBuffer, nDataToWrite, nChunkIndex);
	ERRIFGOTO(result, _EXIT_LOCK);

	pstSharedMemroyChannel->stOutputPortChunk.astChunk[nChunkIndex].nChunkDataLen = nDataToWrite;

	if(pstSharedMemroyChannel->stOutputPortChunk.astChunk[nChunkIndex].nChunkDataLen == pstSharedMemroyChannel->stOutputPortChunk.nChunkLen)
	{
		pstSharedMemroyChannel->nWrittenOutputChunkNum++;
	}

	if(pstSharedMemroyChannel->nWrittenOutputChunkNum == pstSharedMemroyChannel->stOutputPortChunk.nChunkNum)
	{
		void *pNewEnd = NULL;
		int nSegmentLen = 0;
		pNewEnd = pstSharedMemroyChannel->pDataEnd + nExpectedProduceSize;
		if(pNewEnd >= pstSharedMemroyChannel->pBuffer + pstChannel->nBufSize)
		{
			nSegmentLen = pNewEnd - (pstSharedMemroyChannel->pBuffer + pstChannel->nBufSize);
			pNewEnd = pstSharedMemroyChannel->pBuffer + nSegmentLen;
		}
		pstSharedMemroyChannel->nDataLen += nExpectedProduceSize;
	}

	nCurrentReadSampleRateIndex = pstChannel->stInputPort.nCurrentSampleRateIndex;
	nExpectedConsumeSize = pstChannel->stInputPort.astSampleRates[nCurrentReadSampleRateIndex].nSampleRate * pstChannel->stInputPort.nSampleSize;

	if(pstChannel->enChannelType == CHANNEL_TYPE_FULL_ARRAY && nExpectedConsumeSize <= pstSharedMemroyChannel->nDataLen)
	{
		result = makeAvailableInputChunkList(pstChannel);
		ERRIFGOTO(result, _EXIT_LOCK);
	}

	*pnDataWritten = nDataToWrite;

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
	pstSharedMemroyChannel->nWriteReferenceCount--;

	if(pstSharedMemroyChannel->nReadReferenceCount > 0)
	{
		UCThreadEvent_SetEvent(pstSharedMemroyChannel->hReadEvent);
	}

	if(pstSharedMemroyChannel->nWriteReferenceCount > 0)
	{
		UCThreadEvent_SetEvent(pstSharedMemroyChannel->hWriteEvent);
	}

	UCThreadMutex_Unlock(pstSharedMemroyChannel->hMutex);
_EXIT:
	return result;
}


uem_result UKSharedMemoryChannel_WriteToQueue (SChannel *pstChannel, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten)
{
	uem_result result = ERR_UEM_UNKNOWN;

	// array channel
	if(pstChannel->enChannelType == CHANNEL_TYPE_OUTPUT_ARRAY || pstChannel->enChannelType == CHANNEL_TYPE_FULL_ARRAY)
	{
		result = writeToArrayQueue(pstChannel, pBuffer, nDataToWrite, nChunkIndex, pnDataWritten);
		ERRIFGOTO(result, _EXIT);
	}
	else
	{
		result = writeToGeneralQueue(pstChannel, pBuffer, nDataToWrite, nChunkIndex, pnDataWritten);
		ERRIFGOTO(result, _EXIT);
	}

	// to preserve, ERR_UEM_SUSPEND, do not set ERR_UEM_NOERROR here
_EXIT:
	return result;
}


uem_result UKSharedMemoryChannel_WriteToBuffer (SChannel *pstChannel, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSharedMemoryChannel *pstSharedMemroyChannel = NULL;

	pstSharedMemroyChannel = (SSharedMemoryChannel *) pstChannel->pChannelStruct;


	result = UCThreadMutex_Lock(pstSharedMemroyChannel->hMutex);
	ERRIFGOTO(result, _EXIT);

	if(pstChannel->nBufSize >= nDataToWrite)
	{
		if (nDataToWrite > 0)
		{
			UC_memcpy(pstSharedMemroyChannel->pDataStart, pBuffer, nDataToWrite);
		}

		*pnDataWritten = nDataToWrite;
	}
	else // pstChannel->nBufSize < nDataToWrite
	{
		UC_memcpy(pstSharedMemroyChannel->pDataStart, pBuffer, pstChannel->nBufSize);

		*pnDataWritten = pstChannel->nBufSize;
	}

	result = ERR_UEM_NOERROR;

	UCThreadMutex_Unlock(pstSharedMemroyChannel->hMutex);
_EXIT:
	return result;
}

static uem_result getAvailableChunkFromArrayQueue(SChannel *pstChannel, OUT int *pnChunkIndex)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nAvailableIndex = 0;
	SAvailableChunk *pstNextChunk = NULL;
	SAvailableChunk *pstCurChunk = NULL;
	SSharedMemoryChannel *pstSharedMemroyChannel = NULL;

	pstSharedMemroyChannel = (SSharedMemoryChannel *) pstChannel->pChannelStruct;

	result = UCThreadMutex_Lock(pstSharedMemroyChannel->hMutex);
	ERRIFGOTO(result, _EXIT);

	pstSharedMemroyChannel->nReadReferenceCount++;

	while(pstSharedMemroyChannel->pstAvailableInputChunkHead == NULL)
	{
		if(pstSharedMemroyChannel->bReadExit == TRUE)
		{
			UEMASSIGNGOTO(result, ERR_UEM_SUSPEND, _EXIT_LOCK);
		}

		result = UCThreadMutex_Unlock(pstSharedMemroyChannel->hMutex);
		ERRIFGOTO(result, _EXIT);

		result = UCThreadEvent_WaitEvent(pstSharedMemroyChannel->hReadEvent);
		ERRIFGOTO(result, _EXIT);

		result = UCThreadMutex_Lock(pstSharedMemroyChannel->hMutex);
		ERRIFGOTO(result, _EXIT);
	}

	nAvailableIndex = pstSharedMemroyChannel->pstAvailableInputChunkHead->nChunkIndex;
	pstCurChunk = pstSharedMemroyChannel->pstAvailableInputChunkHead;

	pstSharedMemroyChannel->pstAvailableInputChunkTail->pstNext = pstCurChunk;
	pstCurChunk->pstPrev = pstSharedMemroyChannel->pstAvailableInputChunkTail;

	pstNextChunk = pstCurChunk->pstNext;
	pstNextChunk->pstPrev = NULL;

	pstSharedMemroyChannel->pstAvailableInputChunkTail = pstCurChunk;
	pstSharedMemroyChannel->pstAvailableInputChunkTail->pstNext = NULL;

	pstSharedMemroyChannel->pstAvailableInputChunkHead = pstNextChunk;

	*pnChunkIndex = nAvailableIndex;

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
pstSharedMemroyChannel->nReadReferenceCount--;

	if(pstSharedMemroyChannel->nWriteReferenceCount > 0)
	{
		UCThreadEvent_SetEvent(pstSharedMemroyChannel->hWriteEvent);
	}
	UCThreadMutex_Unlock(pstSharedMemroyChannel->hMutex);
_EXIT:
	return result;
}


static uem_result getAvailableChunkFromGeneralQueue(SChannel *pstChannel, OUT int *pnChunkIndex)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nDataToRead = 0;
	SSharedMemoryChannel *pstSharedMemroyChannel = NULL;

	pstSharedMemroyChannel = (SSharedMemoryChannel *) pstChannel->pChannelStruct;

	result = UCThreadMutex_Lock(pstSharedMemroyChannel->hMutex);
	ERRIFGOTO(result, _EXIT);

	pstSharedMemroyChannel->nReadReferenceCount++;

	// wait until the data is arrived
	while(pstSharedMemroyChannel->nDataLen < nDataToRead)
	{
		if(pstSharedMemroyChannel->bReadExit == TRUE)
		{
			UEMASSIGNGOTO(result, ERR_UEM_SUSPEND, _EXIT_LOCK);
		}

		result = UCThreadMutex_Unlock(pstSharedMemroyChannel->hMutex);
		ERRIFGOTO(result, _EXIT);

		result = UCThreadEvent_WaitEvent(pstSharedMemroyChannel->hReadEvent);
		ERRIFGOTO(result, _EXIT);

		result = UCThreadMutex_Lock(pstSharedMemroyChannel->hMutex);
		ERRIFGOTO(result, _EXIT);
	}

	// index is always zero.
	*pnChunkIndex = 0;

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
	pstSharedMemroyChannel->nReadReferenceCount--;

	if(pstSharedMemroyChannel->nWriteReferenceCount > 0)
	{
		UCThreadEvent_SetEvent(pstSharedMemroyChannel->hWriteEvent);
	}
	UCThreadMutex_Unlock(pstSharedMemroyChannel->hMutex);
_EXIT:
	return result;
}


uem_result UKSharedMemoryChannel_GetAvailableChunk (SChannel *pstChannel, OUT int *pnChunkIndex)
{
	uem_result result = ERR_UEM_UNKNOWN;

	if(pstChannel->enChannelType == CHANNEL_TYPE_GENERAL || pstChannel->enChannelType == CHANNEL_TYPE_OUTPUT_ARRAY)
	{
		result = getAvailableChunkFromGeneralQueue(pstChannel, pnChunkIndex);
		ERRIFGOTO(result, _EXIT);
	}
	else
	{
		result = getAvailableChunkFromArrayQueue(pstChannel, pnChunkIndex);
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKSharedMemoryChannel_GetNumOfAvailableData (SChannel *pstChannel, IN int nChunkIndex, OUT int *pnDataNum)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSharedMemoryChannel *pstSharedMemroyChannel = NULL;

	pstSharedMemroyChannel = (SSharedMemoryChannel *) pstChannel->pChannelStruct;

	result = UCThreadMutex_Lock(pstSharedMemroyChannel->hMutex);
	ERRIFGOTO(result, _EXIT);

	if(pstChannel->enChannelType == CHANNEL_TYPE_GENERAL || pstChannel->enChannelType == CHANNEL_TYPE_OUTPUT_ARRAY)
	{
		*pnDataNum = pstSharedMemroyChannel->nDataLen;
	}
	else // if( pstChannel->enChannelType == CHANNEL_TYPE_INPUT_ARRAY || pstChannel->enChannelType == CHANNEL_TYPE_FULL_ARRAY)
	{
		if(pstSharedMemroyChannel->pstAvailableInputChunkHead != NULL)
		{
			*pnDataNum = pstSharedMemroyChannel->stInputPortChunk.astChunk[nChunkIndex].nChunkDataLen;
		}
		else
		{
			*pnDataNum = 0;
		}
	}

	result = ERR_UEM_NOERROR;

	UCThreadMutex_Unlock(pstSharedMemroyChannel->hMutex);
_EXIT:
	return result;
}

uem_result UKGPUSharedMemoryChannel_SetExit(SChannel *pstChannel, int nExitFlag)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSharedMemoryChannel *pstSharedMemroyChannel = NULL;

	pstSharedMemroyChannel = (SSharedMemoryChannel *) pstChannel->pChannelStruct;

	result = UCThreadMutex_Lock(pstSharedMemroyChannel->hMutex);
	ERRIFGOTO(result, _EXIT);

	if((nExitFlag & EXIT_FLAG_READ) != 0)
	{
		pstSharedMemroyChannel->bReadExit = TRUE;

		result = UCThreadEvent_SetEvent(pstSharedMemroyChannel->hReadEvent);
		ERRIFGOTO(result, _EXIT_LOCK);
	}

	if((nExitFlag & EXIT_FLAG_WRITE) != 0)
	{
		pstSharedMemroyChannel->bWriteExit = TRUE;

		result = UCThreadEvent_SetEvent(pstSharedMemroyChannel->hWriteEvent);
		ERRIFGOTO(result, _EXIT_LOCK);
	}

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
	UCThreadMutex_Unlock(pstSharedMemroyChannel->hMutex);
_EXIT:
	return result;
}


uem_result UKGPUSharedMemoryChannel_ClearExit(SChannel *pstChannel, int nExitFlag)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSharedMemoryChannel *pstSharedMemroyChannel = NULL;

	pstSharedMemroyChannel = (SSharedMemoryChannel *) pstChannel->pChannelStruct;

	result = UCThreadMutex_Lock(pstSharedMemroyChannel->hMutex);
	ERRIFGOTO(result, _EXIT);

	if((nExitFlag & EXIT_FLAG_READ) != 0)
	{
		pstSharedMemroyChannel->bReadExit = FALSE;
	}

	if((nExitFlag & EXIT_FLAG_WRITE) != 0)
	{
		pstSharedMemroyChannel->bWriteExit = FALSE;
	}

	result = ERR_UEM_NOERROR;

	UCThreadMutex_Unlock(pstSharedMemroyChannel->hMutex);
_EXIT:
	return result;
}


uem_result UKSharedMemoryChannel_Finalize(SChannel *pstChannel)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSharedMemoryChannel *pstSharedMemroyChannel = NULL;

	pstSharedMemroyChannel = (SSharedMemoryChannel *) pstChannel->pChannelStruct;

	// ignore error
	UCThreadMutex_Destroy(&(pstSharedMemroyChannel->hMutex));

	// ignore error
	UCThreadEvent_Destroy(&(pstSharedMemroyChannel->hReadEvent));
	UCThreadEvent_Destroy(&(pstSharedMemroyChannel->hWriteEvent));

	result = ERR_UEM_NOERROR;

	return result;
}






