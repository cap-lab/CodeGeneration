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


static uem_result setChunkNumAndLen(SPort *pstPort, SChunkInfo *pstChunkInfo)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nCurrentSampleRateIndex = 0;
	SPort *pstCurrentPort = NULL;
	int nOuterMostSampleRate = 0;
	STask *pstCurTask = NULL;
	int nLoop = 0;

	nCurrentSampleRateIndex = pstPort->nCurrentSampleRateIndex;

	nOuterMostSampleRate = pstPort->astSampleRates[nCurrentSampleRateIndex].nSampleRate;

	pstCurrentPort = pstPort;
	while(pstCurrentPort->pstSubGraphPort != NULL)
	{
		pstCurrentPort = pstCurrentPort->pstSubGraphPort;
	}

	if(pstCurrentPort != pstPort)
	{
		nCurrentSampleRateIndex = pstCurrentPort->nCurrentSampleRateIndex;

		pstChunkInfo->nChunkNum = nOuterMostSampleRate / pstCurrentPort->astSampleRates[nCurrentSampleRateIndex].nSampleRate;
		pstChunkInfo->nChunkLen = pstCurrentPort->astSampleRates[nCurrentSampleRateIndex].nSampleRate * pstCurrentPort->nSampleSize;
	}
	else
	{

		result = UKTask_GetTaskFromTaskId(pstPort->nTaskId, &pstCurTask);
		ERRIFGOTO(result, _EXIT);

		if(pstCurTask->pstLoopInfo == NULL) // general task
		{
			pstChunkInfo->nChunkNum = 1;
			pstChunkInfo->nChunkLen = nOuterMostSampleRate;

		}
		else if(pstCurTask->pstLoopInfo->enType == LOOP_TYPE_DATA &&
			pstPort->astSampleRates[nCurrentSampleRateIndex].nMaxAvailableDataNum == 1)
		{
			pstChunkInfo->nChunkNum = nOuterMostSampleRate / pstCurTask->pstLoopInfo->nLoopCount;
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

	// information clear
	// pDataStart => clear (pBuffer)
	// nDataLen => clear (0)
	pstChannel->pDataStart = pstChannel->pBuffer;
	pstChannel->pDataEnd = pstChannel->pBuffer;
	pstChannel->nDataLen = 0;
	pstChannel->nReferenceCount = 0;

	// the chunk num and chunk length is dependent to sample rate of mode transition
	result = setChunkNumAndLen(&(pstChannel->stInputPort), &(pstChannel->stInputPortChunk));
	ERRIFGOTO(result, _EXIT);

	result = setChunkNumAndLen(&(pstChannel->stOutputPort), &(pstChannel->stOutputPortChunk));
	ERRIFGOTO(result, _EXIT);

	pstChannel->nWrittenOutputChunkNum = CHUNK_NUM_NOT_INITIALIZED;

	pstChannel->pstAvailableInputChunkHead = NULL;
	pstChannel->pstAvailableInputChunkTail = NULL;

	for(nLoop = 0 ; nLoop < pstChannel->nMaxChunkNum ; nLoop++)
	{
		pstChannel->astAvailableInputChunkList[nLoop].pstPrev = NULL;
		pstChannel->astAvailableInputChunkList[nLoop].pstNext = NULL;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKSharedMemoryChannel_Initialize(SChannel *pstChannel)
{
	uem_result result = ERR_UEM_UNKNOWN;
	// initialize buffer
	// If not set, initialize those things, // pBuffer => is NULL => alloc
	if(pstChannel->pBuffer == NULL)
	{
		pstChannel->pBuffer = UC_malloc(pstChannel->nBufSize);
		ERRMEMGOTO(pstChannel->pBuffer, result, _EXIT);
	}

	// hMutex => initialize/create
	result = UCThreadMutex_Create(&(pstChannel->hMutex));
	ERRIFGOTO(result, _EXIT);

	// hEvent => initialize/create
	result = UCThreadEvent_Create(&(pstChannel->hEvent));
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

	pstAvailableChunk = pstChannel->pstAvailableInputChunkHead;

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
				pstChannel->pstAvailableInputChunkHead = pstAvailableChunk->pstNext;
			}

			if(pstNextChunk != NULL)
			{
				pstNextChunk->pstPrev = pstPreviousChunk;
			}
			else // pstNextChunk == NULL => pstAvailableChunk is a tail chunk
			{
				pstChannel->pstAvailableInputChunkTail = pstAvailableChunk->pstPrev;
			}

			pstAvailableChunk->pstPrev = NULL;
			pstAvailableChunk->pstNext = NULL;
			break;
		}
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

	// data is divided into two segments because the start point of the chunk is close to the end of buffer range
	if(pstChannel->pDataStart + nDataToRead > pstChannel->pBuffer + pstChannel->nBufSize)
	{
		nSegmentLen = pstChannel->pBuffer + pstChannel->nBufSize - pstChannel->pDataStart;
		nRemainderLen = nDataToRead - nSegmentLen;

		UC_memcpy(pDest, pstChannel->pDataStart, nSegmentLen);
		UC_memcpy(pDest + nSegmentLen, pstChannel->pBuffer, nRemainderLen);

		pstChannel->pDataStart = pstChannel->pBuffer + nRemainderLen;
	}
	else
	{
		UC_memcpy(pDest, pstChannel->pDataStart, nDataToRead);

		if(pstChannel->pDataStart + nDataToRead == pstChannel->pBuffer + pstChannel->nBufSize)
		{
			pstChannel->pDataStart = pstChannel->pBuffer;
		}
		else
		{
			pstChannel->pDataStart += nDataToRead;
		}
	}

	pstChannel->nDataLen -= nDataToRead;

	result = ERR_UEM_NOERROR;

	return result;
}

static uem_result copyToRoundedChunk(SChannel *pstChannel, unsigned char *pSrc, int nDataToWrite, int nChunkIndex)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SChunk *pstDestChunk = NULL;
	int nSegmentLen = 0;
	int nRemainderLen = 0;

	pstDestChunk = &(pstChannel->stOutputPortChunk.astChunk[nChunkIndex]);

	// Chunk is divided into two segments because the start point of the chunk is close to the end of buffer range
	if(pstDestChunk->pDataEnd + nDataToWrite > pstChannel->pBuffer + pstChannel->nBufSize)
	{
		nSegmentLen = pstChannel->pBuffer + pstChannel->nBufSize - pstDestChunk->pDataEnd;
		nRemainderLen = nDataToWrite - nSegmentLen;

		UC_memcpy(pstDestChunk->pDataEnd, pSrc, nSegmentLen);
		UC_memcpy(pstChannel->pBuffer, pSrc + nSegmentLen, nRemainderLen);
	}
	else
	{
		UC_memcpy(pstDestChunk->pDataEnd, pSrc, nDataToWrite);
	}

	result = ERR_UEM_NOERROR;

	return result;
}


static uem_result copyAndMovePointerToRoundedQueue(SChannel *pstChannel, unsigned char *pSrc, int nDataToWrite, int nChunkIndex)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nSegmentLen = 0;
	int nRemainderLen = 0;

	// Chunk is divided into two segments because the start point of the chunk is close to the end of buffer range
	if(pstChannel->pDataEnd + nDataToWrite > pstChannel->pBuffer + pstChannel->nBufSize)
	{
		nSegmentLen = pstChannel->pBuffer + pstChannel->nBufSize - pstChannel->pDataEnd;
		nRemainderLen = nDataToWrite - nSegmentLen;

		UC_memcpy(pstChannel->pDataEnd, pSrc, nSegmentLen);
		UC_memcpy(pstChannel->pBuffer, pSrc + nSegmentLen, nRemainderLen);

		pstChannel->pDataEnd = pstChannel->pBuffer + nDataToWrite - nSegmentLen;
	}
	else
	{
		UC_memcpy(pstChannel->pDataEnd, pSrc, nDataToWrite);

		if(pstChannel->pDataEnd == pstChannel->pBuffer + pstChannel->nBufSize)
		{
			pstChannel->pDataEnd = pstChannel->pBuffer;
		}
		else
		{
			pstChannel->pDataEnd += nDataToWrite;
		}
	}

	pstChannel->nDataLen += nDataToWrite;

	result = ERR_UEM_NOERROR;

	return result;
}


static uem_result copyFromRoundedChunk(unsigned char *pDest, SChannel *pstChannel, int nDataToRead, int nChunkIndex)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SChunk *pstSrcChunk = NULL;
	int nSegmentLen = 0;
	int nRemainderLen = 0;

	pstSrcChunk = &(pstChannel->stInputPortChunk.astChunk[nChunkIndex]);

	// Chunk is divided into two segments because the start point of the chunk is close to the end of buffer range
	if(pstSrcChunk->pDataStart + nDataToRead > pstChannel->pBuffer + pstChannel->nBufSize)
	{
		nSegmentLen = pstChannel->pBuffer + pstChannel->nBufSize - pstSrcChunk->pDataStart;
		nRemainderLen = nDataToRead - nSegmentLen;

		UC_memcpy(pDest, pstSrcChunk->pDataStart, nSegmentLen);
		UC_memcpy(pDest + nSegmentLen, pstChannel->pBuffer, nRemainderLen);
	}
	else
	{
		UC_memcpy(pDest, pstSrcChunk->pDataStart, nDataToRead);
	}

	result = ERR_UEM_NOERROR;

	return result;
}

static uem_result GetMaximumAvailableNum(SPort *pstPort, int *pnMaxAvailableNum)
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
	int nExpectedConsumeSize = 0;
	int nCurrentSampleRateIndex = 0;

	nCurrentSampleRateIndex = pstChannel->stInputPort.nCurrentSampleRateIndex;
	nExpectedConsumeSize = pstChannel->stInputPort.astSampleRates[nCurrentSampleRateIndex].nSampleRate * pstChannel->stInputPort.nSampleSize;

	for(nLoop = 0 ; nLoop < pstChannel->stInputPortChunk.nChunkNum ; nLoop++)
	{
		result = GetMaximumAvailableNum(&(pstChannel->stInputPort), &nMaxAvailableNum);
		ERRIFGOTO(result, _EXIT);
		pstChannel->stInputPortChunk.astChunk[nLoop].nAvailableDataNum = nMaxAvailableNum;
		pstChannel->stInputPortChunk.astChunk[nLoop].nChunkDataLen = pstChannel->stInputPortChunk.nChunkLen;
		pstChannel->stInputPortChunk.astChunk[nLoop].pDataStart = pstChannel->pDataStart + nExpectedConsumeSize * nLoop;
		pstChannel->stInputPortChunk.astChunk[nLoop].pDataEnd = NULL;
		pstChannel->stInputPortChunk.astChunk[nLoop].pChunkStart = NULL;
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

	nChunkNum = pstChannel->stInputPortChunk.nChunkNum;

	for(nLoop = 0; nLoop < nChunkNum ; nLoop++)
	{
		if(nLoop == 0)
		{
			pstChannel->pstAvailableInputChunkHead = &(pstChannel->astAvailableInputChunkList[nLoop]);
			pstChannel->astAvailableInputChunkList[nLoop].pstPrev = NULL;
		}
		else
		{
			pstChannel->astAvailableInputChunkList[nLoop].pstPrev = &(pstChannel->astAvailableInputChunkList[nLoop-1]);
		}

		if(nLoop - 1 == nChunkNum)
		{
			pstChannel->pstAvailableInputChunkTail = &(pstChannel->astAvailableInputChunkList[nLoop]);
			pstChannel->astAvailableInputChunkList[nLoop].pstNext = NULL;
		}
		else
		{
			pstChannel->astAvailableInputChunkList[nLoop].pstNext = &(pstChannel->astAvailableInputChunkList[nLoop+1]);
		}
	}

	result = setInputChunks(pstChannel);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static uem_result moveDataPointerOfArrayQueue(SChannel *pstChannel)
{
	uem_result result = ERR_UEM_UNKNOWN;

	pstChannel->nDataLen = pstChannel->nDataLen - (pstChannel->stInputPortChunk.nChunkNum * pstChannel->stInputPortChunk.nChunkLen);

	// if the next pDataStart needs to be moved to the front pointer
	if(pstChannel->pBuffer + pstChannel->nBufSize < pstChannel->pDataStart + pstChannel->stInputPortChunk.nChunkNum * pstChannel->stInputPortChunk.nChunkLen)
	{
		pstChannel->pDataStart = pstChannel->pBuffer +
					(pstChannel->stInputPortChunk.nChunkNum * pstChannel->stInputPortChunk.nChunkLen) -
					(pstChannel->pBuffer + pstChannel->nBufSize - pstChannel->pDataStart);
	}
	else
	{
		pstChannel->pDataStart = pstChannel->pDataStart + pstChannel->stInputPortChunk.nChunkNum * pstChannel->stInputPortChunk.nChunkLen;

		if(pstChannel->pDataStart == pstChannel->pBuffer + pstChannel->nBufSize)
		{
			pstChannel->pDataStart = pstChannel->pBuffer;
		}
	}

	result = ERR_UEM_NOERROR;

	return result;
}


static uem_result readFromArrayQueue(SChannel *pstChannel, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SChunk *pstTargetChunk = NULL;

	result = UCThreadMutex_Lock(pstChannel->hMutex);
	ERRIFGOTO(result, _EXIT);

	pstChannel->nReferenceCount++;

	// TODO: Error check out exit logic needed
	while(pstChannel->stInputPortChunk.astChunk[nChunkIndex].nAvailableDataNum == 0)
	{
		result = UCThreadMutex_Unlock(pstChannel->hMutex);
		ERRIFGOTO(result, _EXIT);

		result = UCThreadEvent_WaitEvent(pstChannel->hEvent);
		ERRIFGOTO(result, _EXIT);

		result = UCThreadMutex_Lock(pstChannel->hMutex);
		ERRIFGOTO(result, _EXIT);
	}

	pstTargetChunk = &(pstChannel->stInputPortChunk.astChunk[nChunkIndex]);

	if(nDataToRead != pstTargetChunk->nChunkDataLen)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT_LOCK);
	}

	result = copyFromRoundedChunk(pBuffer, pstChannel, nDataToRead, nChunkIndex);
	ERRIFGOTO(result, _EXIT_LOCK);

	pstTargetChunk->nAvailableDataNum--;

	if(pstTargetChunk->nAvailableDataNum == 0)
	{
		pstChannel->stInputPortChunk.astChunk[nChunkIndex].nChunkDataLen = 0;

		result = removeChunkFromAvailableChunkList(pstChannel, nChunkIndex);
		ERRIFGOTO(result, _EXIT_LOCK);

		// All chunk list removed?
		if(pstChannel->pstAvailableInputChunkHead == NULL)
		{
			result = moveDataPointerOfArrayQueue(pstChannel);
			ERRIFGOTO(result, _EXIT_LOCK);
		}

		if(pstChannel->nDataLen > 0)
		{
			int nCurrentSampleRateIndex = 0;
			int nExpectedConsumeSize = 0;

			nCurrentSampleRateIndex = pstChannel->stInputPort.nCurrentSampleRateIndex;
			nExpectedConsumeSize = pstChannel->stInputPort.astSampleRates[nCurrentSampleRateIndex].nSampleRate * pstChannel->stInputPort.nSampleSize;

			if(nExpectedConsumeSize <= pstChannel->nDataLen)
			{
				result = makeAvailableInputChunkList(pstChannel);
				ERRIFGOTO(result, _EXIT_LOCK);
			}
		}
	}

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
	pstChannel->nReferenceCount--;

	if(pstChannel->nReferenceCount > 0)
	{
		UCThreadEvent_SetEvent(pstChannel->hEvent);
	}
	UCThreadMutex_Unlock(pstChannel->hMutex);
_EXIT:
	return result;
}

static uem_result readFromGeneralQueue(SChannel *pstChannel, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = UCThreadMutex_Lock(pstChannel->hMutex);
	ERRIFGOTO(result, _EXIT);

	pstChannel->nReferenceCount++;

	// TODO: Error check out exit logic needed
	while(pstChannel->nDataLen < nDataToRead)
	{
		result = UCThreadMutex_Unlock(pstChannel->hMutex);
		ERRIFGOTO(result, _EXIT);

		result = UCThreadEvent_WaitEvent(pstChannel->hEvent);
		ERRIFGOTO(result, _EXIT);

		result = UCThreadMutex_Lock(pstChannel->hMutex);
		ERRIFGOTO(result, _EXIT);
	}

	result = copyAndMovePointerFromRoundedQueue(pBuffer, pstChannel, nDataToRead);
	ERRIFGOTO(result, _EXIT_LOCK);

	*pnDataRead = nDataToRead;

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
	pstChannel->nReferenceCount--;

	if(pstChannel->nReferenceCount > 0)
	{
		UCThreadEvent_SetEvent(pstChannel->hEvent);
	}
	UCThreadMutex_Unlock(pstChannel->hMutex);
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

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKSharedMemoryChannel_ReadFromBuffer(SChannel *pstChannel, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = UCThreadMutex_Lock(pstChannel->hMutex);
	ERRIFGOTO(result, _EXIT);

	if(pstChannel->nDataLen < nDataToRead)
	{
		if(pstChannel->nDataLen > 0)
		{
			UC_memcpy(pBuffer, pstChannel->pDataStart, pstChannel->nDataLen);
		}

		*pnDataRead = pstChannel->nDataLen;
	}
	else // pstChannel->nDataLen >= nDataToRead
	{
		UC_memcpy(pBuffer, pstChannel->pDataStart, nDataToRead);

		*pnDataRead = nDataToRead;
	}

	result = ERR_UEM_NOERROR;

	UCThreadMutex_Unlock(pstChannel->hMutex);
_EXIT:
	return result;
}

static uem_result writeToGeneralQueue(SChannel *pstChannel, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nCurrentSampleRateIndex = 0;
	int nExpectedConsumeSize = 0;

	result = UCThreadMutex_Lock(pstChannel->hMutex);
	ERRIFGOTO(result, _EXIT);

	pstChannel->nReferenceCount++;

	// TODO: Error check out exit logic needed
	while(pstChannel->nDataLen + nDataToWrite > pstChannel->nBufSize)
	{
		result = UCThreadMutex_Unlock(pstChannel->hMutex);
		ERRIFGOTO(result, _EXIT);

		result = UCThreadEvent_WaitEvent(pstChannel->hEvent);
		ERRIFGOTO(result, _EXIT);

		result = UCThreadMutex_Lock(pstChannel->hMutex);
		ERRIFGOTO(result, _EXIT);
	}

	result = copyAndMovePointerToRoundedQueue(pstChannel, pBuffer, nDataToWrite, nChunkIndex);
	ERRIFGOTO(result, _EXIT_LOCK);

	if(pstChannel->enChannelType == CHANNEL_TYPE_INPUT_ARRAY && pstChannel->pstAvailableInputChunkHead == NULL)
	{
		nCurrentSampleRateIndex = pstChannel->stInputPort.nCurrentSampleRateIndex;
		nExpectedConsumeSize = pstChannel->stInputPort.astSampleRates[nCurrentSampleRateIndex].nSampleRate * pstChannel->stInputPort.nSampleSize;

		if(nExpectedConsumeSize < pstChannel->nDataLen)
		{
			result = makeAvailableInputChunkList(pstChannel);
			ERRIFGOTO(result, _EXIT_LOCK);
		}
	}

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
	pstChannel->nReferenceCount--;

	if(pstChannel->nReferenceCount > 0)
	{
		UCThreadEvent_SetEvent(pstChannel->hEvent);
	}
	UCThreadMutex_Unlock(pstChannel->hMutex);
_EXIT:
	return result;
}


static uem_result clearOutputChunkInfo(SChannel *pstChannel)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;
	void *pNextChunkDataEnd = NULL;
	int nSegmentLen = 0;

	for(nLoop = 0 ; nLoop <= pstChannel->stOutputPortChunk.nChunkNum ; nLoop++)
	{
		// if(pstChannel->stOutputPortChunk.astChunk[nLoop].pDataEnd )
		pNextChunkDataEnd = pstChannel->pDataEnd + pstChannel->stOutputPortChunk.nChunkLen * nLoop;

		if(pNextChunkDataEnd >= pstChannel->pBuffer + pstChannel->nBufSize)
		{
			nSegmentLen = pNextChunkDataEnd - pstChannel->pBuffer + pstChannel->nBufSize;
			pstChannel->stOutputPortChunk.astChunk[nLoop].pDataEnd = pstChannel->pBuffer + nSegmentLen;
		}
		else
		{
			pstChannel->stOutputPortChunk.astChunk[nLoop].pDataEnd = pNextChunkDataEnd;
		}

		// maximum available number is not needed for output chunk
		pstChannel->stOutputPortChunk.astChunk[nLoop].nAvailableDataNum = 1;
	}

	pstChannel->nWrittenOutputChunkNum = 0;

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

	result = UCThreadMutex_Lock(pstChannel->hMutex);
	ERRIFGOTO(result, _EXIT);

	pstChannel->nReferenceCount++;

	// pstChannel->nWrittenOutputChunkNum >= 0means it uses output chunk
	if(pstChannel->nWrittenOutputChunkNum >= 0 && pstChannel->stOutputPortChunk.nChunkLen != nDataToWrite)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT_LOCK);
	}

	nCurrentSampleRateIndex = pstChannel->stOutputPort.nCurrentSampleRateIndex;
	nExpectedProduceSize = pstChannel->stOutputPort.astSampleRates[nCurrentSampleRateIndex].nSampleRate * pstChannel->stOutputPort.nSampleSize;

	// TODO: Error check out exit logic needed
	while(pstChannel->nDataLen + nExpectedProduceSize > pstChannel->nBufSize || // nBuffer is full or
		(pstChannel->nWrittenOutputChunkNum > 0 && pstChannel->nWrittenOutputChunkNum < pstChannel->stOutputPortChunk.nChunkNum &&
		pstChannel->stOutputPortChunk.astChunk[nChunkIndex].nChunkDataLen > 0)) // current chunk index is already filled with data
	{
		result = UCThreadMutex_Unlock(pstChannel->hMutex);
		ERRIFGOTO(result, _EXIT);

		result = UCThreadEvent_WaitEvent(pstChannel->hEvent);
		ERRIFGOTO(result, _EXIT);

		result = UCThreadMutex_Lock(pstChannel->hMutex);
		ERRIFGOTO(result, _EXIT);
	}

	// Chunk needs to be initialized
	if(pstChannel->nWrittenOutputChunkNum < 0 || pstChannel->nWrittenOutputChunkNum >= pstChannel->stOutputPortChunk.nChunkNum)
	{
		result = clearOutputChunkInfo(pstChannel);
		ERRIFGOTO(result, _EXIT_LOCK);
	}

	result = copyToRoundedChunk(pstChannel, pBuffer, nDataToWrite, nChunkIndex);
	ERRIFGOTO(result, _EXIT_LOCK);

	pstChannel->stOutputPortChunk.astChunk[nChunkIndex].nChunkDataLen = nDataToWrite;

	pstChannel->nWrittenOutputChunkNum++;

	if(pstChannel->nWrittenOutputChunkNum == pstChannel->stOutputPortChunk.nChunkNum)
	{
		pstChannel->pDataEnd = pstChannel->pDataEnd + nExpectedProduceSize;
		pstChannel->nDataLen += nExpectedProduceSize;
	}

	nCurrentReadSampleRateIndex = pstChannel->stInputPort.nCurrentSampleRateIndex;
	nExpectedConsumeSize = pstChannel->stInputPort.astSampleRates[nCurrentReadSampleRateIndex].nSampleRate * pstChannel->stInputPort.nSampleSize;

	if(nExpectedConsumeSize <= pstChannel->nDataLen)
	{
		result = makeAvailableInputChunkList(pstChannel);
		ERRIFGOTO(result, _EXIT_LOCK);
	}

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
	pstChannel->nReferenceCount--;

	if(pstChannel->nReferenceCount > 0)
	{
		UCThreadEvent_SetEvent(pstChannel->hEvent);
	}
	UCThreadMutex_Unlock(pstChannel->hMutex);
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

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKSharedMemoryChannel_WriteToBuffer (SChannel *pstChannel, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = UCThreadMutex_Lock(pstChannel->hMutex);
	ERRIFGOTO(result, _EXIT);

	if(pstChannel->nBufSize >= nDataToWrite)
	{
		if (nDataToWrite > 0)
		{
			UC_memcpy(pstChannel->pDataStart, pBuffer, nDataToWrite);
		}

		*pnDataWritten = nDataToWrite;
	}
	else // pstChannel->nBufSize < nDataToWrite
	{
		UC_memcpy(pstChannel->pDataStart, pBuffer, pstChannel->nBufSize);

		*pnDataWritten = pstChannel->nBufSize;
	}

	result = ERR_UEM_NOERROR;

	UCThreadMutex_Unlock(pstChannel->hMutex);
_EXIT:
	return result;
}


uem_result UKSharedMemoryChannel_GetAvailableChunk (SChannel *pstChannel, OUT int *pnChunkIndex)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nAvailableIndex = 0;
	SAvailableChunk *pstNextChunk = NULL;
	SAvailableChunk *pstCurChunk = NULL;

	result = UCThreadMutex_Lock(pstChannel->hMutex);
	ERRIFGOTO(result, _EXIT);

	pstChannel->nReferenceCount++;

	while(pstChannel->pstAvailableInputChunkHead == NULL)
	{
		result = UCThreadMutex_Unlock(pstChannel->hMutex);
		ERRIFGOTO(result, _EXIT);

		result = UCThreadEvent_WaitEvent(pstChannel->hEvent);
		ERRIFGOTO(result, _EXIT);

		result = UCThreadMutex_Lock(pstChannel->hMutex);
		ERRIFGOTO(result, _EXIT);
	}

	nAvailableIndex = pstChannel->pstAvailableInputChunkHead->nChunkIndex;
	pstCurChunk = pstChannel->pstAvailableInputChunkHead;

	pstChannel->pstAvailableInputChunkTail->pstNext = pstCurChunk;
	pstCurChunk->pstPrev = pstChannel->pstAvailableInputChunkTail;

	pstNextChunk = pstCurChunk->pstNext;
	pstNextChunk->pstPrev = NULL;

	pstChannel->pstAvailableInputChunkTail = pstCurChunk;
	pstChannel->pstAvailableInputChunkTail->pstNext = NULL;

	pstChannel->pstAvailableInputChunkHead = pstNextChunk;

	*pnChunkIndex = nAvailableIndex;

	result = ERR_UEM_NOERROR;
//_EXIT_LOCK:
	pstChannel->nReferenceCount--;

	if(pstChannel->nReferenceCount > 0)
	{
		UCThreadEvent_SetEvent(pstChannel->hEvent);
	}
	UCThreadMutex_Unlock(pstChannel->hMutex);
_EXIT:
	return result;
}

uem_result UKSharedMemoryChannel_GetNumOfAvailableData (SChannel *pstChannel, IN int nChunkIndex, OUT int *pnDataNum)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = UCThreadMutex_Lock(pstChannel->hMutex);
	ERRIFGOTO(result, _EXIT);

	if(pstChannel->enChannelType == CHANNEL_TYPE_GENERAL || pstChannel->enChannelType == CHANNEL_TYPE_OUTPUT_ARRAY)
	{
		*pnDataNum = pstChannel->nDataLen;
	}
	else // if( pstChannel->enChannelType == CHANNEL_TYPE_INPUT_ARRAY || pstChannel->enChannelType == CHANNEL_TYPE_FULL_ARRAY)
	{
		if(pstChannel->pstAvailableInputChunkHead != NULL)
		{
			*pnDataNum = pstChannel->stInputPortChunk.astChunk[nChunkIndex].nChunkDataLen;
		}
		else
		{
			*pnDataNum = 0;
		}
	}

	result = ERR_UEM_NOERROR;

	UCThreadMutex_Unlock(pstChannel->hMutex);
_EXIT:
	return result;
}


uem_result UKSharedMemoryChannel_Finalize(SChannel *pstChannel)
{
	uem_result result = ERR_UEM_UNKNOWN;

	// ignore error
	UCThreadMutex_Destroy(&(pstChannel->hMutex));

	// ignore
	UCThreadEvent_Destroy(&(pstChannel->hEvent));

	result = ERR_UEM_NOERROR;

	return result;
}






