/*
 * UKChannelMemory.c
 *
 *  Created on: 2018. 5. 26.
 *      Author: jej
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <UCBasic.h>
#include <UCAlloc.h>
#include <UCThreadMutex.h>

#include <uem_data.h>

#include <UKTask_internal.h>
#include <UKChannel_internal.h>

static uem_result setChunkNumAndLen(SPort *pstPort, SChunkInfo *pstChunkInfo)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;

	result = UKChannel_GetChunkNumAndLen(pstPort, &(pstChunkInfo->nChunkNum), &(pstChunkInfo->nChunkLen));
	ERRIFGOTO(result, _EXIT);

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

uem_result UKChannelMemory_Clear(SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;

	IFVARERRASSIGNGOTO(pstSharedMemoryChannel, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	// information clear
	// pDataStart => clear (pBuffer)
	// nDataLen => clear (0)
	pstSharedMemoryChannel->pDataStart = pstSharedMemoryChannel->pBuffer;
	pstSharedMemoryChannel->pDataEnd = pstSharedMemoryChannel->pBuffer;
	pstSharedMemoryChannel->nDataLen = 0;
	pstSharedMemoryChannel->nReadReferenceCount = 0;
	pstSharedMemoryChannel->nWriteReferenceCount = 0;

	// the chunk num and chunk length is dependent to sample rate of mode transition
	result = setChunkNumAndLen(pstChannel->pstInputPort, &(pstSharedMemoryChannel->stInputPortChunk));
	ERRIFGOTO(result, _EXIT);

	result = setChunkNumAndLen(pstChannel->pstOutputPort, &(pstSharedMemoryChannel->stOutputPortChunk));
	ERRIFGOTO(result, _EXIT);

	pstSharedMemoryChannel->nWrittenOutputChunkNum = CHUNK_NUM_NOT_INITIALIZED;

	pstSharedMemoryChannel->pstAvailableInputChunkHead = NULL;
	pstSharedMemoryChannel->pstAvailableInputChunkTail = NULL;

	for(nLoop = 0 ; nLoop < pstSharedMemoryChannel->nMaxChunkNum ; nLoop++)
	{
		pstSharedMemoryChannel->astAvailableInputChunkList[nLoop].pstPrev = NULL;
		pstSharedMemoryChannel->astAvailableInputChunkList[nLoop].pstNext = NULL;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKChannelMemory_Initialize(SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0 ;

	IFVARERRASSIGNGOTO(pstSharedMemoryChannel, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	// initialize buffer
	// If not set, initialize those things, // pBuffer => is NULL => alloc
	if(pstSharedMemoryChannel->bStaticAllocation == FALSE && pstSharedMemoryChannel->pBuffer == NULL)
	{
		result = pstSharedMemoryChannel->pstMemoryAccessAPI->fnCreateMemory(pstChannel->nBufSize, pstChannel->nProcessorId, &(pstSharedMemoryChannel->pBuffer));
		ERRIFGOTO(result, _EXIT);
	}

	// hMutex => initialize/create
	result = UCThreadMutex_Create(&(pstSharedMemoryChannel->hMutex));
	ERRIFGOTO(result, _EXIT);

	// hEvent => initialize/create
	result = UCThreadEvent_Create(&(pstSharedMemoryChannel->hReadEvent));
	ERRIFGOTO(result, _EXIT);
	result = UCThreadEvent_Create(&(pstSharedMemoryChannel->hWriteEvent));
	ERRIFGOTO(result, _EXIT);

	for(nLoop = 0 ; nLoop < pstSharedMemoryChannel->nOutputMaxChunkNum ; nLoop++)
	{
		pstSharedMemoryChannel->stOutputPortChunk.astChunk[nLoop].pChunkStart = NULL;
		pstSharedMemoryChannel->stOutputPortChunk.astChunk[nLoop].pDataStart = NULL;
		pstSharedMemoryChannel->stOutputPortChunk.astChunk[nLoop].pDataEnd = NULL;
		pstSharedMemoryChannel->stOutputPortChunk.astChunk[nLoop].nAvailableDataNum = 0;
		pstSharedMemoryChannel->stOutputPortChunk.astChunk[nLoop].nChunkDataLen = 0;
	}

	for(nLoop = 0 ; nLoop < pstSharedMemoryChannel->nMaxChunkNum ; nLoop++)
	{
		pstSharedMemoryChannel->stInputPortChunk.astChunk[nLoop].pChunkStart = NULL;
		pstSharedMemoryChannel->stInputPortChunk.astChunk[nLoop].pDataStart = NULL;
		pstSharedMemoryChannel->stInputPortChunk.astChunk[nLoop].pDataEnd = NULL;
		pstSharedMemoryChannel->stInputPortChunk.astChunk[nLoop].nAvailableDataNum = 0;
		pstSharedMemoryChannel->stInputPortChunk.astChunk[nLoop].nChunkDataLen = 0;

		pstSharedMemoryChannel->astAvailableInputChunkList[nLoop].nChunkIndex = nLoop;
		pstSharedMemoryChannel->astAvailableInputChunkList[nLoop].nSampleNum = 0;
		pstSharedMemoryChannel->astAvailableInputChunkList[nLoop].pstNext = NULL;
		pstSharedMemoryChannel->astAvailableInputChunkList[nLoop].pstPrev = NULL;
	}

	result = UKChannelMemory_Clear(pstChannel, pstSharedMemoryChannel);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static uem_result removeChunkFromAvailableChunkList(SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel, int nChunkIndex)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SAvailableChunk *pstAvailableChunk = NULL;
	SAvailableChunk *pstPreviousChunk = NULL;
	SAvailableChunk *pstNextChunk = NULL;

	pstAvailableChunk = pstSharedMemoryChannel->pstAvailableInputChunkHead;

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
				pstSharedMemoryChannel->pstAvailableInputChunkHead = pstAvailableChunk->pstNext;
			}

			if(pstNextChunk != NULL)
			{
				pstNextChunk->pstPrev = pstPreviousChunk;
			}
			else // pstNextChunk == NULL => pstAvailableChunk is a tail chunk
			{
				pstSharedMemoryChannel->pstAvailableInputChunkTail = pstAvailableChunk->pstPrev;
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

static uem_result copyAndMovePointerFromRoundedQueue(unsigned char *pDest, SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel, int nDataToRead)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nSegmentLen = 0;
	int nRemainderLen = 0;
	SGenericMemoryAccess *pstMemoryAPI = NULL;

	pstMemoryAPI = pstSharedMemoryChannel->pstMemoryAccessAPI;

	// data is divided into two segments because the start point of the chunk is close to the end of buffer range
	if(pstSharedMemoryChannel->pDataStart + nDataToRead > pstSharedMemoryChannel->pBuffer + pstChannel->nBufSize)
	{
		nSegmentLen = pstSharedMemoryChannel->pBuffer + pstChannel->nBufSize - pstSharedMemoryChannel->pDataStart;
		nRemainderLen = nDataToRead - nSegmentLen;

		if(pDest != NULL)
		{
			result = pstMemoryAPI->fnCopyFromMemory(pDest, pstSharedMemoryChannel->pDataStart, nSegmentLen);
			ERRIFGOTO(result, _EXIT);
			result = pstMemoryAPI->fnCopyFromMemory(pDest + nSegmentLen, pstSharedMemoryChannel->pBuffer, nRemainderLen);
			ERRIFGOTO(result, _EXIT);
		}

		pstSharedMemoryChannel->pDataStart = pstSharedMemoryChannel->pBuffer + nRemainderLen;
	}
	else
	{
		if(pDest != NULL)
		{
			result = pstMemoryAPI->fnCopyFromMemory(pDest, pstSharedMemoryChannel->pDataStart, nDataToRead);
			ERRIFGOTO(result, _EXIT);
		}

		if(pstSharedMemoryChannel->pDataStart + nDataToRead == pstSharedMemoryChannel->pBuffer + pstChannel->nBufSize)
		{
			pstSharedMemoryChannel->pDataStart = pstSharedMemoryChannel->pBuffer;
		}
		else
		{
			pstSharedMemoryChannel->pDataStart += nDataToRead;
		}
	}

	pstSharedMemoryChannel->nDataLen -= nDataToRead;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static uem_result copyToRoundedChunk(SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel, unsigned char *pSrc, int nDataToWrite, int nChunkIndex)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SChunk *pstDestChunk = NULL;
	int nSegmentLen = 0;
	int nRemainderLen = 0;
	SGenericMemoryAccess *pstMemoryAPI = NULL;

	pstMemoryAPI = pstSharedMemoryChannel->pstMemoryAccessAPI;

	pstDestChunk = &(pstSharedMemoryChannel->stOutputPortChunk.astChunk[nChunkIndex]);

	// Chunk is divided into two segments because the start point of the chunk is close to the end of buffer range
	if(pstDestChunk->pDataEnd + nDataToWrite > pstSharedMemoryChannel->pBuffer + pstChannel->nBufSize)
	{
		nSegmentLen = pstSharedMemoryChannel->pBuffer + pstChannel->nBufSize - pstDestChunk->pDataEnd;
		nRemainderLen = nDataToWrite - nSegmentLen;

		result = pstMemoryAPI->fnCopyToMemory(pstDestChunk->pDataEnd, pSrc, nSegmentLen);
		ERRIFGOTO(result, _EXIT);
		result = pstMemoryAPI->fnCopyToMemory(pstSharedMemoryChannel->pBuffer, pSrc + nSegmentLen, nRemainderLen);
		ERRIFGOTO(result, _EXIT);

		pstDestChunk->pDataEnd = pstSharedMemoryChannel->pBuffer + nDataToWrite - nSegmentLen;
	}
	else
	{
		result = pstMemoryAPI->fnCopyToMemory(pstDestChunk->pDataEnd, pSrc, nDataToWrite);
		ERRIFGOTO(result, _EXIT);

		if(pstDestChunk->pDataEnd == pstSharedMemoryChannel->pBuffer + pstChannel->nBufSize)
		{
			pstDestChunk->pDataEnd = pstSharedMemoryChannel->pBuffer;
		}
		else
		{
			pstDestChunk->pDataEnd += nDataToWrite;
		}
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result copyAndMovePointerToRoundedQueue(SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel,
				unsigned char *pSrc, int nDataToWrite, int nChunkIndex, FnCopyMemory fnCopyFunc)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nSegmentLen = 0;
	int nRemainderLen = 0;

	// Chunk is divided into two segments because the start point of the chunk is close to the end of buffer range
	if(pstSharedMemoryChannel->pDataEnd + nDataToWrite > pstSharedMemoryChannel->pBuffer + pstChannel->nBufSize)
	{
		nSegmentLen = pstSharedMemoryChannel->pBuffer + pstChannel->nBufSize - pstSharedMemoryChannel->pDataEnd;
		nRemainderLen = nDataToWrite - nSegmentLen;

		result = fnCopyFunc(pstSharedMemoryChannel->pDataEnd, pSrc, nSegmentLen);
		ERRIFGOTO(result, _EXIT);
		result = fnCopyFunc(pstSharedMemoryChannel->pBuffer, pSrc + nSegmentLen, nRemainderLen);
		ERRIFGOTO(result, _EXIT);

		pstSharedMemoryChannel->pDataEnd = pstSharedMemoryChannel->pBuffer + nDataToWrite - nSegmentLen;
	}
	else
	{
		result = fnCopyFunc(pstSharedMemoryChannel->pDataEnd, pSrc, nDataToWrite);
		ERRIFGOTO(result, _EXIT);

		if(pstSharedMemoryChannel->pDataEnd == pstSharedMemoryChannel->pBuffer + pstChannel->nBufSize)
		{
			pstSharedMemoryChannel->pDataEnd = pstSharedMemoryChannel->pBuffer;
		}
		else
		{
			pstSharedMemoryChannel->pDataEnd += nDataToWrite;
		}
	}

	pstSharedMemoryChannel->nDataLen += nDataToWrite;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result copyFromRoundedChunk(unsigned char *pDest, SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel, int nDataToRead, int nChunkIndex)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SChunk *pstSrcChunk = NULL;
	int nSegmentLen = 0;
	int nRemainderLen = 0;
	SGenericMemoryAccess *pstMemoryAPI = NULL;

	pstMemoryAPI = pstSharedMemoryChannel->pstMemoryAccessAPI;

	pstSrcChunk = &(pstSharedMemoryChannel->stInputPortChunk.astChunk[nChunkIndex]);

	// Chunk is divided into two segments because the start point of the chunk is close to the end of buffer range
	if(pstSrcChunk->pDataStart + nDataToRead > pstSharedMemoryChannel->pBuffer + pstChannel->nBufSize)
	{
		nSegmentLen = pstSharedMemoryChannel->pBuffer + pstChannel->nBufSize - pstSrcChunk->pDataStart;
		nRemainderLen = nDataToRead - nSegmentLen;

		if(pDest != NULL)
		{
			result = pstMemoryAPI->fnCopyFromMemory(pDest, pstSrcChunk->pDataStart, nSegmentLen);
			ERRIFGOTO(result, _EXIT);
			result = pstMemoryAPI->fnCopyFromMemory(pDest + nSegmentLen, pstSharedMemoryChannel->pBuffer, nRemainderLen);
			ERRIFGOTO(result, _EXIT);
		}
	}
	else
	{
		if(pDest != NULL)
		{
			result = pstMemoryAPI->fnCopyFromMemory(pDest, pstSrcChunk->pDataStart, nDataToRead);
			ERRIFGOTO(result, _EXIT);
		}
	}

	result = ERR_UEM_NOERROR;
_EXIT:
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


static uem_result setInputChunks(SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;
	int nMaxAvailableNum = 0;

	for(nLoop = 0 ; nLoop < pstSharedMemoryChannel->stInputPortChunk.nChunkNum ; nLoop++)
	{
		result = getMaximumAvailableNum(pstChannel->pstInputPort, &nMaxAvailableNum);
		ERRIFGOTO(result, _EXIT);
		pstSharedMemoryChannel->stInputPortChunk.astChunk[nLoop].nAvailableDataNum = nMaxAvailableNum;
		pstSharedMemoryChannel->stInputPortChunk.astChunk[nLoop].nChunkDataLen = pstSharedMemoryChannel->stInputPortChunk.nChunkLen;
		if(pstSharedMemoryChannel->pBuffer + pstChannel->nBufSize <= pstSharedMemoryChannel->pDataStart + pstSharedMemoryChannel->stInputPortChunk.nChunkLen * nLoop)
		{
			int nRemainderLen = pstSharedMemoryChannel->pDataStart + pstSharedMemoryChannel->stInputPortChunk.nChunkLen * nLoop - (pstSharedMemoryChannel->pBuffer + pstChannel->nBufSize);

			pstSharedMemoryChannel->stInputPortChunk.astChunk[nLoop].pDataStart = pstSharedMemoryChannel->pBuffer + nRemainderLen;
		}
		else
		{
			pstSharedMemoryChannel->stInputPortChunk.astChunk[nLoop].pDataStart = pstSharedMemoryChannel->pDataStart + pstSharedMemoryChannel->stInputPortChunk.nChunkLen * nLoop;
		}
		pstSharedMemoryChannel->stInputPortChunk.astChunk[nLoop].pDataEnd = NULL;
		pstSharedMemoryChannel->stInputPortChunk.astChunk[nLoop].pChunkStart = NULL;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result makeAvailableInputChunkList(SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nChunkNum = 0;
	int nLoop = 0;

	nChunkNum = pstSharedMemoryChannel->stInputPortChunk.nChunkNum;

	for(nLoop = 0; nLoop < nChunkNum ; nLoop++)
	{
		if(nLoop == 0)
		{
			pstSharedMemoryChannel->pstAvailableInputChunkHead = &(pstSharedMemoryChannel->astAvailableInputChunkList[nLoop]);
			pstSharedMemoryChannel->astAvailableInputChunkList[nLoop].pstPrev = NULL;
		}
		else
		{
			pstSharedMemoryChannel->astAvailableInputChunkList[nLoop].pstPrev = &(pstSharedMemoryChannel->astAvailableInputChunkList[nLoop-1]);
		}

		if(nLoop == nChunkNum - 1)
		{
			pstSharedMemoryChannel->pstAvailableInputChunkTail = &(pstSharedMemoryChannel->astAvailableInputChunkList[nLoop]);
			pstSharedMemoryChannel->astAvailableInputChunkList[nLoop].pstNext = NULL;
		}
		else
		{
			pstSharedMemoryChannel->astAvailableInputChunkList[nLoop].pstNext = &(pstSharedMemoryChannel->astAvailableInputChunkList[nLoop+1]);
		}
	}

	//UEM_DEBUG_PRINT("available chunk create: %d : %d : %p : %d\n", pstChannel->nChannelIndex, nChunkNum, pstChannel->pstAvailableInputChunkHead, pstChannel->pstAvailableInputChunkHead->nChunkIndex);

	result = setInputChunks(pstChannel, pstSharedMemoryChannel);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static uem_result moveDataPointerOfArrayQueue(SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel)
{
	uem_result result = ERR_UEM_UNKNOWN;

	pstSharedMemoryChannel->nDataLen = pstSharedMemoryChannel->nDataLen - (pstSharedMemoryChannel->stInputPortChunk.nChunkNum * pstSharedMemoryChannel->stInputPortChunk.nChunkLen);

	// if the next pDataStart needs to be moved to the front pointer
	if(pstSharedMemoryChannel->pBuffer + pstChannel->nBufSize < pstSharedMemoryChannel->pDataStart + pstSharedMemoryChannel->stInputPortChunk.nChunkNum * pstSharedMemoryChannel->stInputPortChunk.nChunkLen)
	{
		pstSharedMemoryChannel->pDataStart = pstSharedMemoryChannel->pBuffer +
					(pstSharedMemoryChannel->stInputPortChunk.nChunkNum * pstSharedMemoryChannel->stInputPortChunk.nChunkLen) -
					(pstSharedMemoryChannel->pBuffer + pstChannel->nBufSize - pstSharedMemoryChannel->pDataStart);
	}
	else
	{
		pstSharedMemoryChannel->pDataStart = pstSharedMemoryChannel->pDataStart + pstSharedMemoryChannel->stInputPortChunk.nChunkNum * pstSharedMemoryChannel->stInputPortChunk.nChunkLen;

		if(pstSharedMemoryChannel->pDataStart == pstSharedMemoryChannel->pBuffer + pstChannel->nBufSize)
		{
			pstSharedMemoryChannel->pDataStart = pstSharedMemoryChannel->pBuffer;
		}
	}

	result = ERR_UEM_NOERROR;

	return result;
}


static uem_result readFromArrayQueue(SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SChunk *pstTargetChunk = NULL;

	result = UCThreadMutex_Lock(pstSharedMemoryChannel->hMutex);
	ERRIFGOTO(result, _EXIT);

	pstSharedMemoryChannel->nReadReferenceCount++;

	while(pstSharedMemoryChannel->stInputPortChunk.astChunk[nChunkIndex].nAvailableDataNum == 0 ||
		pstSharedMemoryChannel->bInitialDataUpdated == FALSE)
	{
		if(pstSharedMemoryChannel->bReadExit == TRUE)
		{
			UEMASSIGNGOTO(result, ERR_UEM_SUSPEND, _EXIT_LOCK);
		}

		if(pstSharedMemoryChannel->nReadReferenceCount > 1 &&
				pstSharedMemoryChannel->pstAvailableInputChunkHead != NULL)
		{
			UCThreadEvent_SetEvent(pstSharedMemoryChannel->hReadEvent);
		}

		result = UCThreadMutex_Unlock(pstSharedMemoryChannel->hMutex);
		ERRIFGOTO(result, _EXIT);

		result = UCThreadEvent_WaitEvent(pstSharedMemoryChannel->hReadEvent);
		ERRIFGOTO(result, _EXIT);

		result = UCThreadMutex_Lock(pstSharedMemoryChannel->hMutex);
		ERRIFGOTO(result, _EXIT);
	}

	pstTargetChunk = &(pstSharedMemoryChannel->stInputPortChunk.astChunk[nChunkIndex]);

	if(nDataToRead != pstTargetChunk->nChunkDataLen)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT_LOCK);
	}

	result = copyFromRoundedChunk(pBuffer, pstChannel, pstSharedMemoryChannel, nDataToRead, nChunkIndex);
	ERRIFGOTO(result, _EXIT_LOCK);

	pstTargetChunk->nAvailableDataNum--;

	if(pstTargetChunk->nAvailableDataNum == 0)
	{
		pstSharedMemoryChannel->stInputPortChunk.astChunk[nChunkIndex].nChunkDataLen = 0;

		result = removeChunkFromAvailableChunkList(pstChannel, pstSharedMemoryChannel, nChunkIndex);
		ERRIFGOTO(result, _EXIT_LOCK);

		// All chunk list removed?
		if(pstSharedMemoryChannel->pstAvailableInputChunkHead == NULL)
		{
			result = moveDataPointerOfArrayQueue(pstChannel, pstSharedMemoryChannel);
			ERRIFGOTO(result, _EXIT_LOCK);

			if(pstSharedMemoryChannel->nDataLen > 0)
			{
				int nCurrentSampleRateIndex = 0;
				int nExpectedConsumeSize = 0;

				nCurrentSampleRateIndex = pstChannel->pstInputPort->nCurrentSampleRateIndex;
				nExpectedConsumeSize = pstChannel->pstInputPort->astSampleRates[nCurrentSampleRateIndex].nSampleRate * pstChannel->pstInputPort->nSampleSize;

				if(nExpectedConsumeSize <= pstSharedMemoryChannel->nDataLen)
				{
					result = makeAvailableInputChunkList(pstChannel, pstSharedMemoryChannel);
					ERRIFGOTO(result, _EXIT_LOCK);
				}
			}
		}
	}

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
	pstSharedMemoryChannel->nReadReferenceCount--;

	if(pstSharedMemoryChannel->nWriteReferenceCount > 0)
	{
		UCThreadEvent_SetEvent(pstSharedMemoryChannel->hWriteEvent);
	}

	if(pstSharedMemoryChannel->nReadReferenceCount > 0 &&
			pstSharedMemoryChannel->pstAvailableInputChunkHead != NULL)
	{
		UCThreadEvent_SetEvent(pstSharedMemoryChannel->hReadEvent);
	}

	UCThreadMutex_Unlock(pstSharedMemoryChannel->hMutex);
_EXIT:
	return result;
}

static uem_result readFromGeneralQueue(SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = UCThreadMutex_Lock(pstSharedMemoryChannel->hMutex);
	ERRIFGOTO(result, _EXIT);

	pstSharedMemoryChannel->nReadReferenceCount++;

	// TODO: Error check out exit logic needed
	while(pstSharedMemoryChannel->nDataLen < nDataToRead || pstSharedMemoryChannel->bInitialDataUpdated == FALSE)
	{
		if(pstSharedMemoryChannel->bReadExit == TRUE)
		{
			UEMASSIGNGOTO(result, ERR_UEM_SUSPEND, _EXIT_LOCK);
		}

		result = UCThreadMutex_Unlock(pstSharedMemoryChannel->hMutex);
		ERRIFGOTO(result, _EXIT);

		result = UCThreadEvent_WaitEvent(pstSharedMemoryChannel->hReadEvent);
		ERRIFGOTO(result, _EXIT);

		result = UCThreadMutex_Lock(pstSharedMemoryChannel->hMutex);
		ERRIFGOTO(result, _EXIT);
	}

	result = copyAndMovePointerFromRoundedQueue(pBuffer, pstChannel, pstSharedMemoryChannel, nDataToRead);
	ERRIFGOTO(result, _EXIT_LOCK);

	*pnDataRead = nDataToRead;

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
	pstSharedMemoryChannel->nReadReferenceCount--;

	if(pstSharedMemoryChannel->nWriteReferenceCount > 0)
	{
		UCThreadEvent_SetEvent(pstSharedMemoryChannel->hWriteEvent);
	}
	UCThreadMutex_Unlock(pstSharedMemoryChannel->hMutex);
_EXIT:
	return result;
}


uem_result UKChannelMemory_ReadFromQueue(SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead)
{
	uem_result result = ERR_UEM_UNKNOWN;

	IFVARERRASSIGNGOTO(pstSharedMemoryChannel, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	// array channel
	if(pstChannel->enChannelType == CHANNEL_TYPE_INPUT_ARRAY || pstChannel->enChannelType == CHANNEL_TYPE_FULL_ARRAY)
	{
		result = readFromArrayQueue(pstChannel, pstSharedMemoryChannel, pBuffer, nDataToRead, nChunkIndex, pnDataRead);
		ERRIFGOTO(result, _EXIT);
	}
	else // general channel
	{
		result = readFromGeneralQueue(pstChannel, pstSharedMemoryChannel, pBuffer, nDataToRead, nChunkIndex, pnDataRead);
		ERRIFGOTO(result, _EXIT);
	}

	// to preserve, ERR_UEM_SUSPEND, do not set ERR_UEM_NOERROR here
_EXIT:
	return result;
}

uem_result UKChannelMemory_ReadFromBuffer(SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SGenericMemoryAccess *pstMemoryAPI = NULL;

	IFVARERRASSIGNGOTO(pstSharedMemoryChannel, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	pstMemoryAPI = pstSharedMemoryChannel->pstMemoryAccessAPI;

	result = UCThreadMutex_Lock(pstSharedMemoryChannel->hMutex);
	ERRIFGOTO(result, _EXIT);

	if(pstSharedMemoryChannel->nDataLen < nDataToRead)
	{
		if(pstSharedMemoryChannel->nDataLen > 0)
		{
			result = pstMemoryAPI->fnCopyFromMemory(pBuffer, pstSharedMemoryChannel->pDataStart, pstSharedMemoryChannel->nDataLen);
			ERRIFGOTO(result, _EXIT_LOCK);
		}

		*pnDataRead = pstSharedMemoryChannel->nDataLen;
	}
	else // pstChannel->nDataLen >= nDataToRead
	{
		result = pstMemoryAPI->fnCopyFromMemory(pBuffer, pstSharedMemoryChannel->pDataStart, nDataToRead);
		ERRIFGOTO(result, _EXIT_LOCK);

		*pnDataRead = nDataToRead;
	}

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
	UCThreadMutex_Unlock(pstSharedMemoryChannel->hMutex);
_EXIT:
	return result;
}

static uem_result writeToGeneralQueue(SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nCurrentSampleRateIndex = 0;
	int nExpectedConsumeSize = 0;

	result = UCThreadMutex_Lock(pstSharedMemoryChannel->hMutex);
	ERRIFGOTO(result, _EXIT);

	pstSharedMemoryChannel->nWriteReferenceCount++;

	while(pstSharedMemoryChannel->nDataLen + nDataToWrite > pstChannel->nBufSize)
	{
		if(pstSharedMemoryChannel->bWriteExit == TRUE)
		{
			UEMASSIGNGOTO(result, ERR_UEM_SUSPEND, _EXIT_LOCK);
		}

		result = UCThreadMutex_Unlock(pstSharedMemoryChannel->hMutex);
		ERRIFGOTO(result, _EXIT);

		result = UCThreadEvent_WaitEvent(pstSharedMemoryChannel->hWriteEvent);
		ERRIFGOTO(result, _EXIT);

		result = UCThreadMutex_Lock(pstSharedMemoryChannel->hMutex);
		ERRIFGOTO(result, _EXIT);
	}

	result = copyAndMovePointerToRoundedQueue(pstChannel, pstSharedMemoryChannel, pBuffer, nDataToWrite, nChunkIndex,
											pstSharedMemoryChannel->pstMemoryAccessAPI->fnCopyToMemory);
	ERRIFGOTO(result, _EXIT_LOCK);

	if(pstChannel->enChannelType == CHANNEL_TYPE_INPUT_ARRAY && pstSharedMemoryChannel->pstAvailableInputChunkHead == NULL)
	{
		nCurrentSampleRateIndex = pstChannel->pstInputPort->nCurrentSampleRateIndex;
		nExpectedConsumeSize = pstChannel->pstInputPort->astSampleRates[nCurrentSampleRateIndex].nSampleRate * pstChannel->pstInputPort->nSampleSize;

		if(nExpectedConsumeSize <= pstSharedMemoryChannel->nDataLen)
		{
			result = makeAvailableInputChunkList(pstChannel, pstSharedMemoryChannel);
			ERRIFGOTO(result, _EXIT_LOCK);
		}
	}

	*pnDataWritten = nDataToWrite;

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
	pstSharedMemoryChannel->nWriteReferenceCount--;

	if(pstSharedMemoryChannel->nReadReferenceCount > 0)
	{
		UCThreadEvent_SetEvent(pstSharedMemoryChannel->hReadEvent);
	}
	UCThreadMutex_Unlock(pstSharedMemoryChannel->hMutex);
_EXIT:
	return result;
}


static uem_result clearOutputChunkInfo(SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;
	void *pNextChunkDataEnd = NULL;
	int nSegmentLen = 0;

	if(pstSharedMemoryChannel->pDataEnd == pstSharedMemoryChannel->pBuffer + pstChannel->nBufSize)
	{
		pstSharedMemoryChannel->pDataEnd = pstSharedMemoryChannel->pBuffer;
	}

	for(nLoop = 0 ; nLoop < pstSharedMemoryChannel->stOutputPortChunk.nChunkNum ; nLoop++)
	{
		// if(pstChannel->stOutputPortChunk.astChunk[nLoop].pDataEnd )
		pNextChunkDataEnd = pstSharedMemoryChannel->pDataEnd + pstSharedMemoryChannel->stOutputPortChunk.nChunkLen * nLoop;

		if(pNextChunkDataEnd >= pstSharedMemoryChannel->pBuffer + pstChannel->nBufSize)
		{
			nSegmentLen = pNextChunkDataEnd - (pstSharedMemoryChannel->pBuffer + pstChannel->nBufSize);
			pstSharedMemoryChannel->stOutputPortChunk.astChunk[nLoop].pDataEnd = pstSharedMemoryChannel->pBuffer + nSegmentLen;
		}
		else
		{
			pstSharedMemoryChannel->stOutputPortChunk.astChunk[nLoop].pDataEnd = pNextChunkDataEnd;
		}

		// maximum available number is not needed for output chunk
		pstSharedMemoryChannel->stOutputPortChunk.astChunk[nLoop].nAvailableDataNum = 1;

		// clear chunk data len
		pstSharedMemoryChannel->stOutputPortChunk.astChunk[nLoop].nChunkDataLen = 0;
	}

	pstSharedMemoryChannel->nWrittenOutputChunkNum = 0;

	result = ERR_UEM_NOERROR;
//_EXIT:
	return result;
}



static uem_result writeToArrayQueue(SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nCurrentSampleRateIndex = 0;
	int nCurrentReadSampleRateIndex = 0;
	int nExpectedProduceSize = 0;
	int nExpectedConsumeSize = 0;

	result = UCThreadMutex_Lock(pstSharedMemoryChannel->hMutex);
	ERRIFGOTO(result, _EXIT);

	pstSharedMemoryChannel->nWriteReferenceCount++;

	nCurrentSampleRateIndex = pstChannel->pstOutputPort->nCurrentSampleRateIndex;
	nExpectedProduceSize = pstChannel->pstOutputPort->astSampleRates[nCurrentSampleRateIndex].nSampleRate * pstChannel->pstOutputPort->nSampleSize;

	// TODO: Error check out exit logic needed
	while(pstSharedMemoryChannel->nDataLen + nExpectedProduceSize > pstChannel->nBufSize || // nBuffer is full or
		(pstSharedMemoryChannel->nWrittenOutputChunkNum > 0 &&
		pstSharedMemoryChannel->nWrittenOutputChunkNum < pstSharedMemoryChannel->stOutputPortChunk.nChunkNum &&
		pstSharedMemoryChannel->stOutputPortChunk.astChunk[nChunkIndex].nChunkDataLen > 0)) // current chunk index is already filled with data
	{
		if(pstSharedMemoryChannel->bWriteExit == TRUE)
		{
			UEMASSIGNGOTO(result, ERR_UEM_SUSPEND, _EXIT_LOCK);
		}

		if(pstSharedMemoryChannel->nWriteReferenceCount > 1 &&
			pstSharedMemoryChannel->nDataLen + nExpectedProduceSize <= pstChannel->nBufSize)
		{
			result = UCThreadEvent_SetEvent(pstSharedMemoryChannel->hWriteEvent);
			ERRIFGOTO(result, _EXIT_LOCK);
		}

		result = UCThreadMutex_Unlock(pstSharedMemoryChannel->hMutex);
		ERRIFGOTO(result, _EXIT);

		result = UCThreadEvent_WaitEvent(pstSharedMemoryChannel->hWriteEvent);
		ERRIFGOTO(result, _EXIT);

		result = UCThreadMutex_Lock(pstSharedMemoryChannel->hMutex);
		ERRIFGOTO(result, _EXIT);
	}

	// Chunk needs to be initialized
	if(pstSharedMemoryChannel->nWrittenOutputChunkNum < 0 || pstSharedMemoryChannel->nWrittenOutputChunkNum >= pstSharedMemoryChannel->stOutputPortChunk.nChunkNum)
	{
		result = clearOutputChunkInfo(pstChannel, pstSharedMemoryChannel);
		ERRIFGOTO(result, _EXIT_LOCK);
	}

	result = copyToRoundedChunk(pstChannel, pstSharedMemoryChannel, pBuffer, nDataToWrite, nChunkIndex);
	ERRIFGOTO(result, _EXIT_LOCK);

	pstSharedMemoryChannel->stOutputPortChunk.astChunk[nChunkIndex].nChunkDataLen = nDataToWrite;

	if(pstSharedMemoryChannel->stOutputPortChunk.astChunk[nChunkIndex].nChunkDataLen == pstSharedMemoryChannel->stOutputPortChunk.nChunkLen)
	{
		pstSharedMemoryChannel->nWrittenOutputChunkNum++;
	}

	if(pstSharedMemoryChannel->nWrittenOutputChunkNum == pstSharedMemoryChannel->stOutputPortChunk.nChunkNum)
	{
		void *pNewEnd = NULL;
		int nSegmentLen = 0;
		pNewEnd = pstSharedMemoryChannel->pDataEnd + nExpectedProduceSize;
		if(pNewEnd >= pstSharedMemoryChannel->pBuffer + pstChannel->nBufSize)
		{
			nSegmentLen = pNewEnd - (pstSharedMemoryChannel->pBuffer + pstChannel->nBufSize);
			pNewEnd = pstSharedMemoryChannel->pBuffer + nSegmentLen;
		}
		pstSharedMemoryChannel->nDataLen += nExpectedProduceSize;
		pstSharedMemoryChannel->pDataEnd = pNewEnd;
	}

	nCurrentReadSampleRateIndex = pstChannel->pstInputPort->nCurrentSampleRateIndex;
	nExpectedConsumeSize = pstChannel->pstInputPort->astSampleRates[nCurrentReadSampleRateIndex].nSampleRate * pstChannel->pstInputPort->nSampleSize;

	if(pstChannel->enChannelType == CHANNEL_TYPE_FULL_ARRAY && nExpectedConsumeSize <= pstSharedMemoryChannel->nDataLen)
	{
		result = makeAvailableInputChunkList(pstChannel, pstSharedMemoryChannel);
		ERRIFGOTO(result, _EXIT_LOCK);
	}

	*pnDataWritten = nDataToWrite;

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
	pstSharedMemoryChannel->nWriteReferenceCount--;

	if(pstSharedMemoryChannel->nReadReferenceCount > 0)
	{
		UCThreadEvent_SetEvent(pstSharedMemoryChannel->hReadEvent);
	}

	if(pstSharedMemoryChannel->nWriteReferenceCount > 0)
	{
		UCThreadEvent_SetEvent(pstSharedMemoryChannel->hWriteEvent);
	}

	UCThreadMutex_Unlock(pstSharedMemoryChannel->hMutex);
_EXIT:
	return result;
}


uem_result UKChannelMemory_WriteToQueue (SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten)
{
	uem_result result = ERR_UEM_UNKNOWN;

	IFVARERRASSIGNGOTO(pstSharedMemoryChannel, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	// array channel
	if(pstChannel->enChannelType == CHANNEL_TYPE_OUTPUT_ARRAY || pstChannel->enChannelType == CHANNEL_TYPE_FULL_ARRAY)
	{
		result = writeToArrayQueue(pstChannel, pstSharedMemoryChannel, pBuffer, nDataToWrite, nChunkIndex, pnDataWritten);
		ERRIFGOTO(result, _EXIT);
	}
	else
	{
		result = writeToGeneralQueue(pstChannel, pstSharedMemoryChannel, pBuffer, nDataToWrite, nChunkIndex, pnDataWritten);
		ERRIFGOTO(result, _EXIT);
	}

	// to preserve, ERR_UEM_SUSPEND, do not set ERR_UEM_NOERROR here
_EXIT:
	return result;
}


uem_result UKChannelMemory_WriteToBuffer (SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SGenericMemoryAccess *pstMemoryAPI = NULL;

	IFVARERRASSIGNGOTO(pstSharedMemoryChannel, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	pstMemoryAPI = pstSharedMemoryChannel->pstMemoryAccessAPI;

	result = UCThreadMutex_Lock(pstSharedMemoryChannel->hMutex);
	ERRIFGOTO(result, _EXIT);

	if(pstChannel->nBufSize >= nDataToWrite)
	{
		if (nDataToWrite > 0)
		{
			result = pstMemoryAPI->fnCopyToMemory(pstSharedMemoryChannel->pDataStart, pBuffer, nDataToWrite);
			ERRIFGOTO(result, _EXIT_LOCK);
		}

		*pnDataWritten = nDataToWrite;
	}
	else // pstChannel->nBufSize < nDataToWrite
	{
		result = pstMemoryAPI->fnCopyToMemory(pstSharedMemoryChannel->pDataStart, pBuffer, pstChannel->nBufSize);
		ERRIFGOTO(result, _EXIT_LOCK);

		*pnDataWritten = pstChannel->nBufSize;
	}

	pstSharedMemoryChannel->nDataLen = *pnDataWritten;

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
	if(pstSharedMemoryChannel->nReadReferenceCount > 0)
	{
		UCThreadEvent_SetEvent(pstSharedMemoryChannel->hReadEvent);
	}
	UCThreadMutex_Unlock(pstSharedMemoryChannel->hMutex);
_EXIT:
	return result;
}


static uem_result getAvailableChunkFromArrayQueue(SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel, OUT int *pnChunkIndex)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nAvailableIndex = 0;
	SAvailableChunk *pstNextChunk = NULL;
	SAvailableChunk *pstCurChunk = NULL;

	result = UCThreadMutex_Lock(pstSharedMemoryChannel->hMutex);
	ERRIFGOTO(result, _EXIT);

	pstSharedMemoryChannel->nReadReferenceCount++;

	while(pstSharedMemoryChannel->pstAvailableInputChunkHead == NULL || pstSharedMemoryChannel->bInitialDataUpdated == FALSE)
	{
		if(pstSharedMemoryChannel->bReadExit == TRUE)
		{
			UEMASSIGNGOTO(result, ERR_UEM_SUSPEND, _EXIT_LOCK);
		}

		result = UCThreadMutex_Unlock(pstSharedMemoryChannel->hMutex);
		ERRIFGOTO(result, _EXIT);

		result = UCThreadEvent_WaitEvent(pstSharedMemoryChannel->hReadEvent);
		ERRIFGOTO(result, _EXIT);

		result = UCThreadMutex_Lock(pstSharedMemoryChannel->hMutex);
		ERRIFGOTO(result, _EXIT);
	}

	nAvailableIndex = pstSharedMemoryChannel->pstAvailableInputChunkHead->nChunkIndex;
	pstCurChunk = pstSharedMemoryChannel->pstAvailableInputChunkHead;

	pstSharedMemoryChannel->pstAvailableInputChunkTail->pstNext = pstCurChunk;
	pstCurChunk->pstPrev = pstSharedMemoryChannel->pstAvailableInputChunkTail;

	pstNextChunk = pstCurChunk->pstNext;
	pstNextChunk->pstPrev = NULL;

	pstSharedMemoryChannel->pstAvailableInputChunkTail = pstCurChunk;
	pstSharedMemoryChannel->pstAvailableInputChunkTail->pstNext = NULL;

	pstSharedMemoryChannel->pstAvailableInputChunkHead = pstNextChunk;

	*pnChunkIndex = nAvailableIndex;

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
pstSharedMemoryChannel->nReadReferenceCount--;

	if(pstSharedMemoryChannel->nWriteReferenceCount > 0)
	{
		UCThreadEvent_SetEvent(pstSharedMemoryChannel->hWriteEvent);
	}
	UCThreadMutex_Unlock(pstSharedMemoryChannel->hMutex);
_EXIT:
	return result;
}


static uem_result getAvailableChunkFromGeneralQueue(SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel, OUT int *pnChunkIndex)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = UCThreadMutex_Lock(pstSharedMemoryChannel->hMutex);
	ERRIFGOTO(result, _EXIT);

	pstSharedMemoryChannel->nReadReferenceCount++;

	// wait until the data is arrived
	while(pstSharedMemoryChannel->nDataLen <= 0 || pstSharedMemoryChannel->bInitialDataUpdated == FALSE)
	{
		if(pstSharedMemoryChannel->bReadExit == TRUE)
		{
			UEMASSIGNGOTO(result, ERR_UEM_SUSPEND, _EXIT_LOCK);
		}

		result = UCThreadMutex_Unlock(pstSharedMemoryChannel->hMutex);
		ERRIFGOTO(result, _EXIT);

		result = UCThreadEvent_WaitEvent(pstSharedMemoryChannel->hReadEvent);
		ERRIFGOTO(result, _EXIT);

		result = UCThreadMutex_Lock(pstSharedMemoryChannel->hMutex);
		ERRIFGOTO(result, _EXIT);
	}

	// index is always zero.
	*pnChunkIndex = 0;

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
	pstSharedMemoryChannel->nReadReferenceCount--;

	if(pstSharedMemoryChannel->nWriteReferenceCount > 0)
	{
		UCThreadEvent_SetEvent(pstSharedMemoryChannel->hWriteEvent);
	}
	UCThreadMutex_Unlock(pstSharedMemoryChannel->hMutex);
_EXIT:
	return result;
}


uem_result UKChannelMemory_GetAvailableChunk (SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel, OUT int *pnChunkIndex)
{
	uem_result result = ERR_UEM_UNKNOWN;

	IFVARERRASSIGNGOTO(pstSharedMemoryChannel, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	if(pstChannel->enChannelType == CHANNEL_TYPE_GENERAL || pstChannel->enChannelType == CHANNEL_TYPE_OUTPUT_ARRAY)
	{
		result = getAvailableChunkFromGeneralQueue(pstChannel, pstSharedMemoryChannel, pnChunkIndex);
		ERRIFGOTO(result, _EXIT);
	}
	else
	{
		result = getAvailableChunkFromArrayQueue(pstChannel, pstSharedMemoryChannel, pnChunkIndex);
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKChannelMemory_GetNumOfAvailableData (SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel, IN int nChunkIndex, OUT int *pnDataNum)
{
	uem_result result = ERR_UEM_UNKNOWN;

	IFVARERRASSIGNGOTO(pstSharedMemoryChannel, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	result = UCThreadMutex_Lock(pstSharedMemoryChannel->hMutex);
	ERRIFGOTO(result, _EXIT);

	if(pstChannel->enChannelType == CHANNEL_TYPE_GENERAL || pstChannel->enChannelType == CHANNEL_TYPE_OUTPUT_ARRAY)
	{
		*pnDataNum = pstSharedMemoryChannel->nDataLen;
	}
	else // if( pstChannel->enChannelType == CHANNEL_TYPE_INPUT_ARRAY || pstChannel->enChannelType == CHANNEL_TYPE_FULL_ARRAY)
	{
		if(pstSharedMemoryChannel->pstAvailableInputChunkHead != NULL)
		{
			*pnDataNum = pstSharedMemoryChannel->stInputPortChunk.astChunk[nChunkIndex].nChunkDataLen;
		}
		else
		{
			*pnDataNum = 0;
		}
	}

	result = ERR_UEM_NOERROR;

	UCThreadMutex_Unlock(pstSharedMemoryChannel->hMutex);
_EXIT:
	return result;
}

uem_result UKChannelMemory_SetExit(SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel, int nExitFlag)
{
	uem_result result = ERR_UEM_UNKNOWN;

	IFVARERRASSIGNGOTO(pstSharedMemoryChannel, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	result = UCThreadMutex_Lock(pstSharedMemoryChannel->hMutex);
	ERRIFGOTO(result, _EXIT);

	if((nExitFlag & EXIT_FLAG_READ) != 0)
	{
		pstSharedMemoryChannel->bReadExit = TRUE;

		result = UCThreadEvent_SetEvent(pstSharedMemoryChannel->hReadEvent);
		ERRIFGOTO(result, _EXIT_LOCK);
	}

	if((nExitFlag & EXIT_FLAG_WRITE) != 0)
	{
		pstSharedMemoryChannel->bWriteExit = TRUE;

		result = UCThreadEvent_SetEvent(pstSharedMemoryChannel->hWriteEvent);
		ERRIFGOTO(result, _EXIT_LOCK);
	}

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
	UCThreadMutex_Unlock(pstSharedMemoryChannel->hMutex);
_EXIT:
	return result;
}


uem_result UKChannelMemory_ClearExit(SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel, int nExitFlag)
{
	uem_result result = ERR_UEM_UNKNOWN;

	IFVARERRASSIGNGOTO(pstSharedMemoryChannel, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	result = UCThreadMutex_Lock(pstSharedMemoryChannel->hMutex);
	ERRIFGOTO(result, _EXIT);

	if((nExitFlag & EXIT_FLAG_READ) != 0)
	{
		pstSharedMemoryChannel->bReadExit = FALSE;
	}

	if((nExitFlag & EXIT_FLAG_WRITE) != 0)
	{
		pstSharedMemoryChannel->bWriteExit = FALSE;
	}

	result = ERR_UEM_NOERROR;

	UCThreadMutex_Unlock(pstSharedMemoryChannel->hMutex);
_EXIT:
	return result;
}


static uem_result fillInitialData(SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel, int nDataToFill)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nCurrentSampleRateIndex = 0;
	void *pBuffer = NULL;
	int nExpectedConsumeSize = 0;
	SGenericMemoryAccess *pstMemoryAPI = NULL;

	pstMemoryAPI = pstSharedMemoryChannel->pstMemoryAccessAPI;

	// fill non-set memory (does not need to set the memory after creating memory)
	result = pstMemoryAPI->fnCreateMemory(nDataToFill, pstChannel->nProcessorId, &pBuffer);
	ERRIFGOTO(result, _EXIT);

	result = copyAndMovePointerToRoundedQueue(pstChannel, pstSharedMemoryChannel, pBuffer, nDataToFill, 0, pstMemoryAPI->fnCopyInMemory);
	ERRIFGOTO(result, _EXIT);

	if(pstChannel->enChannelType == CHANNEL_TYPE_INPUT_ARRAY && pstSharedMemoryChannel->pstAvailableInputChunkHead == NULL)
	{
		nCurrentSampleRateIndex = pstChannel->pstInputPort->nCurrentSampleRateIndex;
		nExpectedConsumeSize = pstChannel->pstInputPort->astSampleRates[nCurrentSampleRateIndex].nSampleRate * pstChannel->pstInputPort->nSampleSize;

		if(nExpectedConsumeSize <= pstSharedMemoryChannel->nDataLen)
		{
			result = makeAvailableInputChunkList(pstChannel, pstSharedMemoryChannel);
			ERRIFGOTO(result, _EXIT);
		}
	}

	result = ERR_UEM_NOERROR;

_EXIT:
	if(pBuffer != NULL)
	{
		pstMemoryAPI->fnDestroyMemory((void *) &pBuffer);
	}
	return result;
}



uem_result UKChannelMemory_FillInitialData(SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nDataToWrite = 0;

	IFVARERRASSIGNGOTO(pstSharedMemoryChannel, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	if(pstChannel->nInitialDataLen == 0) // do nothing if there is no initial data
	{
		UEMASSIGNGOTO(result, ERR_UEM_NOERROR, _EXIT);
	}

	result = UCThreadMutex_Lock(pstSharedMemoryChannel->hMutex);
	ERRIFGOTO(result, _EXIT);

	pstSharedMemoryChannel->bInitialDataUpdated = TRUE;
	nDataToWrite = pstChannel->nInitialDataLen - pstSharedMemoryChannel->nDataLen;

	// if the initial data is already set in TASK_INIT function, do not fill the initial data
	if(nDataToWrite > 0)
	{
		result = fillInitialData(pstChannel,pstSharedMemoryChannel, nDataToWrite);
		ERRIFGOTO(result, _EXIT_LOCK);
	}

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
	if(pstSharedMemoryChannel->nReadReferenceCount > 0)
	{
		UCThreadEvent_SetEvent(pstSharedMemoryChannel->hReadEvent);
	}
	UCThreadMutex_Unlock(pstSharedMemoryChannel->hMutex);
_EXIT:
	return result;
}



uem_result UKChannelMemory_Finalize(SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel)
{
	uem_result result = ERR_UEM_UNKNOWN;

	IFVARERRASSIGNGOTO(pstSharedMemoryChannel, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	// ignore error
	UCThreadMutex_Destroy(&(pstSharedMemoryChannel->hMutex));

	// ignore error
	UCThreadEvent_Destroy(&(pstSharedMemoryChannel->hReadEvent));
	UCThreadEvent_Destroy(&(pstSharedMemoryChannel->hWriteEvent));

	if(pstSharedMemoryChannel->pBuffer != NULL && pstSharedMemoryChannel->bStaticAllocation == FALSE)
	{
		pstSharedMemoryChannel->pstMemoryAccessAPI->fnDestroyMemory(&(pstSharedMemoryChannel->pBuffer));
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}






