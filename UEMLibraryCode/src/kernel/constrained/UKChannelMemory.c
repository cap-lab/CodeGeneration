/*
 * UKChannelMemory.c
 *
 *  Created on: 2018. 9. 4.
 *      Author: chjej202
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <UCBasic.h>

#include <uem_data.h>

uem_result UKChannelMemory_Clear(SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel)
{
	uem_result result = ERR_UEM_UNKNOWN;

	IFVARERRASSIGNGOTO(pstSharedMemoryChannel, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	// information clear
	// pDataStart => clear (pBuffer)
	// nDataLen => clear (0)
	pstSharedMemoryChannel->pDataStart = pstSharedMemoryChannel->pBuffer;
	pstSharedMemoryChannel->pDataEnd = pstSharedMemoryChannel->pBuffer;
	pstSharedMemoryChannel->nDataLen = 0;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKChannelMemory_Initialize(SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel)
{
	uem_result result = ERR_UEM_UNKNOWN;

	IFVARERRASSIGNGOTO(pstSharedMemoryChannel, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	result = UKChannelMemory_Clear(pstChannel, pstSharedMemoryChannel);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static uem_result copyAndMovePointerFromRoundedQueue(unsigned char *pDest, SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel, int nDataToRead)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nSegmentLen = 0;
	int nRemainderLen = 0;

	// data is divided into two segments because the start point of the chunk is close to the end of buffer range
	if(pstSharedMemoryChannel->pDataStart + nDataToRead > pstSharedMemoryChannel->pBuffer + pstChannel->nBufSize)
	{
		nSegmentLen = pstSharedMemoryChannel->pBuffer + pstChannel->nBufSize - pstSharedMemoryChannel->pDataStart;
		nRemainderLen = nDataToRead - nSegmentLen;

		UC_memcpy(pDest, pstSharedMemoryChannel->pDataStart, nSegmentLen);
		UC_memcpy(pDest + nSegmentLen, pstSharedMemoryChannel->pBuffer, nRemainderLen);

		pstSharedMemoryChannel->pDataStart = pstSharedMemoryChannel->pBuffer + nRemainderLen;
	}
	else
	{
		UC_memcpy(pDest, pstSharedMemoryChannel->pDataStart, nDataToRead);

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

	return result;
}


static uem_result copyAndMovePointerToRoundedQueue(SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel,
				unsigned char *pSrc, int nDataToWrite)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nSegmentLen = 0;
	int nRemainderLen = 0;

	// Chunk is divided into two segments because the start point of the chunk is close to the end of buffer range
	if(pstSharedMemoryChannel->pDataEnd + nDataToWrite > pstSharedMemoryChannel->pBuffer + pstChannel->nBufSize)
	{
		nSegmentLen = pstSharedMemoryChannel->pBuffer + pstChannel->nBufSize - pstSharedMemoryChannel->pDataEnd;
		nRemainderLen = nDataToWrite - nSegmentLen;

		UC_memcpy(pstSharedMemoryChannel->pDataEnd, pSrc, nSegmentLen);
		UC_memcpy(pstSharedMemoryChannel->pBuffer, pSrc + nSegmentLen, nRemainderLen);

		pstSharedMemoryChannel->pDataEnd = pstSharedMemoryChannel->pBuffer + nDataToWrite - nSegmentLen;
	}
	else
	{
		UC_memcpy(pstSharedMemoryChannel->pDataEnd, pSrc, nDataToWrite);

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

	return result;
}


static uem_result readFromGeneralQueue(SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead)
{
	uem_result result = ERR_UEM_UNKNOWN;

	if(pstSharedMemoryChannel->nDataLen < nDataToRead || pstSharedMemoryChannel->bInitialDataUpdated == FALSE)
	{
		ERRASSIGNGOTO(result, ERR_UEM_READ_BLOCK, _EXIT);
	}

	result = copyAndMovePointerFromRoundedQueue(pBuffer, pstChannel, pstSharedMemoryChannel, nDataToRead);
	ERRIFGOTO(result, _EXIT);

	*pnDataRead = nDataToRead;

	result = ERR_UEM_NOERROR;
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
		ERRASSIGNGOTO(result, ERR_UEM_NOT_SUPPORTED, _EXIT);
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

	IFVARERRASSIGNGOTO(pstSharedMemoryChannel, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	if(pstSharedMemoryChannel->nDataLen < nDataToRead)
	{
		if(pstSharedMemoryChannel->nDataLen > 0)
		{
			UC_memcpy(pBuffer, pstSharedMemoryChannel->pDataStart, pstSharedMemoryChannel->nDataLen);
		}

		*pnDataRead = pstSharedMemoryChannel->nDataLen;
	}
	else // pstChannel->nDataLen >= nDataToRead
	{
		UC_memcpy(pBuffer, pstSharedMemoryChannel->pDataStart, nDataToRead);

		*pnDataRead = nDataToRead;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static uem_result writeToGeneralQueue(SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten)
{
	uem_result result = ERR_UEM_UNKNOWN;

	if(pstSharedMemoryChannel->nDataLen + nDataToWrite > pstChannel->nBufSize)
	{
		ERRASSIGNGOTO(result, ERR_UEM_WRITE_BLOCK, _EXIT);
	}

	result = copyAndMovePointerToRoundedQueue(pstChannel, pstSharedMemoryChannel, pBuffer, nDataToWrite);
	ERRIFGOTO(result, _EXIT);

	*pnDataWritten = nDataToWrite;

	result = ERR_UEM_NOERROR;
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
		ERRASSIGNGOTO(result, ERR_UEM_NOT_SUPPORTED, _EXIT);
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

	IFVARERRASSIGNGOTO(pstSharedMemoryChannel, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	if(pstChannel->nBufSize >= nDataToWrite)
	{
		if (nDataToWrite > 0)
		{
			UC_memcpy(pstSharedMemoryChannel->pBuffer, pBuffer, nDataToWrite);
		}

		*pnDataWritten = nDataToWrite;
	}
	else // pstChannel->nBufSize < nDataToWrite
	{
		UC_memcpy(pstSharedMemoryChannel->pBuffer, pBuffer, pstChannel->nBufSize);

		*pnDataWritten = pstChannel->nBufSize;
	}

	pstSharedMemoryChannel->pDataStart = pstSharedMemoryChannel->pBuffer;

	pstSharedMemoryChannel->nDataLen = *pnDataWritten;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result getAvailableChunkFromGeneralQueue(SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel, OUT int *pnChunkIndex)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nDataToRead = 0;

	if(pstSharedMemoryChannel->nDataLen < nDataToRead || pstSharedMemoryChannel->bInitialDataUpdated == FALSE)
	{
		ERRASSIGNGOTO(result, ERR_UEM_READ_BLOCK, _EXIT);
	}

	// index is always zero.
	*pnChunkIndex = 0;

	result = ERR_UEM_NOERROR;
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
		ERRASSIGNGOTO(result, ERR_UEM_NOT_SUPPORTED, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKChannelMemory_GetNumOfAvailableData (SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel, IN int nChunkIndex, OUT int *pnDataNum)
{
	uem_result result = ERR_UEM_UNKNOWN;

	IFVARERRASSIGNGOTO(pstSharedMemoryChannel, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	if(pstChannel->enChannelType == CHANNEL_TYPE_GENERAL || pstChannel->enChannelType == CHANNEL_TYPE_OUTPUT_ARRAY)
	{
		*pnDataNum = pstSharedMemoryChannel->nDataLen;
	}
	else // if( pstChannel->enChannelType == CHANNEL_TYPE_INPUT_ARRAY || pstChannel->enChannelType == CHANNEL_TYPE_FULL_ARRAY)
	{
		ERRASSIGNGOTO(result, ERR_UEM_NOT_SUPPORTED, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKChannelMemory_SetExit(SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel, int nExitFlag)
{
	return ERR_UEM_NOERROR;
}


uem_result UKChannelMemory_ClearExit(SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel, int nExitFlag)
{
	return ERR_UEM_NOERROR;
}


static uem_result fillInitialData(SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel, int nDataToFill)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nValue = 0;
	int nDataFilled = 0;

	while(nDataFilled < nDataToFill)
	{
		result = copyAndMovePointerToRoundedQueue(pstChannel, pstSharedMemoryChannel, (unsigned char *) &nValue, MIN(sizeof(int), nDataToFill - nDataFilled));
		ERRIFGOTO(result, _EXIT);
		nDataFilled += MIN(sizeof(int), nDataToFill - nDataFilled);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
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

	pstSharedMemoryChannel->bInitialDataUpdated = TRUE;
	nDataToWrite = pstChannel->nInitialDataLen - pstSharedMemoryChannel->nDataLen;

	// if the initial data is already set in TASK_INIT function, do not fill the initial data
	if(nDataToWrite > 0)
	{
		result = fillInitialData(pstChannel,pstSharedMemoryChannel, nDataToWrite);
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}



uem_result UKChannelMemory_Finalize(SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel)
{
	uem_result result = ERR_UEM_UNKNOWN;

	IFVARERRASSIGNGOTO(pstSharedMemoryChannel, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	// do nothing

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


