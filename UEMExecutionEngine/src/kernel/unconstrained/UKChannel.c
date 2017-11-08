/*
 * UKChannel.c
 *
 *  Created on: 2017. 8. 28.
 *      Author: jej
 */


#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <UCBasic.h>

#include <uem_data.h>



uem_result UKChannel_InitializeSharedMemory(SChannel *pstChannel)
{
	uem_result result = ERR_UEM_UNKNOWN;

	// initialize buffer
	// If not set, initialize those things

	// information clear
	// pBuffer => is NULL => alloc
	if(pstChannel->pBuffer == NULL)
	{
		pstChannel->pBuffer = UC_malloc(pstChannel->nBufSize);
		ERRMEMGOTO(pstChannel->pBuffer, result, _EXIT);
	}

	// pDataStart => clear (pBuffer)
	// nDataLen => clear (0)
	pstChannel->pDataStart = pstChannel->pBuffer;
	pstChannel->pDataEnd = pstChannel->pBuffer;
	pstChannel->nDataLen = 0;
	pstChannel->nReferenceCount = 0;

	// hMutex => initialize/create
	result = UCThreadMutex_Create(&(pstChannel->hMutex));
	ERRIFGOTO(result, _EXIT);

	// hEvent => initialize/create
	result = UCThreadEvent_Create(&(pstChannel->hEvent));
	ERRIFGOTO(result, _EXIT);


	// SPort
	// nCurrentSampleRateIndex
	pstChannel->stInputPort.nCurrentSampleRateIndex;
	// iterative access on subgraph port
	pstChannel->stOutputPort.nCurrentSampleRateIndex;
	//pstChannel->stInputPort.pstSubGraphPort->
	pstChannel->stInputPortChunk.astChunk;


	// SChunkInfo
	// nChunkNum - 1 (for general task) or total sample rate / most inner task port's sample rate (for loop task)
	// nChunkLen -  most inner task port's sample rate * sample size

	// SChunk
	// chunk start pointer clear
	// data start pointer clear
	// data end pointer clear
	// written data length = 0
	// available data number clear



	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKChannel_InitializeTCPServer()
{
	uem_result result = ERR_UEM_UNKNOWN;

	// TCP Server initialize

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKChannel_InitializeTCPClient()
{
	uem_result result = ERR_UEM_UNKNOWN;

	// TCP client Initialize

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKChannel_Initialize()
{
	uem_result result = ERR_UEM_NOERROR;
	int nLoop = 0;

	for(nLoop = 0; nLoop < g_nChannelNum; nLoop++)
	{
		switch(g_astChannels[nLoop].enType)
		{
		case COMMUNICATION_TYPE_SHARED_MEMORY:
			UKChannel_InitializeSharedMemory(&(g_astChannels[nLoop]));
			// Shared memory initialization
			break;
		case COMMUNICATION_TYPE_TCP_SERVER:
			// Server-side TCP channel initialization
			break;
		case COMMUNICATION_TYPE_TCP_CLIENT:
			// Client-side TCP channel initialization
			break;
		default:
			break;
		}

		if( g_astChannels[nLoop].stInputPortChunk.nChunkNum > 1 )
		{
			//channel is read by multiple tasks
		}

		if( g_astChannels[nLoop].stOutputPortChunk.nChunkNum > 1 )
		{
			// channel is written by multiple tasks
		}
	}

	return result;
}




uem_result UKChannel_Read(int nChannelId, int nChunkIndex)
{
	uem_result result = ERR_UEM_UNKNOWN;

	//Assign chunk
	//g_astChannels[nChannelId].stInputPortChunk.astChunk[nChunkIndex].pChunkStart;



	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}




uem_result UKChannel_Finalize()
{
	uem_result result = ERR_UEM_NOERROR;

	return result;
}


