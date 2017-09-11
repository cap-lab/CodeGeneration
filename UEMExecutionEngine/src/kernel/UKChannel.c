/*
 * UKChannel.c
 *
 *  Created on: 2017. 8. 28.
 *      Author: jej
 */


#include <uem_data.h>

uem_result UKChannel_Initialize()
{
	uem_result result = ERR_UEM_NOERROR;
	int nLoop = 0;

	for(nLoop = 0; nLoop < ARRAYLEN(g_astChannels) ; nLoop++)
	{
		switch(g_astChannels[nLoop].enType)
		{
		case CHANNEL_TYPE_SHARED_MEMORY:
			UKChannel_InitializeSharedMemory(&g_astChannels[nLoop]);
			// Shared memory initialization
			break;
		case CHANNEL_TYPE_TCP_SERVER:
			// Server-side TCP channel initialization
			break;
		case CHANNEL_TYPE_TCP_CLIENT:
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


uem_result UKChannel_InitializeSharedMemory(SChannel *pstChannel)
{
	uem_result result = ERR_UEM_UNKNOWN;

	// initialize buffer
	// If not set, initialize those things

	// information clear
	// pBuffer => is NULL => alloc
	// pDataStart => clear (pBuffer)
	// nDataLen => clear (0)
	// hMutex => initialize/create

	// SPort
	// nCurrentSampleRateIndex
	// iterative access on subgraph port

	// SChunkInfo
	// nChunkNum - 1 (for general task) or total sample rate / most inner task port's sample rate (for loop task)
	// nChunkLen -  most inner task port's sample rate * sample size

	// SChunk
	// chunk start pointer clear
	// data start pointer clear
	// data end pointer clear
	// written data length = 0
	// available data number clear
	// mutex
	// conditional variable



	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKChannel_Read(int nChannelId, int nChunkIndex)
{
	uem_result result = ERR_UEM_UNKNOWN;

	//Assign chunk
	g_astChannels[nChannelId].stInputPortChunk.astChunk[nChunkIndex].pChunkStart;



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


uem_result UKChannel_Finalize()
{
	uem_result result = ERR_UEM_NOERROR;

	return result;
}


