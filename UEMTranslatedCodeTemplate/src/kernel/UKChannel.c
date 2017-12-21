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
#include <UKSharedMemoryChannel.h>

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
			UKSharedMemoryChannel_Initialize(&(g_astChannels[nLoop]));
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


