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

	for(nLoop = 0; nLoop < ARRAYLEN(g_pstChannels) ; nLoop++)
	{
		switch(g_pstChannels[nLoop].enType)
		{
		case CHANNEL_TYPE_SHARED_MEMORY:
			break;
		case CHANNEL_TYPE_TCP_SERVER:
			break;
		case CHANNEL_TYPE_TCP_CLIENT:
			break;
		default:
			break;
		}

		if( g_pstChannels[nLoop].stInputPortChunk.nChunkNum > 1 )
		{
			//channel is read by multiple tasks
		}

		if( g_pstChannels[nLoop].stOutputPortChunk.nChunkNum > 1 )
		{
			// channel is written by multiple tasks
		}
	}



	return result;
}


uem_result UKChannel_Finalize()
{
	uem_result result = ERR_UEM_NOERROR;

	return result;
}


