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

static int getChannelIndexById(int nChannelId)
{
	int nLoop = 0;
	int nIndex = INVALID_CHANNEL_ID;

	for(nLoop = 0; nLoop < g_nChannelNum; nLoop++)
	{
		if(g_astChannels[nLoop].nChannelIndex == nChannelId)
		{
			nIndex = nLoop;
			break;
		}
	}

	return nIndex;
}

uem_result UKChannel_WriteToBuffer(int nChannelId, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nIndex = 0;

	nIndex = getChannelIndexById(nChannelId);
	IFVARERRASSIGNGOTO(nIndex, INVALID_CHANNEL_ID, result, ERR_UEM_INVALID_PARAM, _EXIT);

	switch(g_astChannels[nIndex].enType)
	{
	case COMMUNICATION_TYPE_SHARED_MEMORY:
		result = UKSharedMemoryChannel_WriteToBuffer(g_astChannels[nIndex], pBuffer, nDataToWrite, nChunkIndex, pnDataWritten);
		ERRIFGOTO(result, _EXIT);
		break;
	case COMMUNICATION_TYPE_TCP_SERVER:
		ERRASSIGNGOTO(result, ERR_UEM_NOT_SUPPORTED_YET, _EXIT);
		break;
	case COMMUNICATION_TYPE_TCP_CLIENT:
		ERRASSIGNGOTO(result, ERR_UEM_NOT_SUPPORTED_YET, _EXIT);
		break;
	default:
		break;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKChannel_WriteToQueue(int nChannelId, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nIndex = 0;

	nIndex = getChannelIndexById(nChannelId);
	IFVARERRASSIGNGOTO(nIndex, INVALID_CHANNEL_ID, result, ERR_UEM_INVALID_PARAM, _EXIT);

	switch(g_astChannels[nIndex].enType)
	{
	case COMMUNICATION_TYPE_SHARED_MEMORY:
		result = UKSharedMemoryChannel_WriteToQueue(g_astChannels[nIndex], pBuffer, nDataToWrite, nChunkIndex, pnDataWritten);
		ERRIFGOTO(result, _EXIT);
		break;
	case COMMUNICATION_TYPE_TCP_SERVER:
		ERRASSIGNGOTO(result, ERR_UEM_NOT_SUPPORTED_YET, _EXIT);
		break;
	case COMMUNICATION_TYPE_TCP_CLIENT:
		ERRASSIGNGOTO(result, ERR_UEM_NOT_SUPPORTED_YET, _EXIT);
		break;
	default:
		break;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKChannel_ReadFromQueue(int nChannelId, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nIndex = 0;

	nIndex = getChannelIndexById(nChannelId);
	IFVARERRASSIGNGOTO(nIndex, INVALID_CHANNEL_ID, result, ERR_UEM_INVALID_PARAM, _EXIT);

	switch(g_astChannels[nIndex].enType)
	{
	case COMMUNICATION_TYPE_SHARED_MEMORY:
		result = UKSharedMemoryChannel_ReadFromQueue(&g_astChannels[nIndex], pBuffer, nDataToRead, nChunkIndex, pnDataRead);
		ERRIFGOTO(result, _EXIT);
		break;
	case COMMUNICATION_TYPE_TCP_SERVER:
		ERRASSIGNGOTO(result, ERR_UEM_NOT_SUPPORTED_YET, _EXIT);
		break;
	case COMMUNICATION_TYPE_TCP_CLIENT:
		ERRASSIGNGOTO(result, ERR_UEM_NOT_SUPPORTED_YET, _EXIT);
		break;
	default:
		break;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKChannel_ReadFromBuffer(int nChannelId, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nIndex = 0;

	nIndex = getChannelIndexById(nChannelId);
	IFVARERRASSIGNGOTO(nIndex, INVALID_CHANNEL_ID, result, ERR_UEM_INVALID_PARAM, _EXIT);

	switch(g_astChannels[nIndex].enType)
	{
	case COMMUNICATION_TYPE_SHARED_MEMORY:
		result = UKSharedMemoryChannel_ReadFromBuffer(&g_astChannels[nIndex], pBuffer, nDataToRead, nChunkIndex, pnDataRead);
		ERRIFGOTO(result, _EXIT);
		break;
	case COMMUNICATION_TYPE_TCP_SERVER:
		ERRASSIGNGOTO(result, ERR_UEM_NOT_SUPPORTED_YET, _EXIT);
		break;
	case COMMUNICATION_TYPE_TCP_CLIENT:
		ERRASSIGNGOTO(result, ERR_UEM_NOT_SUPPORTED_YET, _EXIT);
		break;
	default:
		break;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKChannel_Finalize()
{
	uem_result result = ERR_UEM_NOERROR;
	int nLoop = 0;

	for(nLoop = 0; nLoop < g_nChannelNum; nLoop++)
	{
		UKSharedMemoryChannel_Finalize(&(g_astChannels[nLoop]));
	}

	return result;
}


