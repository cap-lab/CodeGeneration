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
#include <UCString.h>

#include <uem_data.h>
#include <UKSharedMemoryChannel.h>

typedef uem_result (*FnChannelInitialize)(SChannel *pstChannel);
typedef uem_result (*FnChannelReadFromQueue)(SChannel *pstChannel, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead);
typedef uem_result (*FnChannelReadFromBuffer)(SChannel *pstChannel, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead);
typedef uem_result (*FnChannelWriteToBuffer)(SChannel *pstChannel, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten);
typedef uem_result (*FnChannelWriteToQueue)(SChannel *pstChannel, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten);
typedef uem_result (*FnChannelGetAvailableChunk)(SChannel *pstChannel, OUT int *pnChunkIndex);
typedef uem_result (*FnChannelGetNumOfAvailableData)(SChannel *pstChannel, IN int nChunkIndex, OUT int *pnDataNum);
typedef uem_result (*FnChannelClear)(SChannel *pstChannel);
typedef uem_result (*FnChannelSetExit)(SChannel *pstChannel);
typedef uem_result (*FnChannelFinalize)(SChannel *pstChannel);

typedef struct _SChannelAPI {
	FnChannelInitialize fnInitialize;
	FnChannelReadFromQueue fnReadFromQueue;
	FnChannelReadFromBuffer fnReadFromBuffer;
	FnChannelWriteToQueue fnWriteToQueue;
	FnChannelWriteToBuffer fnWriteToBuffer;
	FnChannelGetAvailableChunk fnGetAvailableChunk;
	FnChannelGetNumOfAvailableData fnGetNumOfAvailableData;
	FnChannelClear fnClear;
	FnChannelSetExit fnSetExit;
	FnChannelFinalize fnFinalize;
} SChannelAPI;


SChannelAPI g_stSharedMemoryChannel = {
	UKSharedMemoryChannel_Initialize, // fnInitialize
	UKSharedMemoryChannel_ReadFromQueue, // fnReadFromQueue
	UKSharedMemoryChannel_ReadFromBuffer, // fnReadFromBuffer
	UKSharedMemoryChannel_WriteToQueue, // fnWriteToQueue
	UKSharedMemoryChannel_WriteToBuffer, // fnWriteToBuffer
	UKSharedMemoryChannel_GetAvailableChunk, // fnGetAvailableChunk
	UKSharedMemoryChannel_GetNumOfAvailableData, // fnGetNumOfAvailableData
	UKSharedMemoryChannel_Clear, // fnClear
	UKSharedMemoryChannel_SetExit,
	UKSharedMemoryChannel_Finalize, // fnFinalize
};


static uem_result getAPIStructureFromCommunicationType(IN ECommunicationType enType, OUT SChannelAPI **ppstChannelAPI)
{
	uem_result result = ERR_UEM_UNKNOWN;
	switch(enType)
	{
	case COMMUNICATION_TYPE_SHARED_MEMORY:
		*ppstChannelAPI = &g_stSharedMemoryChannel;
		break;
	case COMMUNICATION_TYPE_TCP_SERVER:
		ERRASSIGNGOTO(result, ERR_UEM_NOT_SUPPORTED_YET, _EXIT)
		break;
	case COMMUNICATION_TYPE_TCP_CLIENT:
		ERRASSIGNGOTO(result, ERR_UEM_NOT_SUPPORTED_YET, _EXIT)
		break;
	default:
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT)
		break;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKChannel_Initialize()
{
	uem_result result = ERR_UEM_NOERROR;
	int nLoop = 0;
	SChannelAPI *pstChannelAPI = NULL;

	for(nLoop = 0; nLoop < g_nChannelNum; nLoop++)
	{
		result = getAPIStructureFromCommunicationType(g_astChannels[nLoop].enType, &pstChannelAPI);
		ERRIFGOTO(result, _EXIT);

		result = pstChannelAPI->fnInitialize(&(g_astChannels[nLoop]));
		ERRIFGOTO(result, _EXIT);

		if( g_astChannels[nLoop].stInputPortChunk.nChunkNum > 1 )
		{
			//channel is read by multiple tasks
		}

		if( g_astChannels[nLoop].stOutputPortChunk.nChunkNum > 1 )
		{
			// channel is written by multiple tasks
		}
	}
_EXIT:
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

static uem_bool isPortTaskIdAndPortNameEqual(SPort *pstPort, uem_string strPortName, int nTaskId)
{
	uem_string_struct stStructPortName;
	uem_result result = ERR_UEM_UNKNOWN;
	uem_bool bIsMatch = FALSE;

	if(pstPort->nTaskId == nTaskId)
	{
		result = UCString_New(&stStructPortName, pstPort->pszPortName, UEMSTRING_MAX);
		ERRIFGOTO(result, _EXIT);

		if(UCString_IsEqual(strPortName, &stStructPortName) == TRUE)
		{
			bIsMatch = TRUE;
		}
	}
_EXIT:
	return bIsMatch;
}

int UKChannel_GetChannelIdByTaskAndPortName(int nTaskId, char *szPortName)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;
	int nIndex = INVALID_CHANNEL_ID;
	uem_string_struct stArgPortName;

	result = UCString_New(&stArgPortName, szPortName, UEMSTRING_MAX);
	ERRIFGOTO(result, _EXIT);

	for(nLoop = 0; nLoop < g_nChannelNum; nLoop++)
	{
		if(isPortTaskIdAndPortNameEqual(&(g_astChannels[nLoop].stInputPort), &stArgPortName, nTaskId) == TRUE ||
			isPortTaskIdAndPortNameEqual(&(g_astChannels[nLoop].stOutputPort), &stArgPortName, nTaskId) == TRUE)
		{
			nIndex = nLoop;
			break;
		}
	}
_EXIT:
	return nIndex;
}



uem_result UKChannel_WriteToBuffer(int nChannelId, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nIndex = 0;
	SChannelAPI *pstChannelAPI = NULL;

	nIndex = getChannelIndexById(nChannelId);
	IFVARERRASSIGNGOTO(nIndex, INVALID_CHANNEL_ID, result, ERR_UEM_INVALID_PARAM, _EXIT);

	result = getAPIStructureFromCommunicationType(g_astChannels[nIndex].enType, &pstChannelAPI);
	ERRIFGOTO(result, _EXIT);

	result = pstChannelAPI->fnWriteToBuffer(&g_astChannels[nIndex], pBuffer, nDataToWrite, nChunkIndex, pnDataWritten);
	ERRIFGOTO(result, _EXIT);

	// to preserve, ERR_UEM_SUSPEND, do not set UEM_NOERROR here
_EXIT:
	return result;
}


uem_result UKChannel_WriteToQueue(int nChannelId, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nIndex = 0;
	SChannelAPI *pstChannelAPI = NULL;

	nIndex = getChannelIndexById(nChannelId);
	IFVARERRASSIGNGOTO(nIndex, INVALID_CHANNEL_ID, result, ERR_UEM_INVALID_PARAM, _EXIT);

	result = getAPIStructureFromCommunicationType(g_astChannels[nIndex].enType, &pstChannelAPI);
	ERRIFGOTO(result, _EXIT);

	result = pstChannelAPI->fnWriteToQueue(&g_astChannels[nIndex], pBuffer, nDataToWrite, nChunkIndex, pnDataWritten);
	ERRIFGOTO(result, _EXIT);

	// to preserve, ERR_UEM_SUSPEND, do not set UEM_NOERROR here
_EXIT:
	return result;
}


uem_result UKChannel_ReadFromQueue(int nChannelId, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nIndex = 0;
	SChannelAPI *pstChannelAPI = NULL;

	nIndex = getChannelIndexById(nChannelId);
	IFVARERRASSIGNGOTO(nIndex, INVALID_CHANNEL_ID, result, ERR_UEM_INVALID_PARAM, _EXIT);

	result = getAPIStructureFromCommunicationType(g_astChannels[nIndex].enType, &pstChannelAPI);
	ERRIFGOTO(result, _EXIT);

	result = pstChannelAPI->fnReadFromQueue(&g_astChannels[nIndex], pBuffer, nDataToRead, nChunkIndex, pnDataRead);
	ERRIFGOTO(result, _EXIT);

	// to preserve, ERR_UEM_SUSPEND, do not set UEM_NOERROR here
_EXIT:
	return result;
}

uem_result UKChannel_ReadFromBuffer(int nChannelId, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nIndex = 0;
	SChannelAPI *pstChannelAPI = NULL;

	nIndex = getChannelIndexById(nChannelId);
	IFVARERRASSIGNGOTO(nIndex, INVALID_CHANNEL_ID, result, ERR_UEM_INVALID_PARAM, _EXIT);

	result = getAPIStructureFromCommunicationType(g_astChannels[nIndex].enType, &pstChannelAPI);
	ERRIFGOTO(result, _EXIT);

	result = pstChannelAPI->fnReadFromBuffer(&g_astChannels[nIndex], pBuffer, nDataToRead, nChunkIndex, pnDataRead);
	ERRIFGOTO(result, _EXIT);

	// to preserve, ERR_UEM_SUSPEND, do not set UEM_NOERROR here
_EXIT:
	return result;
}


uem_result UKChannel_GetNumOfAvailableData (IN int nChannelId, IN int nChunkIndex, OUT int *pnDataNum)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nIndex = 0;
	SChannelAPI *pstChannelAPI = NULL;

	nIndex = getChannelIndexById(nChannelId);
	IFVARERRASSIGNGOTO(nIndex, INVALID_CHANNEL_ID, result, ERR_UEM_INVALID_PARAM, _EXIT);

	result = getAPIStructureFromCommunicationType(g_astChannels[nIndex].enType, &pstChannelAPI);
	ERRIFGOTO(result, _EXIT);

	result = pstChannelAPI->fnGetNumOfAvailableData(&g_astChannels[nIndex], nChunkIndex, pnDataNum);
	ERRIFGOTO(result, _EXIT);

	// to preserve, ERR_UEM_SUSPEND, do not set UEM_NOERROR here
_EXIT:
	return result;
}

uem_result UKChannel_Clear(IN int nChannelId)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nIndex = 0;
	SChannelAPI *pstChannelAPI = NULL;

	nIndex = getChannelIndexById(nChannelId);
	IFVARERRASSIGNGOTO(nIndex, INVALID_CHANNEL_ID, result, ERR_UEM_INVALID_PARAM, _EXIT);

	result = getAPIStructureFromCommunicationType(g_astChannels[nIndex].enType, &pstChannelAPI);
	ERRIFGOTO(result, _EXIT);

	result = pstChannelAPI->fnClear(&g_astChannels[nIndex]);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}



uem_result UKChannel_GetAvailableIndex (IN int nChannelId, OUT int *pnChunkIndex)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nIndex = 0;
	SChannelAPI *pstChannelAPI = NULL;

	nIndex = getChannelIndexById(nChannelId);
	IFVARERRASSIGNGOTO(nIndex, INVALID_CHANNEL_ID, result, ERR_UEM_INVALID_PARAM, _EXIT);

	result = getAPIStructureFromCommunicationType(g_astChannels[nIndex].enType, &pstChannelAPI);
	ERRIFGOTO(result, _EXIT);

	result = pstChannelAPI->fnGetAvailableChunk(&g_astChannels[nIndex], pnChunkIndex);
	ERRIFGOTO(result, _EXIT);

	// to preserve, ERR_UEM_SUSPEND, do not set UEM_NOERROR here
_EXIT:
	return result;
}

uem_result UKChannel_SetExit()
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;
	SChannelAPI *pstChannelAPI = NULL;

	for(nLoop = 0; nLoop < g_nChannelNum; nLoop++)
	{
		result = getAPIStructureFromCommunicationType(g_astChannels[nLoop].enType, &pstChannelAPI);
		if(result == ERR_UEM_NOERROR)
		{
			pstChannelAPI->fnSetExit(&(g_astChannels[nLoop]));
		}
	}

	return result;
}


uem_result UKChannel_Finalize()
{
	uem_result result = ERR_UEM_NOERROR;
	int nLoop = 0;
	SChannelAPI *pstChannelAPI = NULL;

	for(nLoop = 0; nLoop < g_nChannelNum; nLoop++)
	{
		result = getAPIStructureFromCommunicationType(g_astChannels[nLoop].enType, &pstChannelAPI);

		if(result == ERR_UEM_NOERROR)
		{
			pstChannelAPI->fnFinalize(&(g_astChannels[nLoop]));
		}
	}

	return result;
}


