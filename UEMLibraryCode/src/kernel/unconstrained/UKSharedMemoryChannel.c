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

#include <UKChannelMemory.h>
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


uem_result UKSharedMemoryChannel_Clear(SChannel *pstChannel)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSharedMemoryChannel *pstSharedMemoryChannel = NULL;

	pstSharedMemoryChannel = (SSharedMemoryChannel *) pstChannel->pChannelStruct;

	result = UKChannelMemory_Clear(pstChannel, pstSharedMemoryChannel);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKSharedMemoryChannel_Initialize(SChannel *pstChannel)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSharedMemoryChannel *pstSharedMemoryChannel = NULL;

	pstSharedMemoryChannel = (SSharedMemoryChannel *) pstChannel->pChannelStruct;

	result = UKChannelMemory_Initialize(pstChannel, pstSharedMemoryChannel);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKSharedMemoryChannel_ReadFromQueue(SChannel *pstChannel, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSharedMemoryChannel *pstSharedMemoryChannel = NULL;

	pstSharedMemoryChannel = (SSharedMemoryChannel *) pstChannel->pChannelStruct;

	result = UKChannelMemory_ReadFromQueue(pstChannel, pstSharedMemoryChannel, pBuffer, nDataToRead, nChunkIndex, pnDataRead);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKSharedMemoryChannel_ReadFromBuffer(SChannel *pstChannel, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSharedMemoryChannel *pstSharedMemoryChannel = NULL;

	pstSharedMemoryChannel = (SSharedMemoryChannel *) pstChannel->pChannelStruct;

	result = UKChannelMemory_ReadFromBuffer(pstChannel, pstSharedMemoryChannel, pBuffer, nDataToRead, nChunkIndex, pnDataRead);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKSharedMemoryChannel_WriteToQueue (SChannel *pstChannel, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSharedMemoryChannel *pstSharedMemoryChannel = NULL;

	pstSharedMemoryChannel = (SSharedMemoryChannel *) pstChannel->pChannelStruct;

	result = UKChannelMemory_WriteToQueue(pstChannel, pstSharedMemoryChannel, pBuffer, nDataToWrite, nChunkIndex, pnDataWritten);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKSharedMemoryChannel_WriteToBuffer (SChannel *pstChannel, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSharedMemoryChannel *pstSharedMemoryChannel = NULL;

	pstSharedMemoryChannel = (SSharedMemoryChannel *) pstChannel->pChannelStruct;

	result = UKChannelMemory_WriteToBuffer(pstChannel, pstSharedMemoryChannel, pBuffer, nDataToWrite, nChunkIndex, pnDataWritten);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKSharedMemoryChannel_GetAvailableChunk (SChannel *pstChannel, OUT int *pnChunkIndex)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSharedMemoryChannel *pstSharedMemoryChannel = NULL;

	pstSharedMemoryChannel = (SSharedMemoryChannel *) pstChannel->pChannelStruct;

	result = UKChannelMemory_GetAvailableChunk(pstChannel, pstSharedMemoryChannel, pnChunkIndex);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKSharedMemoryChannel_GetNumOfAvailableData (SChannel *pstChannel, IN int nChunkIndex, OUT int *pnDataNum)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSharedMemoryChannel *pstSharedMemoryChannel = NULL;

	pstSharedMemoryChannel = (SSharedMemoryChannel *) pstChannel->pChannelStruct;

	result = UKChannelMemory_GetNumOfAvailableData(pstChannel, pstSharedMemoryChannel, nChunkIndex, pnDataNum);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKSharedMemoryChannel_SetExit(SChannel *pstChannel, int nExitFlag)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSharedMemoryChannel *pstSharedMemoryChannel = NULL;

	pstSharedMemoryChannel = (SSharedMemoryChannel *) pstChannel->pChannelStruct;

	result = UKChannelMemory_SetExit(pstChannel, pstSharedMemoryChannel, nExitFlag);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKSharedMemoryChannel_ClearExit(SChannel *pstChannel, int nExitFlag)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSharedMemoryChannel *pstSharedMemoryChannel = NULL;

	pstSharedMemoryChannel = (SSharedMemoryChannel *) pstChannel->pChannelStruct;

	result = UKChannelMemory_ClearExit(pstChannel, pstSharedMemoryChannel, nExitFlag);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKSharedMemoryChannel_FillInitialData(SChannel *pstChannel)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSharedMemoryChannel *pstSharedMemoryChannel = NULL;

	pstSharedMemoryChannel = (SSharedMemoryChannel *) pstChannel->pChannelStruct;

	result = UKChannelMemory_FillInitialData(pstChannel, pstSharedMemoryChannel);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKSharedMemoryChannel_Finalize(SChannel *pstChannel)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSharedMemoryChannel *pstSharedMemoryChannel = NULL;

	pstSharedMemoryChannel = (SSharedMemoryChannel *) pstChannel->pChannelStruct;

	result = UKChannelMemory_Finalize(pstChannel, pstSharedMemoryChannel);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}






