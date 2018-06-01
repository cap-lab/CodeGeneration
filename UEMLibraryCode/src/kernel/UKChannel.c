/*
 * UKChannel.c
 *
 *  Created on: 2017. 8. 28.
 *      Author: jej
 */


#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <UCBasic.h>
#include <UCString.h>

#include <uem_data.h>
#include <UKTask.h>
#include <UKSharedMemoryChannel.h>

typedef uem_result (*FnChannelInitialize)(SChannel *pstChannel);
typedef uem_result (*FnChannelReadFromQueue)(SChannel *pstChannel, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead);
typedef uem_result (*FnChannelReadFromBuffer)(SChannel *pstChannel, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead);
typedef uem_result (*FnChannelWriteToBuffer)(SChannel *pstChannel, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten);
typedef uem_result (*FnChannelWriteToQueue)(SChannel *pstChannel, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten);
typedef uem_result (*FnChannelGetAvailableChunk)(SChannel *pstChannel, OUT int *pnChunkIndex);
typedef uem_result (*FnChannelGetNumOfAvailableData)(SChannel *pstChannel, IN int nChunkIndex, OUT int *pnDataNum);
typedef uem_result (*FnChannelClear)(SChannel *pstChannel);
typedef uem_result (*FnChannelSetExit)(SChannel *pstChannel, int nExitFlag);
typedef uem_result (*FnChannelClearExit)(SChannel *pstChannel, int nExitFlag);
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
	FnChannelClearExit fnClearExit;
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
	UKSharedMemoryChannel_ClearExit,
	UKSharedMemoryChannel_Finalize, // fnFinalize
};

/*
SChannelAPI g_stSharedGPUMemoryChannel = {
	UKGPUSharedMemoryChannel_Initialize, // fnInitialize
	UKGPUSharedMemoryChannel_ReadFromQueue, // fnReadFromQueue
	NULL, // fnReadFromBuffer
	UKGPUSharedMemoryChannel_WriteToQueue, // fnWriteToQueue
	NULL, // fnWriteToBuffer
	NULL, // fnGetAvailableChunk
	UKGPUSharedMemoryChannel_GetNumOfAvailableData, // fnGetNumOfAvailableData
	UKGPUSharedMemoryChannel_Clear, // fnClear
	UKGPUSharedMemoryChannel_SetExit,
	UKGPUSharedMemoryChannel_ClearExit,
	UKGPUSharedMemoryChannel_Finalize, // fnFinalize
};*/

#define DEFAUT_INITIAL_BUF_SIZE (4)


static uem_result getAPIStructureFromCommunicationType(IN ECommunicationType enType, OUT SChannelAPI **ppstChannelAPI)
{
	uem_result result = ERR_UEM_UNKNOWN;
	switch(enType)
	{
	case COMMUNICATION_TYPE_SHARED_MEMORY:
		*ppstChannelAPI = &g_stSharedMemoryChannel;
		break;
	case COMMUNICATION_TYPE_TCP_SERVER_READER:
	case COMMUNICATION_TYPE_TCP_SERVER_WRITER:
	case COMMUNICATION_TYPE_TCP_CLIENT_READER:
	case COMMUNICATION_TYPE_TCP_CLIENT_WRITER:
		ERRASSIGNGOTO(result, ERR_UEM_NOT_SUPPORTED_YET, _EXIT)
		break;
	case COMMUNICATION_TYPE_CPU_GPU:
	case COMMUNICATION_TYPE_GPU_CPU:
	case COMMUNICATION_TYPE_GPU_GPU:
	case COMMUNICATION_TYPE_GPU_GPU_DIFFERENT:
		*ppstChannelAPI = &g_stSharedMemoryChannel;
//		*ppstChannelAPI = &g_stSharedGPUMemoryChannel;
		break;
	default:
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT)
		break;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result fillInitialData(SChannel *pstChannel, SChannelAPI *pstChannelAPI)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nBuffer = 0; // 4-byte buffer
	int nTotalDataToWrite = pstChannel->nInitialDataLen;
	int nDataWritten = 0;
	int nTotalDataWritten = 0;

	if(pstChannel->nInitialDataLen > pstChannel->nBufSize)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
	}

	while(nTotalDataWritten < nTotalDataToWrite)
	{
		result = pstChannelAPI->fnWriteToQueue(pstChannel, (unsigned char *) &nBuffer, MIN(nTotalDataToWrite - nTotalDataWritten, sizeof(int)), 0, &nDataWritten);
		ERRIFGOTO(result, _EXIT);
		nTotalDataWritten += nDataWritten;
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

		if(g_astChannels[nLoop].nInitialDataLen > 0)
		{
			result = fillInitialData(&(g_astChannels[nLoop]), pstChannelAPI);
			ERRIFGOTO(result, _EXIT);
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

static uem_bool isPortTaskIdAndPortNameEqual(SPort *pstTopPort, uem_string strPortName, int nTaskId)
{
	uem_string_struct stStructPortName;
	uem_result result = ERR_UEM_UNKNOWN;
	uem_bool bIsMatch = FALSE;
	SPort *pstPort = NULL;

	pstPort = pstTopPort;

	while(pstPort != NULL)
	{
		if(pstPort->nTaskId == nTaskId)
		{
			result = UCString_New(&stStructPortName, (char *) pstTopPort->pszPortName, UEMSTRING_CONST);
			ERRIFGOTO(result, _EXIT);

			if(UCString_IsEqual(strPortName, &stStructPortName) == TRUE)
			{
				bIsMatch = TRUE;
				break;
			}
		}
		pstPort = pstPort->pstSubGraphPort;
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

	result = UCString_New(&stArgPortName, szPortName, UEMSTRING_CONST);
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
#ifdef ARGUMENT_CHECK
	if(nChunkIndex < 0 )
	{
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
#endif
	nIndex = getChannelIndexById(nChannelId);
	IFVARERRASSIGNGOTO(nIndex, INVALID_CHANNEL_ID, result, ERR_UEM_INVALID_PARAM, _EXIT);

	while(nIndex >= 0)
	{
		result = getAPIStructureFromCommunicationType(g_astChannels[nIndex].enType, &pstChannelAPI);
		ERRIFGOTO(result, _EXIT);

		result = pstChannelAPI->fnWriteToBuffer(&g_astChannels[nIndex], pBuffer, nDataToWrite, nChunkIndex, pnDataWritten);
		ERRIFGOTO(result, _EXIT);

		nIndex = getChannelIndexById(g_astChannels[nIndex].nNextChannelIndex);
	}

	// to preserve, ERR_UEM_SUSPEND, do not set UEM_NOERROR here
_EXIT:
	return result;
}


uem_result UKChannel_WriteToQueue(int nChannelId, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nIndex = 0;
	SChannelAPI *pstChannelAPI = NULL;
#ifdef ARGUMENT_CHECK
	if(nChunkIndex < 0 )
	{
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
#endif
	nIndex = getChannelIndexById(nChannelId);
	IFVARERRASSIGNGOTO(nIndex, INVALID_CHANNEL_ID, result, ERR_UEM_INVALID_PARAM, _EXIT);

	while(nIndex >= 0)
	{
		result = getAPIStructureFromCommunicationType(g_astChannels[nIndex].enType, &pstChannelAPI);
		ERRIFGOTO(result, _EXIT);

		result = pstChannelAPI->fnWriteToQueue(&g_astChannels[nIndex], pBuffer, nDataToWrite, nChunkIndex, pnDataWritten);
		ERRIFGOTO(result, _EXIT);

		nIndex = getChannelIndexById(g_astChannels[nIndex].nNextChannelIndex);
	}

	// to preserve, ERR_UEM_SUSPEND, do not set UEM_NOERROR here
_EXIT:
	return result;
}


uem_result UKChannel_ReadFromQueue(int nChannelId, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nIndex = 0;
	SChannelAPI *pstChannelAPI = NULL;
#ifdef ARGUMENT_CHECK
	if(nChunkIndex < 0 )
	{
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
#endif
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
#ifdef ARGUMENT_CHECK
	if(nChunkIndex < 0 )
	{
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
#endif
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
#ifdef ARGUMENT_CHECK
	if(nChunkIndex < 0 )
	{
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
#endif
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

	if(g_astChannels[nIndex].nInitialDataLen > 0)
	{
		result = fillInitialData(&(g_astChannels[nIndex]), pstChannelAPI);
		ERRIFGOTO(result, _EXIT);
	}

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
			pstChannelAPI->fnSetExit(&(g_astChannels[nLoop]), EXIT_FLAG_READ | EXIT_FLAG_WRITE);
		}
	}

	return result;
}

static uem_bool matchTaskIdInPort(SPort *pstPort, int nTaskId)
{
	uem_bool bIsMatched = FALSE;

	while(pstPort != NULL)
	{
		if(pstPort->nTaskId == nTaskId)
		{
			bIsMatched = TRUE;
			break;
		}

		pstPort = pstPort->pstSubGraphPort;
	}

	return bIsMatched;
}

static uem_bool isChannelLocatedInSameTaskGraph(SChannel *pstChannel)
{
	SPort *pstInputPort = NULL;
	SPort *pstOutputPort = NULL;
	uem_bool bShareSameTaskGraph = FALSE;

	pstInputPort = &(pstChannel->stInputPort);
	pstOutputPort = &(pstChannel->stOutputPort);

	while (pstInputPort != NULL && pstOutputPort != NULL)
	{
		if(pstInputPort->nTaskId != pstOutputPort->nTaskId) // input and output task is different
		{
			// last node?
			if(pstInputPort->pstSubGraphPort == NULL && pstOutputPort->pstSubGraphPort == NULL)
			{
				bShareSameTaskGraph = TRUE;
			}
			break;
		}

		pstInputPort = pstInputPort->pstSubGraphPort;
		pstOutputPort = pstOutputPort->pstSubGraphPort;
	}

	return bShareSameTaskGraph;
}


uem_bool UKChannel_IsPortRateAvailableTask(int nTaskId, char *pszModeName)
{
	int nLoop = 0;
	int nInLoop = 0;
	uem_bool bAvailable = FALSE;
	uem_string_struct stModeName;
	uem_string_struct stPortModeName;
	uem_result result;

	result = UCString_New(&stModeName, pszModeName, UEMSTRING_CONST);
	ERRIFGOTO(result, _EXIT);

	for(nLoop = 0; nLoop < g_nChannelNum; nLoop++)
	{
		if(matchTaskIdInPort(&(g_astChannels[nLoop].stInputPort), nTaskId) == TRUE)
		{
			if(g_astChannels[nLoop].stInputPort.nNumOfSampleRates == 1)
			{
				bAvailable = TRUE;
				break;
			}

			for(nInLoop = 0 ; nInLoop < g_astChannels[nLoop].stInputPort.nNumOfSampleRates; nInLoop++)
			{
				result = UCString_New(&stPortModeName, (char *) g_astChannels[nLoop].stInputPort.astSampleRates[nInLoop].pszModeName, UEMSTRING_CONST);
				ERRIFGOTO(result, _EXIT);

				if(UCString_IsEqual(&stModeName, &stPortModeName) == TRUE &&
					g_astChannels[nLoop].stInputPort.astSampleRates[nInLoop].nSampleRate > 0)
				{
					bAvailable = TRUE;
					break;
				}
			}

			if(bAvailable == TRUE)
			{
				break;
			}
		}
	}
_EXIT:
	return bAvailable;
}

// all matched input must have data more than sample rate
uem_bool UKChannel_IsTaskSourceTask(int nTaskId)
{
	int nLoop = 0;
	uem_bool bIsLocatedInSameTaskGraph = FALSE;
	uem_bool bIsSourceTask = TRUE;

	for(nLoop = 0; nLoop < g_nChannelNum; nLoop++)
	{
		if(matchTaskIdInPort(&(g_astChannels[nLoop].stInputPort), nTaskId) == TRUE)
		{
			bIsLocatedInSameTaskGraph = isChannelLocatedInSameTaskGraph(&(g_astChannels[nLoop]));

			if(bIsLocatedInSameTaskGraph == TRUE)
			{
				bIsSourceTask = FALSE;
				break;
			}
		}
	}

	return bIsSourceTask;
}


uem_result UKChannel_SetExitByTaskId(int nTaskId)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;
	SChannelAPI *pstChannelAPI = NULL;

	for(nLoop = 0; nLoop < g_nChannelNum; nLoop++)
	{
		result = getAPIStructureFromCommunicationType(g_astChannels[nLoop].enType, &pstChannelAPI);
		if(result == ERR_UEM_NOERROR)
		{
			if(matchTaskIdInPort(&(g_astChannels[nLoop].stInputPort), nTaskId) == TRUE)
			{
				pstChannelAPI->fnSetExit(&(g_astChannels[nLoop]), EXIT_FLAG_READ);
			}
			else if(matchTaskIdInPort(&(g_astChannels[nLoop].stOutputPort), nTaskId) == TRUE)
			{
				pstChannelAPI->fnSetExit(&(g_astChannels[nLoop]), EXIT_FLAG_WRITE);
			}
			else
			{
				// no match
			}
		}
	}

	return result;
}


uem_result UKChannel_ClearExitByTaskId(int nTaskId)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;
	SChannelAPI *pstChannelAPI = NULL;

	for(nLoop = 0; nLoop < g_nChannelNum; nLoop++)
	{
		result = getAPIStructureFromCommunicationType(g_astChannels[nLoop].enType, &pstChannelAPI);
		if(result == ERR_UEM_NOERROR)
		{
			if(matchTaskIdInPort(&(g_astChannels[nLoop].stInputPort), nTaskId) == TRUE)
			{
				pstChannelAPI->fnClearExit(&(g_astChannels[nLoop]), EXIT_FLAG_READ);
			}
			else if(matchTaskIdInPort(&(g_astChannels[nLoop].stOutputPort), nTaskId) == TRUE)
			{
				pstChannelAPI->fnClearExit(&(g_astChannels[nLoop]), EXIT_FLAG_WRITE);
			}
			else
			{
				// no match
			}
		}
	}

	return result;
}

static uem_bool matchIsSubgraphPort(SPort *pstPort, int nParentTaskId)
{
	uem_bool bIsMatched = FALSE;

	while(pstPort != NULL)
	{
		if(UKTask_isParentTask(pstPort->nTaskId, nParentTaskId) == TRUE)
		{
			bIsMatched = TRUE;
			break;
		}

		pstPort = pstPort->pstSubGraphPort;
	}

	return bIsMatched;
}

uem_result UKChannel_ClearChannelInSubgraph(int nParentTaskId)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;
	SChannelAPI *pstChannelAPI = NULL;

	for(nLoop = 0; nLoop < g_nChannelNum; nLoop++)
	{
		result = getAPIStructureFromCommunicationType(g_astChannels[nLoop].enType, &pstChannelAPI);
		ERRIFGOTO(result, _EXIT);

		if(matchIsSubgraphPort(&(g_astChannels[nLoop].stInputPort), nParentTaskId) == TRUE &&
			matchIsSubgraphPort(&(g_astChannels[nLoop].stOutputPort), nParentTaskId) == TRUE)
		{ // this channel is located in subgraph

			result = pstChannelAPI->fnClear(&(g_astChannels[nLoop]));
			ERRIFGOTO(result, _EXIT);

			if(g_astChannels[nLoop].nInitialDataLen > 0)
			{
				result = fillInitialData(&(g_astChannels[nLoop]), pstChannelAPI);
				ERRIFGOTO(result, _EXIT);
			}
		}
		else
		{
			// no match
		}

	}
	result = ERR_UEM_NOERROR;
_EXIT:
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


