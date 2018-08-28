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
#include <UKTask_internal.h>

#include <UKChannel.h>


#define DEFAUT_INITIAL_BUF_SIZE (4)




uem_result ChannelAPI_GetAPIStructureFromCommunicationType(IN ECommunicationType enType, OUT SChannelAPI **ppstChannelAPI);


uem_result UKChannel_Initialize()
{
	uem_result result = ERR_UEM_NOERROR;
	int nLoop = 0;
	SChannelAPI *pstChannelAPI = NULL;

	for(nLoop = 0; nLoop < g_nChannelAPINum ; nLoop++)
	{
		if(g_astChannelAPIList[nLoop]->fnAPIInitialize != NULL)
		{
			result = g_astChannelAPIList[nLoop]->fnAPIInitialize();
			ERRIFGOTO(result, _EXIT);
		}
	}

	for(nLoop = 0; nLoop < g_nChannelNum; nLoop++)
	{
		result = ChannelAPI_GetAPIStructureFromCommunicationType(g_astChannels[nLoop].enType, &pstChannelAPI);
		ERRIFGOTO(result, _EXIT);

		if(pstChannelAPI->fnInitialize != NULL)
		{
			result = pstChannelAPI->fnInitialize(&(g_astChannels[nLoop]));
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
			result = UCString_New(&stStructPortName, (char *) pstPort->pszPortName, UEMSTRING_CONST);
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
			nIndex = g_astChannels[nLoop].nChannelIndex;
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
		result = ChannelAPI_GetAPIStructureFromCommunicationType(g_astChannels[nIndex].enType, &pstChannelAPI);
		ERRIFGOTO(result, _EXIT);

		if(pstChannelAPI->fnWriteToBuffer == NULL)
		{
			ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_CONTROL, _EXIT);
		}

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
		result = ChannelAPI_GetAPIStructureFromCommunicationType(g_astChannels[nIndex].enType, &pstChannelAPI);
		ERRIFGOTO(result, _EXIT);

		if(pstChannelAPI->fnWriteToQueue == NULL)
		{
			ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_CONTROL, _EXIT);
		}

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

	result = ChannelAPI_GetAPIStructureFromCommunicationType(g_astChannels[nIndex].enType, &pstChannelAPI);
	ERRIFGOTO(result, _EXIT);

	if(pstChannelAPI->fnReadFromQueue == NULL)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_CONTROL, _EXIT);
	}

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

	result = ChannelAPI_GetAPIStructureFromCommunicationType(g_astChannels[nIndex].enType, &pstChannelAPI);
	ERRIFGOTO(result, _EXIT);

	if(pstChannelAPI->fnReadFromBuffer == NULL)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_CONTROL, _EXIT);
	}

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

	result = ChannelAPI_GetAPIStructureFromCommunicationType(g_astChannels[nIndex].enType, &pstChannelAPI);
	ERRIFGOTO(result, _EXIT);

	if(pstChannelAPI->fnGetNumOfAvailableData == NULL)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_CONTROL, _EXIT);
	}

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

	result = ChannelAPI_GetAPIStructureFromCommunicationType(g_astChannels[nIndex].enType, &pstChannelAPI);
	ERRIFGOTO(result, _EXIT);

	if(pstChannelAPI->fnClear == NULL)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_CONTROL, _EXIT);
	}

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

	result = ChannelAPI_GetAPIStructureFromCommunicationType(g_astChannels[nIndex].enType, &pstChannelAPI);
	ERRIFGOTO(result, _EXIT);

	if(pstChannelAPI->fnGetAvailableChunk == NULL)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_CONTROL, _EXIT);
	}

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
		result = ChannelAPI_GetAPIStructureFromCommunicationType(g_astChannels[nLoop].enType, &pstChannelAPI);
		if(result == ERR_UEM_NOERROR)
		{
			if(pstChannelAPI->fnSetExit != NULL)
			{
				pstChannelAPI->fnSetExit(&(g_astChannels[nLoop]), EXIT_FLAG_READ | EXIT_FLAG_WRITE);
			}
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
		result = ChannelAPI_GetAPIStructureFromCommunicationType(g_astChannels[nLoop].enType, &pstChannelAPI);
		if(result == ERR_UEM_NOERROR)
		{
			if(pstChannelAPI->fnSetExit != NULL)
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
		result = ChannelAPI_GetAPIStructureFromCommunicationType(g_astChannels[nLoop].enType, &pstChannelAPI);
		if(result == ERR_UEM_NOERROR)
		{
			if(pstChannelAPI->fnClearExit != NULL)
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
		result = ChannelAPI_GetAPIStructureFromCommunicationType(g_astChannels[nLoop].enType, &pstChannelAPI);
		ERRIFGOTO(result, _EXIT);

		if(pstChannelAPI->fnClear == NULL)
		{
			ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_CONTROL, _EXIT);
		}

		if(matchIsSubgraphPort(&(g_astChannels[nLoop].stInputPort), nParentTaskId) == TRUE &&
			matchIsSubgraphPort(&(g_astChannels[nLoop].stOutputPort), nParentTaskId) == TRUE)
		{ // this channel is located in subgraph

			result = pstChannelAPI->fnClear(&(g_astChannels[nLoop]));
			ERRIFGOTO(result, _EXIT);
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


uem_result UKChannel_FillInitialDataBySourceTaskId(int nTaskId)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;
	SChannelAPI *pstChannelAPI = NULL;

	for(nLoop = 0; nLoop < g_nChannelNum; nLoop++)
	{
		result = ChannelAPI_GetAPIStructureFromCommunicationType(g_astChannels[nLoop].enType, &pstChannelAPI);
		if(result == ERR_UEM_NOERROR && pstChannelAPI->fnFillInitialData != NULL)
		{
			if(matchTaskIdInPort(&(g_astChannels[nLoop].stOutputPort), nTaskId) == TRUE)
			{
				result = pstChannelAPI->fnFillInitialData(&(g_astChannels[nLoop]));
				ERRIFGOTO(result, _EXIT);
			}
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

	// set TRUE for exiting other finalize codes
	g_bSystemExit = TRUE;

	for(nLoop = 0; nLoop < g_nChannelNum; nLoop++)
	{
		result = ChannelAPI_GetAPIStructureFromCommunicationType(g_astChannels[nLoop].enType, &pstChannelAPI);

		if(result == ERR_UEM_NOERROR)
		{
			if(pstChannelAPI->fnFinalize != NULL)
			{
				pstChannelAPI->fnFinalize(&(g_astChannels[nLoop]));
			}
		}
	}

	for(nLoop = 0; nLoop < g_nChannelAPINum ; nLoop++)
	{
		if(g_astChannelAPIList[nLoop]->fnAPIFinalize != NULL)
		{
			g_astChannelAPIList[nLoop]->fnAPIFinalize();
			// ignore error
		}
	}

	return result;
}


