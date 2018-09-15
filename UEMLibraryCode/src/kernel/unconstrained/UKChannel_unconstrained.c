/*
 * UKChannel_unconstrained.c
 *
 *  Created on: 2017. 8. 28.
 *      Author: jej
 *
 *  - This file is changed from UKChannel.c to UKChannel_unconstrained.c
 */


#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <UCString.h>

#include <uem_data.h>
#include <uem_channel_data.h>

#include <UKTask_internal.h>


uem_result ChannelAPI_GetAPIStructureFromCommunicationType(IN ECommunicationType enType, OUT SChannelAPI **ppstChannelAPI);

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

	pstInputPort = pstChannel->pstInputPort;
	pstOutputPort = pstChannel->pstOutputPort;

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
		if(matchTaskIdInPort(g_astChannels[nLoop].pstInputPort, nTaskId) == TRUE)
		{
			if(g_astChannels[nLoop].pstInputPort->nNumOfSampleRates == 1)
			{
				bAvailable = TRUE;
				break;
			}

			for(nInLoop = 0 ; nInLoop < g_astChannels[nLoop].pstInputPort->nNumOfSampleRates; nInLoop++)
			{
				result = UCString_New(&stPortModeName, (char *) g_astChannels[nLoop].pstInputPort->astSampleRates[nInLoop].pszModeName, UEMSTRING_CONST);
				ERRIFGOTO(result, _EXIT);

				if(UCString_IsEqual(&stModeName, &stPortModeName) == TRUE &&
					g_astChannels[nLoop].pstInputPort->astSampleRates[nInLoop].nSampleRate > 0)
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
		if(matchTaskIdInPort(g_astChannels[nLoop].pstInputPort, nTaskId) == TRUE)
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
				if(matchTaskIdInPort(g_astChannels[nLoop].pstInputPort, nTaskId) == TRUE)
				{
					pstChannelAPI->fnSetExit(&(g_astChannels[nLoop]), EXIT_FLAG_READ);
				}
				else if(matchTaskIdInPort(g_astChannels[nLoop].pstOutputPort, nTaskId) == TRUE)
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
				if(matchTaskIdInPort(g_astChannels[nLoop].pstInputPort, nTaskId) == TRUE)
				{
					pstChannelAPI->fnClearExit(&(g_astChannels[nLoop]), EXIT_FLAG_READ);
				}
				else if(matchTaskIdInPort(g_astChannels[nLoop].pstOutputPort, nTaskId) == TRUE)
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
			if(matchTaskIdInPort(g_astChannels[nLoop].pstOutputPort, nTaskId) == TRUE)
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


