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

static uem_result getClosestParentDTypeLoopTask(IN STask* pstCurTask, OUT STask** ppstParentTask)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstTask = NULL;

	IFVARERRASSIGNGOTO(ppstParentTask, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	pstTask = pstCurTask->pstParentGraph->pstParentTask;
	while(pstTask != NULL)
	{
		if(pstTask->pstLoopInfo != NULL && pstTask->pstLoopInfo->enType == LOOP_TYPE_DATA)//if parent task is looptype task
		{
			break;
		}
		else
		{
			pstTask = pstTask->pstParentGraph->pstParentTask;
		}
	}

	if(pstTask == NULL || pstTask->pstLoopInfo == NULL || pstTask->pstLoopInfo->enType != LOOP_TYPE_DATA)
	{
		*ppstParentTask = NULL;
	}
	else
	{
		*ppstParentTask = pstTask;
	}
	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static uem_result calculateNChunkNumInsideDTypeLoopTask(STask* pstCurTask, EChannelType enChannelType, OUT int *pnChunkNum)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask* pstParentTask = NULL;

	result = getClosestParentDTypeLoopTask(pstCurTask, &pstParentTask);
	ERRIFGOTO(result, _EXIT);

	if(pstParentTask != NULL)
	{
		if(enChannelType == CHANNEL_TYPE_FULL_ARRAY)
		{
			*pnChunkNum = pstCurTask->nTaskThreadSetNum;
		}
		else if(enChannelType == CHANNEL_TYPE_INPUT_ARRAY || enChannelType == CHANNEL_TYPE_OUTPUT_ARRAY)
		{
			// do nothing
		}
		else //channel connected with task inside DTypeLoopTask cannnot be CHANNEL_TYPE_GENERAL.
		{				
			ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
		}
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKChannel_GetChunkNumAndLen(SPort *pstPort, OUT int *pnChunkNum, OUT int *pnChunkLen, IN EChannelType enChannelType)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nCurrentSampleRateIndex = 0;
	SPort *pstMostInnerPort = NULL;
	int nOuterMostSampleRate = 0;
	STask *pstCurTask = NULL;
	int nChunkNum = 0;
	int nChunkLen = 0;

	IFVARERRASSIGNGOTO(pnChunkNum, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	IFVARERRASSIGNGOTO(pnChunkLen, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	nCurrentSampleRateIndex = pstPort->nCurrentSampleRateIndex;

	if(pstPort->nNumOfSampleRates > 0)
	{
		nOuterMostSampleRate = pstPort->astSampleRates[nCurrentSampleRateIndex].nSampleRate;
	}

	pstMostInnerPort = pstPort;

	while(pstMostInnerPort->pstSubGraphPort != NULL)
	{
		pstMostInnerPort = pstMostInnerPort->pstSubGraphPort;
	}

	result = UKTask_GetTaskFromTaskId(pstMostInnerPort->nTaskId, &pstCurTask);
	if(result == ERR_UEM_NO_DATA)
	{
		result = ERR_UEM_NOERROR;
	}
	ERRIFGOTO(result, _EXIT);


	if(pstPort != pstMostInnerPort)
	{
		nCurrentSampleRateIndex = pstMostInnerPort->nCurrentSampleRateIndex;

		if(pstMostInnerPort->nNumOfSampleRates > 0)
		{
			nChunkNum = nOuterMostSampleRate / pstMostInnerPort->astSampleRates[nCurrentSampleRateIndex].nSampleRate;
			nChunkLen = pstMostInnerPort->astSampleRates[nCurrentSampleRateIndex].nSampleRate * pstMostInnerPort->nSampleSize;
		}
		else
		{
			//not used.
			nChunkNum = 1;
			nChunkLen = nOuterMostSampleRate * pstPort->nSampleSize;
		}
	}
	else
	{
		// If the task id cannot be obtained by "UKTask_GetTaskFromTaskId",
		// it means the information is missing because of remote communication, so set as a general task information

		if(pstCurTask == NULL || pstCurTask->pstLoopInfo == NULL) // general task
		{
			nChunkNum = 1;
			nChunkLen = nOuterMostSampleRate * pstPort->nSampleSize;
		}
		else if(pstCurTask->pstLoopInfo->enType == LOOP_TYPE_DATA &&
			pstPort->astSampleRates[nCurrentSampleRateIndex].nMaxAvailableDataNum == 1)
		{
			nChunkNum = nOuterMostSampleRate / (nOuterMostSampleRate / pstCurTask->pstLoopInfo->nLoopCount);
			nChunkLen = (nOuterMostSampleRate * pstPort->nSampleSize) / nChunkNum;
		}
		else // broadcasting or convergent
		{
			nChunkNum = 1;
			nChunkLen = nOuterMostSampleRate * pstPort->nSampleSize;
		}
	}

	if(pstCurTask != NULL) //Task care only for tasks in current device.
	{
		result = calculateNChunkNumInsideDTypeLoopTask(pstCurTask, enChannelType, &nChunkNum);
		ERRIFGOTO(result, _EXIT);
	}

	*pnChunkNum = nChunkNum;
	*pnChunkLen = nChunkLen;

	result = ERR_UEM_NOERROR;
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

	result = ERR_UEM_NOERROR;

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

	result = ERR_UEM_NOERROR;

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

	result = ERR_UEM_NOERROR;

	return result;
}


static uem_result popDataFromQueue(SChannel *pstChannel, SChannelAPI *pstChannelAPI, int nNumOfDataToPop)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nChunkNum = 0;
	int nChunkLen = 0;
	int nDataRead = 0;
	int nLoop = 0;
	int nSubLoop = 0;
	int nDataCanRead = 0;
	int nBroadcastNum = 0;

	result = UKChannel_GetChunkNumAndLen(pstChannel->pstInputPort, &nChunkNum, &nChunkLen, pstChannel->enChannelType);
	ERRIFGOTO(result, _EXIT);

	nBroadcastNum = nNumOfDataToPop / nChunkNum;

	// if the buffer pointer is NULL, no copy is required
	for(nLoop = 0 ; nLoop < nBroadcastNum ; nLoop++)
	{
		for(nSubLoop = 0 ; nSubLoop < nChunkNum ; nSubLoop++)
		{
			result = pstChannelAPI->fnGetNumOfAvailableData(pstChannel, nSubLoop, &nDataCanRead);
			ERRIFGOTO(result, _EXIT);

			if(nDataCanRead >= nDataRead)
			{
				result = pstChannelAPI->fnReadFromQueue(pstChannel, NULL, nChunkLen, nSubLoop, &nDataRead);
				ERRIFGOTO(result, _EXIT);
			}
		}
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKChannel_PopLoopTaskBroadcastingDataFromQueueByTaskId(int nLoopTaskId, int nTaskId, int nNumOfDataToPop)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;
	SChannelAPI *pstChannelAPI = NULL;

	for(nLoop = 0; nLoop < g_nChannelNum; nLoop++)
	{
		result = ChannelAPI_GetAPIStructureFromCommunicationType(g_astChannels[nLoop].enType, &pstChannelAPI);
		if(result == ERR_UEM_NOERROR && pstChannelAPI->fnGetNumOfAvailableData != NULL && pstChannelAPI->fnReadFromQueue != NULL)
		{
			if(matchTaskIdInPort(g_astChannels[nLoop].pstInputPort, nTaskId) == TRUE)
			{
				if(matchTaskIdInPort(g_astChannels[nLoop].pstInputPort, nLoopTaskId) == TRUE)
				{
					result = popDataFromQueue(&(g_astChannels[nLoop]), pstChannelAPI, nNumOfDataToPop);
					ERRIFGOTO(result, _EXIT);
				}
			}
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


