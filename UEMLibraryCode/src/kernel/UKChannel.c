/*
 * UKChannel.c
 *
 *  Created on: 2018. 8. 28.
 *      Author: chjej202
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>
#include <UCString.h>

#include <uem_channel_data.h>

#include <UKSharedMemoryChannel.h>

uem_result ChannelAPI_GetAPIStructureFromCommunicationType(IN ECommunicationType enType, OUT SChannelAPI **ppstChannelAPI);

uem_result UKChannel_Initialize()
{
	uem_result result = ERR_UEM_UNKNOWN;
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
	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
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

uem_result UKChannel_GetChannelSize(IN int nChannelId, OUT int *pnChannelSize)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nIndex = 0;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(pnChannelSize, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	nIndex = getChannelIndexById(nChannelId);
	IFVARERRASSIGNGOTO(nIndex, INVALID_CHANNEL_ID, result, ERR_UEM_INVALID_PARAM, _EXIT);

	*pnChannelSize = g_astChannels[nIndex].nBufSize;

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

