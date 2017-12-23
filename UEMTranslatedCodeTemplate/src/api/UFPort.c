/*
 * UFPort.c
 *
 *  Created on: 2017. 8. 12.
 *      Author: jej
 */

#include <uem_common.h>

#include <UFPort.h>
#include <UKChannel.h>

uem_result UFPort_Initialize(IN int nTaskId, IN char *szPortName, OUT int *pnChannelId)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nChannelId;

#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(szPortName, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	IFVARERRASSIGNGOTO(pnChannelId, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif

	nChannelId = UKChannel_GetChannelIdByTaskAndPortName(nTaskId, szPortName);

	*pnChannelId = nChannelId;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UFPort_ReadFromQueue (IN int nChannelId, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead)
{
	uem_result result = ERR_UEM_UNKNOWN;

#ifdef ARGUMENT_CHECK
	if(nDataToRead <= 0)
	{
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}

	IFVARERRASSIGNGOTO(pBuffer, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	IFVARERRASSIGNGOTO(pnDataRead, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	result = UKChannel_ReadFromQueue(nChannelId, pBuffer, nDataToRead, nChunkIndex, pnDataRead);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UFPort_ReadFromBuffer (IN int nChannelId, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead)
{
	uem_result result = ERR_UEM_UNKNOWN;
#ifdef ARGUMENT_CHECK
	if(nDataToRead <= 0)
	{
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}

	IFVARERRASSIGNGOTO(pBuffer, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	IFVARERRASSIGNGOTO(pnDataRead, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	result =  UKChannel_ReadFromBuffer(nChannelId, pBuffer, nDataToRead, nChunkIndex, pnDataRead);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UFPort_WriteToQueue (IN int nChannelId, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten)
{
	uem_result result = ERR_UEM_UNKNOWN;
#ifdef ARGUMENT_CHECK
	if(nDataToWrite <= 0)
	{
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}

	IFVARERRASSIGNGOTO(pnDataWritten, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	IFVARERRASSIGNGOTO(pBuffer, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	result = UKChannel_WriteToQueue(nChannelId, pBuffer, nDataToWrite, nChunkIndex, pnDataWritten);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UFPort_WriteToBuffer (IN int nChannelId, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten)
{
	uem_result result = ERR_UEM_UNKNOWN;
#ifdef ARGUMENT_CHECK
	if(nDataToWrite <= 0)
	{
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}

	IFVARERRASSIGNGOTO(pnDataWritten, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	IFVARERRASSIGNGOTO(pBuffer, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	result = UKChannel_WriteToBuffer(nChannelId, pBuffer, nDataToWrite, nChunkIndex, pnDataWritten);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UFPort_GetNumOfAvailableData (IN int nChannelId, IN int nChunkIndex, OUT int *pnDataNum)
{
	uem_result result = ERR_UEM_UNKNOWN;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(pnDataNum, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	result = UKChannel_GetNumOfAvailableData (nChannelId, nChunkIndex, pnDataNum);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UFPort_GetAvailableIndex (IN int nChannelId, OUT int *pnChunkIndex)
{
	uem_result result = ERR_UEM_UNKNOWN;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(pnChunkIndex, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	result = UKChannel_GetAvailableIndex (nChannelId, pnChunkIndex);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


