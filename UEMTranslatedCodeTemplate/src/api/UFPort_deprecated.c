/*
 * UFPort_deprecated.c
 *
 *  Created on: 2017. 8. 24.
 *      Author: jej
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif


#include <uem_common.h>

#include <uem_data.h>

#include <UFPort.h>


int PORT_INITIALIZE(int nTaskId, char *pszPortName)
{
	int nChannelId = INVALID_CHANNEL_ID;
	uem_result result = ERR_UEM_UNKNOWN;

	result = UFPort_Initialize(nTaskId, pszPortName, &nChannelId);
	ERRIFGOTO(result, _EXIT);

_EXIT:
	return nChannelId;
}

int MQ_RECEIVE(int nChannelId, unsigned char *pBuffer, int nLen)
{
	int nDataRead = 0;
	uem_result result = ERR_UEM_UNKNOWN;

	result = UFPort_ReadFromQueue (nChannelId, pBuffer, nLen, 0, &nDataRead);
	ERRIFGOTO(result, _EXIT);

_EXIT:
	return nDataRead;
}


int MQ_SEND(int nChannelId, unsigned char *pBuffer, int nLen)
{
	int nDataWritten = 0;
	uem_result result = ERR_UEM_UNKNOWN;

	result = UFPort_WriteToQueue (nChannelId, pBuffer, nLen, 0, &nDataWritten);
	ERRIFGOTO(result, _EXIT);

_EXIT:
	return nDataWritten;
}


int MQ_AVAILABLE(int nChannelId)
{
	int nAvailableDataLen = 0;
	uem_result result = ERR_UEM_UNKNOWN;

	result = UFPort_GetNumOfAvailableData(nChannelId, 0, &nAvailableDataLen);
	ERRIFGOTO(result, _EXIT);

_EXIT:
	return nAvailableDataLen;
}


int AC_RECEIVE(int nChannelId, unsigned char *pBuffer, int nLen, int nIndex)
{
	int nDataRead = 0;
	uem_result result = ERR_UEM_UNKNOWN;

	result = UFPort_ReadFromQueue (nChannelId, pBuffer, nLen, nIndex, &nDataRead);
	ERRIFGOTO(result, _EXIT);

_EXIT:
	return nDataRead;
}


int AC_SEND(int nChannelId, unsigned char *pBuffer, int nLen, int nIndex)
{
	int nDataWritten = 0;
	uem_result result = ERR_UEM_UNKNOWN;

	result = UFPort_WriteToQueue (nChannelId, pBuffer, nLen, 0, &nDataWritten);
	ERRIFGOTO(result, _EXIT);

_EXIT:
	return nDataWritten;
}


int AC_CHECK(int nChannelId)
{
	int nAvailableChunkIndex = INVALID_CHUNK_INDEX;
	uem_result result = ERR_UEM_UNKNOWN;

	result = UFPort_GetAvailableIndex(nChannelId, &nAvailableChunkIndex);
	ERRIFGOTO(result, _EXIT);

_EXIT:
	return nAvailableChunkIndex;
}


int BUF_RECEIVE(int nChannelId, unsigned char *pBuffer, int nLen)
{
	int nDataRead = 0;
	uem_result result = ERR_UEM_UNKNOWN;

	result = UFPort_ReadFromBuffer (nChannelId, pBuffer, nLen, 0, OUT &nDataRead);
	ERRIFGOTO(result, _EXIT);
_EXIT:
	return nDataRead;
}


int BUF_SEND(int nChannelId, unsigned char *pBuffer, int nLen)
{
	int nDataWritten = 0;
	uem_result result = ERR_UEM_UNKNOWN;

	result = UFPort_WriteToBuffer (nChannelId, pBuffer, nLen, 0, &nDataWritten);
	ERRIFGOTO(result, _EXIT);

_EXIT:
	return nDataWritten;
}

