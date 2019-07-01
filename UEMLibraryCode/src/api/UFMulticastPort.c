/*
 * UFMulticastPort.c
 *
 *  Created on: 2019. 8. 12.
 *      Author: wecracy
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif


#include <uem_common.h>

#include <UFMulticastPort.h>
#include <UKMulticast.h>

uem_result UFMulticastPort_Initialize(IN int nTaskId, IN char *szPortName, OUT int *pnMulticastGroupId, OUT int *pnMulticastPortId)
{
	uem_result result = ERR_UEM_UNKNOWN;

#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(szPortName, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	IFVARERRASSIGNGOTO(pnMulticastGroupId, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	IFVARERRASSIGNGOTO(pnMulticastPortId, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	result = UKMulticast_GetMulticastGroupIdByTaskAndPortName(nTaskId, szPortName, pnMulticastGroupId);
	ERRIFGOTO(result, _EXIT);
	result = UKMulticast_GetMulticastPortIdByTaskAndPortName(nTaskId, szPortName, pnMulticastPortId);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UFMulticastPort_ReadFromBuffer (IN int nMulticastGroupId, IN int nMulticastPortId, IN OUT unsigned char *pBuffer, IN int nDataToRead, OUT int *pnDataRead)
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
	result =  UKMulticast_ReadFromBuffer(nMulticastGroupId, nMulticastPortId, pBuffer, nDataToRead, pnDataRead);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UFMulticastPort_WriteToBuffer (IN int nMulticastGroupId, IN int nMulticastPortId, IN unsigned char *pBuffer, IN int nDataToWrite, OUT int *pnDataWritten)
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
	result = UKMulticast_WriteToBuffer(nMulticastGroupId, nMulticastPortId, pBuffer, nDataToWrite, pnDataWritten);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UFMulticastPort_GetMulticastSize(IN int nMulticastGroupId, OUT int *pnMulticastGroupSize)
{
	uem_result result = ERR_UEM_UNKNOWN;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(pnMulticastGroupSize, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	result = UKMulticast_GetChannelSize(nMulticastGroupId, pnMulticastGroupSize);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}
