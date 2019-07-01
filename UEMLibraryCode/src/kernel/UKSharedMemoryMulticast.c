/*
 * UKSharedMemoryMulticast.c
 *
 *  Created on: 2019. 11. 9.
 *      Author: wecracy
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <uem_data.h>

#include <UKMulticastMemory.h>

uem_result UKSharedMemoryMulticast_Clear(SMulticastGroup *pstMulticastGroup)
{
	int index = 0;

	uem_result result = ERR_UEM_UNKNOWN;

	SMulticastPort *pstMulticastPort = NULL;
	SSharedMemoryMulticast *pstSharedMemoryMulticast = NULL;

	pstSharedMemoryMulticast = (SSharedMemoryMulticast *) pstMulticastGroup->pMulticastStruct;

	result = UKMulticastMemory_Clear(pstMulticastGroup, pstSharedMemoryMulticast);

	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKSharedMemoryMulticast_Initialize(SMulticastGroup *pstMulticastGroup)
{
	int index = 0;

	uem_result result = ERR_UEM_UNKNOWN;

	SSharedMemoryMulticast *pstSharedMemoryMulticast = NULL;

	pstSharedMemoryMulticast = (SSharedMemoryMulticast *) pstMulticastGroup->pMulticastStruct;

	result = UKMulticastMemory_Initialize(pstMulticastGroup, pstSharedMemoryMulticast);

	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKSharedMemoryMulticast_ReadFromBuffer(SMulticastPort *pstMulticastPort, IN OUT unsigned char *pBuffer, IN int nDataToRead, OUT int *pnDataRead)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSharedMemoryMulticast *pstSharedMemoryMulticast = NULL;

	pstSharedMemoryMulticast = (SSharedMemoryMulticast *) pstMulticastPort->pMulticastGroup->pMulticastStruct;

	result = UKMulticastMemory_ReadFromBuffer(pstMulticastPort, pstSharedMemoryMulticast, pBuffer, nDataToRead, pnDataRead);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKSharedMemoryChannel_WriteToBuffer (SMulticastPort *pstMulticastPort, IN unsigned char *pBuffer, IN int nDataToWrite, OUT int *pnDataWritten)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSharedMemoryMulticast *pstSharedMemoryMulticast = NULL;

	pstSharedMemoryMulticast = (SSharedMemoryMulticast *) pstMulticastPort->pMulticastGroup->pMulticastStruct;

	result = UKMulticastMemory_WriteToBuffer(pstMulticastPort, pstSharedMemoryMulticast, pBuffer, nDataToWrite, pnDataWritten);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKSharedMemoryMulticast_Finalize(SMulticastGroup *pstMulticastGroup)
{

	int index = 0;

	uem_result result = ERR_UEM_UNKNOWN;

	SMulticastPort *pstMulticastPort = NULL;
	SSharedMemoryMulticast *pstSharedMemoryMulticast = NULL;

	pstSharedMemoryMulticast = (SSharedMemoryMulticast *) pstMulticastGroup->pMulticastStruct;

	result = UKMulticastMemory_Finalize(pstMulticastGroup, pstSharedMemoryMulticast);

	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}
