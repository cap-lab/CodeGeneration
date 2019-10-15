/*
 * UKSharedMemoryMulticast.c
 *
 *  Created on: 2019. 6. 20.
 *      Author: wecracy
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <uem_data.h>

#include <UKMulticastMemory.h>

uem_result UKSharedMemoryMulticastGroup_Initialize(SMulticastGroup *pstMulticastGroup)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SMulticastCommunicationGate *pstCommunicationGate = NULL;

	result = UKMulticast_GetCommunicationGate(pstMulticastGroup->astMulticastGateList, pstMulticastGroup->nCommunicationTypeNum, SHARED_MEMORY, &pstCommunicationGate);
	ERRIFGOTO(result, _EXIT);

	result = UKMulticastMemory_Initialize(pstMulticastGroup, (SSharedMemoryMulticast *) pstCommunicationGate->pstSocket);

	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKSharedMemoryMulticast_ReadFromBuffer(SMulticastPort *pstMulticastPort, IN OUT unsigned char *pBuffer, IN int nDataToRead, OUT int *pnDataRead)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SMulticastCommunicationGate *pstCommunicationGate = NULL;

	result = UKMulticast_GetCommunicationGate(pstMulticastPort->astMulticastGateList, pstMulticastPort->nCommunicationTypeNum, SHARED_MEMORY, &pstCommunicationGate);
	ERRIFGOTO(result, _EXIT);

	result = UKMulticastMemory_ReadFromBuffer(pstMulticastPort, (SSharedMemoryMulticast *) pstCommunicationGate->pstSocket, pBuffer, nDataToRead, pnDataRead);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKSharedMemoryMulticast_WriteToBuffer(SMulticastPort *pstMulticastPort, IN unsigned char *pBuffer, IN int nDataToWrite, OUT int *pnDataWritten)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SMulticastCommunicationGate *pstCommunicationGate = NULL;

	result = UKMulticast_GetCommunicationGate(pstMulticastPort->astMulticastGateList, pstMulticastPort->nCommunicationTypeNum, SHARED_MEMORY, &pstCommunicationGate);
	ERRIFGOTO(result, _EXIT);

	result = UKMulticastMemory_WriteToBuffer(pstMulticastPort, (SSharedMemoryMulticast *) pstCommunicationGate->pstSocket, pBuffer, nDataToWrite, pnDataWritten);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKSharedMemoryMulticastGroup_Finalize(SMulticastGroup *pstMulticastGroup)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SMulticastCommunicationGate *pstCommunicationGate = NULL;

	result = UKMulticast_GetCommunicationGate(pstMulticastGroup->astMulticastGateList, pstMulticastGroup->nCommunicationTypeNum, SHARED_MEMORY, &pstCommunicationGate);
	ERRIFGOTO(result, _EXIT);

	result = UKMulticastMemory_Finalize(pstMulticastGroup, (SSharedMemoryMulticast *) pstCommunicationGate->pstSocket);

	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}
