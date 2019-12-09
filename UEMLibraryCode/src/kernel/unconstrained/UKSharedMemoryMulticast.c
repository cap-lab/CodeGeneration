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
	SMulticastCommunication *pstCommunication = NULL;

	result = UKMulticast_GetCommunication(pstMulticastGroup->astCommunicationList, pstMulticastGroup->nCommunicationTypeNum, SHARED_MEMORY, &pstCommunication);
	ERRIFGOTO(result, _EXIT);

	result = UKMulticastMemory_Initialize(pstMulticastGroup, (SSharedMemoryMulticast *) pstCommunication->pstSocket);

	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKSharedMemoryMulticast_ReadFromBuffer(SMulticastPort *pstMulticastPort, IN OUT unsigned char *pBuffer, IN int nDataToRead, OUT int *pnDataRead)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SMulticastCommunication *pstCommunication = NULL;

	result = UKMulticast_GetCommunication(pstMulticastPort->astCommunicationList, pstMulticastPort->nCommunicationTypeNum, SHARED_MEMORY, &pstCommunication);
	ERRIFGOTO(result, _EXIT);

	result = UKMulticastMemory_ReadFromBuffer(pstMulticastPort, (SSharedMemoryMulticast *) pstCommunication->pstSocket, pBuffer, nDataToRead, pnDataRead);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKSharedMemoryMulticast_WriteToBuffer(SMulticastPort *pstMulticastPort, IN unsigned char *pBuffer, IN int nDataToWrite, OUT int *pnDataWritten)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SMulticastCommunication *pstCommunication = NULL;

	result = UKMulticast_GetCommunication(pstMulticastPort->astCommunicationList, pstMulticastPort->nCommunicationTypeNum, SHARED_MEMORY, &pstCommunication);
	ERRIFGOTO(result, _EXIT);

	result = UKMulticastMemory_WriteToBuffer(pstMulticastPort, (SSharedMemoryMulticast *) pstCommunication->pstSocket, pBuffer, nDataToWrite, pnDataWritten);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKSharedMemoryMulticastGroup_Finalize(SMulticastGroup *pstMulticastGroup)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SMulticastCommunication *pstCommunication = NULL;

	result = UKMulticast_GetCommunication(pstMulticastGroup->astCommunicationList, pstMulticastGroup->nCommunicationTypeNum, SHARED_MEMORY, &pstCommunication);
	ERRIFGOTO(result, _EXIT);

	result = UKMulticastMemory_Finalize(pstMulticastGroup, (SSharedMemoryMulticast *) pstCommunication->pstSocket);

	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}
