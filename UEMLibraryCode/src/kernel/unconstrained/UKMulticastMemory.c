/*
 * UKMulticastMemory.c
 *
 *  Created on: 2019. 6. 20.
 *      Author: wecracy
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <UCBasic.h>
#include <UCAlloc.h>
#include <UCThreadMutex.h>

#include <UKHostSystem.h>

#include <uem_data.h>

uem_result UKMulticastMemory_Clear(SMulticastGroup *pstMulticastGroup, SSharedMemoryMulticast *pstSharedMemoryMulticast)
{
	uem_result result = ERR_UEM_UNKNOWN;

	IFVARERRASSIGNGOTO(pstSharedMemoryMulticast, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	// information clear
	// pDataStart => clear (pBuffer)
	// nDataLen => clear (0)
	pstSharedMemoryMulticast->pDataStart = pstSharedMemoryMulticast->pBuffer;
	pstSharedMemoryMulticast->pDataEnd = pstSharedMemoryMulticast->pBuffer;
	pstSharedMemoryMulticast->nDataLen = 0;
	pstSharedMemoryMulticast->nReadReferenceCount = 0;
	pstSharedMemoryMulticast->nWriteReferenceCount = 0;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKMulticastMemory_Initialize(SMulticastGroup *pstMulticastGroup, SSharedMemoryMulticast *pstSharedMemoryMulticast)
{
	uem_result result = ERR_UEM_UNKNOWN;

	IFVARERRASSIGNGOTO(pstSharedMemoryMulticast, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	// initialize buffer
	// If not set, initialize those things, // pBuffer => is NULL => alloc
	if(pstSharedMemoryMulticast->pBuffer == NULL)
	{
		result = UKHostSystem_CreateMemory(pstMulticastGroup->nBufSize, 0 /* processor ID don't need for creating memory */, &(pstSharedMemoryMulticast->pBuffer));
		ERRIFGOTO(result, _EXIT);
	}

	// hMutex => initialize/create
	if(pstSharedMemoryMulticast->hMutex == NULL)
	{
		result = UCThreadMutex_Create(&(pstSharedMemoryMulticast->hMutex));
		ERRIFGOTO(result, _EXIT);
	}

	result = UKMulticastMemory_Clear(pstMulticastGroup, pstSharedMemoryMulticast);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKMulticastMemory_ReadFromBuffer(SMulticastPort *pstMulticastPort, SSharedMemoryMulticast *pstSharedMemoryMulticast, IN OUT unsigned char *pBuffer, IN int nDataToRead, OUT int *pnDataRead)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SGenericMemoryAccess *pstMemoryAPI = NULL;

	IFVARERRASSIGNGOTO(pstSharedMemoryMulticast, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	pstMemoryAPI = pstMulticastPort->pstMemoryAccessAPI;

	result = UCThreadMutex_Lock(pstSharedMemoryMulticast->hMutex);
	ERRIFGOTO(result, _EXIT);

	if(pstSharedMemoryMulticast->nDataLen < nDataToRead)
	{
		if(pstSharedMemoryMulticast->nDataLen > 0)
		{
			result = pstMemoryAPI->fnCopyFromMemory(pBuffer, pstSharedMemoryMulticast->pDataStart, pstSharedMemoryMulticast->nDataLen);
			ERRIFGOTO(result, _EXIT_LOCK);
		}

		*pnDataRead = pstSharedMemoryMulticast->nDataLen;
	}
	else // pstChannel->nDataLen >= nDataToRead
	{
		result = pstMemoryAPI->fnCopyFromMemory(pBuffer, pstSharedMemoryMulticast->pDataStart, nDataToRead);
		ERRIFGOTO(result, _EXIT_LOCK);

		*pnDataRead = nDataToRead;
	}

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
	UCThreadMutex_Unlock(pstSharedMemoryMulticast->hMutex);
_EXIT:
	return result;
}

uem_result UKMulticastMemory_WriteToBuffer (SMulticastPort *pstMulticastPort, SSharedMemoryMulticast *pstSharedMemoryMulticast, IN unsigned char *pBuffer, IN int nDataToWrite, OUT int *pnDataWritten)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SGenericMemoryAccess *pstMemoryAPI = NULL;

	IFVARERRASSIGNGOTO(pstSharedMemoryMulticast, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	pstMemoryAPI = pstMulticastPort->pstMemoryAccessAPI;

	result = UCThreadMutex_Lock(pstSharedMemoryMulticast->hMutex);
	ERRIFGOTO(result, _EXIT);

	if(pstMulticastPort->pMulticastGroup->nBufSize >= nDataToWrite)
	{
		if (nDataToWrite > 0)
		{
			result = pstMemoryAPI->fnCopyToMemory(pstSharedMemoryMulticast->pDataStart, pBuffer, nDataToWrite);
			ERRIFGOTO(result, _EXIT_LOCK);
		}

		*pnDataWritten = nDataToWrite;
	}
	else // pstChannel->nBufSize < nDataToWrite
	{
		result = pstMemoryAPI->fnCopyToMemory(pstSharedMemoryMulticast->pDataStart, pBuffer, pstMulticastPort->pMulticastGroup->nBufSize);
		ERRIFGOTO(result, _EXIT_LOCK);

		*pnDataWritten = pstMulticastPort->pMulticastGroup->nBufSize;
	}

	pstSharedMemoryMulticast->nDataLen = *pnDataWritten;

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
	UCThreadMutex_Unlock(pstSharedMemoryMulticast->hMutex);
_EXIT:
	return result;
}

uem_result UKMulticastMemory_Finalize(SMulticastGroup *pstMulticastGroup, SSharedMemoryMulticast *pstSharedMemoryMulticast)
{
	uem_result result = ERR_UEM_UNKNOWN;

	IFVARERRASSIGNGOTO(pstSharedMemoryMulticast, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	// ignore error
	UCThreadMutex_Destroy(&(pstSharedMemoryMulticast->hMutex));

	if(pstSharedMemoryMulticast->pBuffer != NULL)
	{
		UKHostSystem_DestroyMemory(&(pstSharedMemoryMulticast->pBuffer));
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}
