/*
 * UKUDPSocketMulticast.c
 *
 *  Created on: 2019. 6. 20.
 *      Author: wecracy
 */


#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#ifndef WIN32
#include <sys/socket.h>
#else
#include <winsock.h>
#endif


#include <uem_common.h>

#include <UCTime.h>
#include <UCBasic.h>
#include <UCAlloc.h>
#include <UCUDPSocket.h>
#include <UCThreadMutex.h>
#include <UCEndian.h>

#include <uem_data.h>

#include <UKUDPSocketMulticast.h>
#include <UKSharedMemoryMulticast.h>
#include <UKHostSystem.h>
#include <UKTask.h>

#include <uem_udp_data.h>
#include <uem_multicast_data.h>

#define MULTICAST_UDP_HEADER_DEVICE_ID_SIZE (4)
#define MULTICAST_UDP_HEADER_GROUP_ID_SIZE (4)
#define MULTICAST_UDP_HEADER_CHUNK_ID_SIZE (4)
#define MULTICAST_UDP_HEADER_SIZE (MULTICAST_UDP_HEADER_DEVICE_ID_SIZE + \
								   MULTICAST_UDP_HEADER_GROUP_ID_SIZE + \
								   MULTICAST_UDP_HEADER_CHUNK_ID_SIZE)

#define MULTICAST_UDP_HEADER_START (0)
#define MULTICAST_UDP_HEADER_DEVICE_ID_START (MULTICAST_UDP_HEADER_START)
#define MULTICAST_UDP_HEADER_GROUP_ID_START (MULTICAST_UDP_HEADER_DEVICE_ID_START + \
		                                     MULTICAST_UDP_HEADER_DEVICE_ID_SIZE)
#define MULTICAST_UDP_HEADER_CHUNK_ID_START (MULTICAST_UDP_HEADER_GROUP_ID_START + \
		                                     MULTICAST_UDP_HEADER_GROUP_ID_SIZE)
#define MULTICAST_UDP_BODY_START (MULTICAST_UDP_HEADER_START + \
		                          MULTICAST_UDP_HEADER_SIZE)

static uem_result UKUDPSocketMulticast_AllocBuffer(SUDPSocket *pstUDPSocket, int nBufSize)
{
	uem_result result = ERR_UEM_UNKNOWN;

	if(pstUDPSocket->pHeader != NULL)
	{
		ERRASSIGNGOTO(result, ERR_UEM_REALLOCATE_BUFFER, _EXIT);
	}

	pstUDPSocket->pHeader = UCAlloc_calloc(nBufSize + MULTICAST_UDP_HEADER_SIZE, sizeof(unsigned char));
	ERRMEMGOTO(pstUDPSocket->pHeader, result, _EXIT);
	pstUDPSocket->pData = pstUDPSocket->pHeader + MULTICAST_UDP_BODY_START;
	pstUDPSocket->nDataLen = 0;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static uem_result checkDeviceId(SUDPMulticastReceiver *pstUDPMulticastReceiver)
{
	uem_result result = ERR_UEM_UNKNOWN;
	uem_bool bRet = FALSE;
	int nDeviceId;

	bRet = UCEndian_LittleEndianCharToSystemInt(
			(char *)&pstUDPMulticastReceiver->stReceiverSocket.pHeader[MULTICAST_UDP_HEADER_DEVICE_ID_START],
			MULTICAST_UDP_HEADER_DEVICE_ID_SIZE, &nDeviceId);
	IFVARERRASSIGNGOTO(bRet, FALSE, result, ERR_UEM_SKIP_THIS, _EXIT);

	if (nDeviceId == g_nDeviceId)
	{
		result = ERR_UEM_SKIP_THIS;
	}
	else
	{
		result = ERR_UEM_NOERROR;
	}
_EXIT:
	return result;
}


static uem_result checkMessageOwner(SUDPMulticastReceiver *pstUDPMulticastReceiver, int *pnGroupID)
{
	uem_result result = ERR_UEM_SKIP_THIS;
	uem_bool bRet = FALSE;
    int nLoop = 0;

	bRet = UCEndian_LittleEndianCharToSystemInt(
			(char *)&pstUDPMulticastReceiver->stReceiverSocket.pHeader[MULTICAST_UDP_HEADER_GROUP_ID_START],
			MULTICAST_UDP_HEADER_GROUP_ID_SIZE, pnGroupID);
	IFVARERRASSIGNGOTO(bRet, FALSE, result, ERR_UEM_SKIP_THIS, _EXIT);

	for(nLoop = 0 ; nLoop < pstUDPMulticastReceiver->pstUDPMulticast->nReceiverNum ; nLoop++)
	{
		if(pstUDPMulticastReceiver->pstUDPMulticast->anReceivers[nLoop] == *pnGroupID)
		{
			result = ERR_UEM_NOERROR;
			break;
		}
	}

_EXIT:
	return result;
}

static uem_result getMessageOwnerIndex(SUDPMulticastReceiver *pstUDPMulticastReceiver, int nGroupId, int *pnGroupIndex)
{
	uem_result result = ERR_UEM_SKIP_THIS;
	int nLoop = 0;

	for(nLoop = 0 ; nLoop < g_nMulticastGroupNum ; nLoop++)
	{
		if(g_astMulticastGroups[nLoop].nMulticastGroupId == nGroupId)
		{
			*pnGroupIndex = nLoop;
			result = ERR_UEM_NOERROR;
			break;
		}
	}

	return result;
}

static uem_result getMessageChunkIndex(SUDPMulticastReceiver *pstUDPMulticastReceiver, int *pnChunkIndex)
{
	uem_result result = ERR_UEM_SKIP_THIS;
	uem_bool bRet = FALSE;

	bRet = UCEndian_LittleEndianCharToSystemInt(
			(char *)&pstUDPMulticastReceiver->stReceiverSocket.pHeader[MULTICAST_UDP_HEADER_CHUNK_ID_START],
			MULTICAST_UDP_HEADER_CHUNK_ID_SIZE, pnChunkIndex);
	IFVARERRASSIGNGOTO(bRet, FALSE, result, ERR_UEM_SKIP_THIS, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static void *multicastHandlingThread(void *pData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SUDPMulticastReceiver *pstUDPMulticastReceiver = NULL;
	SMulticastCommunicationGate *pstMulticastCommunicationGate = NULL;
	int nGroupId = 0;
    int nGroupIndex = 0;
    int nChunkIndex = 0;

    pstUDPMulticastReceiver = pData;

	while(pstUDPMulticastReceiver->bExit == FALSE)
	{
		result = UCDynamicSocket_RecvFrom(pstUDPMulticastReceiver->stReceiverSocket.hSocket,
				pstUDPMulticastReceiver->pstUDPMulticast->stUDPInfo.pszIP, 3, UDP_MAX, pstUDPMulticastReceiver->stReceiverSocket.pHeader,
				&pstUDPMulticastReceiver->stReceiverSocket.nDataLen);
		IFVARERRASSIGNGOTO(result, ERR_UEM_NET_TIMEOUT, result, ERR_UEM_NOERROR, _EXIT_CONTINUE);
		ERRIFGOTO(result, _EXIT);
		if(pstUDPMulticastReceiver->stReceiverSocket.nDataLen < MULTICAST_UDP_HEADER_SIZE)
		{
			continue;
		}

		result = checkDeviceId(pstUDPMulticastReceiver);
		IFVARERRASSIGNGOTO(result, ERR_UEM_SKIP_THIS, result, ERR_UEM_NOERROR, _EXIT_CONTINUE);
		ERRIFGOTO(result, _EXIT);

		result = checkMessageOwner(pstUDPMulticastReceiver, &nGroupId);
		IFVARERRASSIGNGOTO(result, ERR_UEM_SKIP_THIS, result, ERR_UEM_NOERROR, _EXIT_CONTINUE);
		ERRIFGOTO(result, _EXIT);

		result = getMessageOwnerIndex(pstUDPMulticastReceiver, nGroupId, &nGroupIndex);
		IFVARERRASSIGNGOTO(result, ERR_UEM_SKIP_THIS, result, ERR_UEM_NOERROR, _EXIT_CONTINUE);
		ERRIFGOTO(result, _EXIT);

		result = getMessageChunkIndex(pstUDPMulticastReceiver, &nChunkIndex);
		IFVARERRASSIGNGOTO(result, ERR_UEM_SKIP_THIS, result, ERR_UEM_NOERROR, _EXIT_CONTINUE);
		ERRIFGOTO(result, _EXIT);

		if(nChunkIndex*(UDP_MAX-MULTICAST_UDP_HEADER_SIZE) + (pstUDPMulticastReceiver->stReceiverSocket.nDataLen-MULTICAST_UDP_HEADER_SIZE) > g_astMulticastGroups[nGroupIndex].nBufSize)
		{
			continue;
		}

		UKMulticast_GetCommunicationGate(g_astMulticastGroups[nGroupIndex].astMulticastGateList, g_astMulticastGroups[nGroupIndex].nCommunicationTypeNum, SHARED_MEMORY, &pstMulticastCommunicationGate);
		ERRIFGOTO(result, _EXIT);

		result = UCThreadMutex_Lock(((SSharedMemoryMulticast *)pstMulticastCommunicationGate->pstSocket)->hMutex);
		ERRIFGOTO(result, _EXIT);

		if(nChunkIndex == 0)
		{
			((SSharedMemoryMulticast *)pstMulticastCommunicationGate->pstSocket)->nDataLen = pstUDPMulticastReceiver->stReceiverSocket.nDataLen - MULTICAST_UDP_HEADER_SIZE;
		}
		else
		{
			((SSharedMemoryMulticast *)pstMulticastCommunicationGate->pstSocket)->nDataLen += pstUDPMulticastReceiver->stReceiverSocket.nDataLen - MULTICAST_UDP_HEADER_SIZE;
		}

		result = UKHostSystem_CopyToMemory(((SSharedMemoryMulticast *)pstMulticastCommunicationGate->pstSocket)->pData + nChunkIndex*(UDP_MAX - MULTICAST_UDP_HEADER_SIZE),
				pstUDPMulticastReceiver->stReceiverSocket.pData, pstUDPMulticastReceiver->stReceiverSocket.nDataLen - MULTICAST_UDP_HEADER_SIZE);
		ERRIFGOTO(result, _EXIT_LOCK);

_EXIT_LOCK:
		UCThreadMutex_Unlock(((SSharedMemoryMulticast *)pstMulticastCommunicationGate->pstSocket)->hMutex);
_EXIT_CONTINUE:
		ERRIFGOTO(result, _EXIT);
	}

_EXIT:
	if(result != ERR_UEM_NOERROR && pstUDPMulticastReceiver->bExit == FALSE)
	{
		UEM_DEBUG_PRINT("channelReceiverHandlingThread is exited with error: %08x\n", result);
	}
	return NULL;
}


static uem_result createMulticastReceiverThread(SUDPMulticastReceiver *pstUDPMulticastReceiver)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = UCThread_Create(multicastHandlingThread, (void *) pstUDPMulticastReceiver, &(pstUDPMulticastReceiver->hManagementThread));
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static uem_result destroyMulticastReceiverThread(SUDPMulticastReceiver *pstUDPMulticastReceiver)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = UCThread_Destroy(&(pstUDPMulticastReceiver->hManagementThread), FALSE, 0);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKUDPSocketMulticast_SocketInitialize(SUDPSocket *pstUDPSocket, SUDPInfo *pstUDPInfo, IN uem_bool bIsServer)
{
	SSocketInfo stSocketInfo;
	uem_result result = ERR_UEM_UNKNOWN;

	if (pstUDPSocket->hSocket == NULL)
	{
		stSocketInfo.enSocketType = SOCKET_TYPE_UDP;
		stSocketInfo.nPort = pstUDPInfo->nPort;
		stSocketInfo.pszSocketPath = pstUDPInfo->pszIP;
		result = UCDynamicSocket_Create(&stSocketInfo, bIsServer, &pstUDPSocket->hSocket);
		ERRIFGOTO(result, _EXIT);
	}

	result = UCThreadMutex_Create(&(pstUDPSocket->hMutex));
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static uem_result UKUDPSocketMulticast_MakeHeader(IN SMulticastPort *pstMulticastPort, IN int nChunk, IN OUT unsigned char *pHeader)
{
	uem_result result = ERR_UEM_UNKNOWN;
	uem_bool bRet = FALSE;

	bRet = UCEndian_SystemIntToLittleEndianChar(g_nDeviceId,
			(char *)&pHeader[MULTICAST_UDP_HEADER_DEVICE_ID_START], MULTICAST_UDP_HEADER_DEVICE_ID_SIZE);
	IFVARERRASSIGNGOTO(bRet, FALSE, result, ERR_UEM_CONVERSION_ERROR, _EXIT);

	bRet = UCEndian_SystemIntToLittleEndianChar(pstMulticastPort->pstMulticastGroup->nMulticastGroupId,
			(char *)&pHeader[MULTICAST_UDP_HEADER_GROUP_ID_START], MULTICAST_UDP_HEADER_GROUP_ID_SIZE);
	IFVARERRASSIGNGOTO(bRet, FALSE, result, ERR_UEM_CONVERSION_ERROR, _EXIT);

	bRet = UCEndian_SystemIntToLittleEndianChar(nChunk,
			(char *)&pHeader[MULTICAST_UDP_HEADER_CHUNK_ID_START], MULTICAST_UDP_HEADER_CHUNK_ID_SIZE);
	IFVARERRASSIGNGOTO(bRet, FALSE, result, ERR_UEM_CONVERSION_ERROR, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static uem_result UKUDPSocketMulticast_MakeData(IN SMulticastPort *pstMulticastPort, IN unsigned char *pData, IN int nDataToWrite, IN OUT unsigned char *pBuffer)
{
	uem_result result = ERR_UEM_UNKNOWN;

	if(nDataToWrite > UDP_MAX-MULTICAST_UDP_HEADER_SIZE)
	{
		nDataToWrite = UDP_MAX-MULTICAST_UDP_HEADER_SIZE;
	}

	result = pstMulticastPort->pstMemoryAccessAPI->fnCopyToMemory(pBuffer, pData, nDataToWrite);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;

_EXIT:
	return result;
}

uem_result UKUDPSocketMulticast_WriteToBuffer(IN SMulticastPort *pstMulticastPort, IN unsigned char *pData, IN int nDataToWrite, OUT int *pnDataWritten)
{
	uem_result result = ERR_UEM_UNKNOWN;

	SUDPMulticastSender *pstUDPMulticastSender = NULL;
	SMulticastCommunicationGate *pstCommunicationGate = NULL;
	int nChunk = 0;

	result = UKMulticast_GetCommunicationGate(pstMulticastPort->astMulticastGateList, pstMulticastPort->nCommunicationTypeNum, UDP, &pstCommunicationGate);
	ERRIFGOTO(result, _EXIT);

	pstUDPMulticastSender = (SUDPMulticastSender *)pstCommunicationGate->pstSocket;

	*pnDataWritten = 0;

	for(nChunk = 0 ; nDataToWrite > 0 ; nDataToWrite-=(UDP_MAX - MULTICAST_UDP_HEADER_SIZE), nChunk++)
	{
		result = UKUDPSocketMulticast_MakeHeader(pstMulticastPort, nChunk, (unsigned char *)pstUDPMulticastSender->stSenderSocket.pHeader);
		ERRIFGOTO(result, _EXIT);

		result = UKUDPSocketMulticast_MakeData(pstMulticastPort, pData, nDataToWrite, (unsigned char *)pstUDPMulticastSender->stSenderSocket.pData);
		ERRIFGOTO(result, _EXIT);

		result = UCDynamicSocket_Sendto(pstUDPMulticastSender->stSenderSocket.hSocket, pstUDPMulticastSender->pstUDPMulticast->stUDPInfo.pszIP, 1,
				(unsigned char *)pstUDPMulticastSender->stSenderSocket.pHeader, MULTICAST_UDP_HEADER_SIZE + (nDataToWrite > (UDP_MAX - MULTICAST_UDP_HEADER_SIZE)? UDP_MAX - MULTICAST_UDP_HEADER_SIZE: nDataToWrite),
				&pstUDPMulticastSender->stSenderSocket.nDataLen);
		ERRIFGOTO(result, _EXIT);

		*pnDataWritten+=(pstUDPMulticastSender->stSenderSocket.nDataLen -MULTICAST_UDP_HEADER_SIZE);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKUDPSocketMulticastPort_Initialize(IN SMulticastPort *pstMulticastPort)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SMulticastCommunicationGate *pstCommunicationGate = NULL;
	int nUDPLoop = 0;
	int nSenderLoop = 0;

	if(pstMulticastPort->enDirection == PORT_DIRECTION_OUTPUT)
	{
		result = UKMulticast_GetCommunicationGate(pstMulticastPort->astMulticastGateList, pstMulticastPort->nCommunicationTypeNum, UDP, &pstCommunicationGate);
		ERRIFGOTO(result, _EXIT);

		pstCommunicationGate->pstSocket = UCAlloc_calloc(1, sizeof(SUDPMulticastSender));

		for(nUDPLoop = 0 ; nUDPLoop < g_nMulticastUDPNum ; nUDPLoop++)
		{
			for(nSenderLoop = 0 ; nSenderLoop < g_astMulticastUDPList[nUDPLoop].nSenderNum ; nSenderLoop++)
			{
				if(pstMulticastPort->pstMulticastGroup->nMulticastGroupId == g_astMulticastUDPList[nUDPLoop].anSenders[nSenderLoop])
				{
					((SUDPMulticastSender *)pstCommunicationGate->pstSocket)->pstUDPMulticast = &g_astMulticastUDPList[nUDPLoop];
					break;
				}
			}
			if(((SUDPMulticastSender *)pstCommunicationGate->pstSocket)->pstUDPMulticast != NULL)
			{
				break;
			}
		}

		result = UKUDPSocketMulticast_AllocBuffer(&((SUDPMulticastSender *)pstCommunicationGate->pstSocket)->stSenderSocket, pstMulticastPort->pstMulticastGroup->nBufSize);
		ERRIFGOTO(result, _EXIT);

		result = UKUDPSocketMulticast_SocketInitialize(&((SUDPMulticastSender *)pstCommunicationGate->pstSocket)->stSenderSocket, &((SUDPMulticastSender *)pstCommunicationGate->pstSocket)->pstUDPMulticast->stUDPInfo, FALSE);
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKUDPSocketMulticastPort_Finalize(IN SMulticastPort *pstMulticastPort)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SMulticastCommunicationGate *pstCommunicationGate = NULL;

	if(pstMulticastPort->enDirection == PORT_DIRECTION_OUTPUT)
	{
		result = UKMulticast_GetCommunicationGate(pstMulticastPort->astMulticastGateList, pstMulticastPort->nCommunicationTypeNum, UDP, &pstCommunicationGate);
		ERRIFGOTO(result, _EXIT);

		SAFEMEMFREE(((SUDPMulticastSender *)pstCommunicationGate->pstSocket)->stSenderSocket.pHeader);

		SAFEMEMFREE(pstCommunicationGate->pstSocket);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKUDPSocketMulticastAPI_Initialize()
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;

	for(nLoop = 0 ; nLoop < g_nMulticastUDPNum ; nLoop++)
	{
		if(g_astMulticastUDPList[nLoop].nReceiverNum == 0)
		{
			continue;
		}

		g_astMulticastUDPList[nLoop].pstUDPMulticastReceiver = (SUDPMulticastReceiver *) UCAlloc_calloc(1, sizeof(SUDPMulticastReceiver));
		g_astMulticastUDPList[nLoop].pstUDPMulticastReceiver->pstUDPMulticast = &g_astMulticastUDPList[nLoop];

		result = UKUDPSocketMulticast_AllocBuffer(&g_astMulticastUDPList[nLoop].pstUDPMulticastReceiver->stReceiverSocket, UDP_MAX);
		ERRIFGOTO(result, _EXIT);

		result = UKUDPSocketMulticast_SocketInitialize(&g_astMulticastUDPList[nLoop].pstUDPMulticastReceiver->stReceiverSocket, &g_astMulticastUDPList[nLoop].pstUDPMulticastReceiver->pstUDPMulticast->stUDPInfo, TRUE);
		ERRIFGOTO(result, _EXIT);

		result = UCDynamicSocket_Bind(g_astMulticastUDPList[nLoop].pstUDPMulticastReceiver->stReceiverSocket.hSocket);
		ERRIFGOTO(result, _EXIT);

		// create receive thread
		result = createMulticastReceiverThread(g_astMulticastUDPList[nLoop].pstUDPMulticastReceiver);
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKUDPSocketMulticastAPI_Finalize()
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;

	for(nLoop = 0 ; nLoop < g_nMulticastUDPNum ; nLoop++)
	{
		if(g_astMulticastUDPList[nLoop].pstUDPMulticastReceiver == NULL)
		{
			continue;
		}
		g_astMulticastUDPList[nLoop].pstUDPMulticastReceiver->bExit = TRUE;
		if(g_astMulticastUDPList[nLoop].pstUDPMulticastReceiver->hManagementThread != NULL)
		{
			result = destroyMulticastReceiverThread(g_astMulticastUDPList[nLoop].pstUDPMulticastReceiver);
			ERRIFGOTO(result, _EXIT);
		}
		SAFEMEMFREE(g_astMulticastUDPList[nLoop].pstUDPMulticastReceiver->stReceiverSocket.pHeader);
		SAFEMEMFREE(g_astMulticastUDPList[nLoop].pstUDPMulticastReceiver);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

