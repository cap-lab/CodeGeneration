/*
 * UKUDPSocketMulticast.c
 *
 *  Created on: 2019. 6. 20.
 *      Author: wecracy
 */


#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <UCTime.h>
#include <UCBasic.h>
#include <UCAlloc.h>
#include <UCUDPSocket.h>
#include <UCThreadMutex.h>

#include <uem_data.h>

#include <UKUDPSocketMulticast.h>
#include <UKSharedMemoryMulticast.h>
#include <UKHostSystem.h>
#include <UKTask.h>

#include <uem_udp_data.h>
#include <uem_multicast_data.h>

// MulticastAPI_xxx functions are generated by UEM Translator
uem_result MulticastAPI_GetMulticastCommunicationTypeIndex(IN SMulticastGroup *pstMulticastGroup, IN EPortDirection eDirection, IN EMulticastCommunicationType eMulticastCommunicationType, OUT int *pnCommunicationTypeIndex);

static uem_result UKUDPSocketMulticast_AllocBuffer(SUDPSocket *pstUDPSocket, int nBufSize)
{
	uem_result result = ERR_UEM_UNKNOWN;

	if(pstUDPSocket->pHeader != NULL)
	{
		ERRASSIGNGOTO(result, ERR_UEM_REALLOCATE_BUFFER, _EXIT);
	}

	pstUDPSocket->pHeader = UCAlloc_malloc(nBufSize + MULTICAST_UDP_HEADER_SIZE);
	ERRMEMGOTO(pstUDPSocket->pHeader, result, _EXIT);

	pstUDPSocket->nHeaderLen = MULTICAST_UDP_HEADER_SIZE;
	pstUDPSocket->pBuffer = pstUDPSocket->pHeader + MULTICAST_UDP_HEADER_SIZE;
	pstUDPSocket->nBufLen = nBufSize;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static uem_result checkMessageOwner(SMulticastGroup *pstMulticastGroup, char *pHeader, int nReceivedDataLength, int *pnGroupNum)
{
	uem_result result = ERR_UEM_UNKNOWN;
	unsigned char ucGroupIndex;
    int nLoop = 0;

	if(nReceivedDataLength <= 0)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
	}

	UC_memcpy(&ucGroupIndex, pHeader, MULTICAST_UDP_HEADER_GROUP_ID_SIZE);

    for(nLoop = 0 ; nLoop < g_nMulticastGroupNum ; nLoop++)
    {
	    if (g_astMulticastGroups[nLoop].nMulticastGroupId == (int)ucGroupIndex)
        {
            *pnGroupNum = nLoop;
            break;
        }

    }

	if (nLoop == g_nMulticastGroupNum)
	{
		result = ERR_UEM_NOERROR;
	}
	else
	{
		result = ERR_UEM_SKIP_THIS;
	}

_EXIT:
	return result;
}

static void *multicastHandlingThread(void *pData)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SMulticastGroup *pstMulticastGroup = NULL;
	SUDPMulticast *pstUDPMulticastSocket = NULL;
	SUDPSocket *pstUDPSocket = NULL;
	int nReceivedDataLength;
	int nCommunicationTypeIndex = 0;
	int nBufSize = 0;

	pstMulticastGroup = (SMulticastGroup *) pData;

	result = MulticastAPI_GetMulticastCommunicationTypeIndex(pstMulticastGroup, PORT_DIRECTION_INPUT, MULTICAST_COMMUNICATION_TYPE_UDP, &nCommunicationTypeIndex);
	ERRIFGOTO(result, _EXIT);

	pstUDPMulticastSocket = (SUDPMulticast *) pstMulticastGroup->pMulticastRecvGateList[nCommunicationTypeIndex];

	nBufSize = pstMulticastGroup->nBufSize + MULTICAST_UDP_HEADER_SIZE;

	while(pstUDPMulticastSocket->bExit == FALSE)
	{
		// recieve
		result = UCUDPSocket_RecvFrom(pstUDPMulticastSocket->pstSocket->hSocket, "255.255.255.255", 10, nBufSize, pstUDPMulticastSocket->pstSocket->pHeader, &nReceivedDataLength);
		if(result == ERR_UEM_NET_TIMEOUT)
		{
			continue;
		}
		ERRIFGOTO(result, _EXIT);

		// check the groupName
		result = checkMessageOwner(pstMulticastGroup, pstUDPMulticastSocket->pstSocket->pHeader, nReceivedDataLength);
		if(result == ERR_UEM_SKIP_THIS)
		{
			continue;
		}
		ERRIFGOTO(result, _EXIT);

		// write  pstSharedMemoryMulticast->pDataStart, pBuffer, nDataToWrite
	    result = UCThreadMutex_Lock(g_astMulticastGroups[nLoop].pMulticastStruct->hMutex);
	    ERRIFGOTO(result, _EXIT);

		result = UKHostSystem_CopyToMemory(g_astMulticastGroups[nLoop].pMulticastStruct->pDataStart, pstUDPMulticastSocket->pstSocket->pBuffer, nReceivedDataLength);
		ERRIFGOTO(result, _EXIT_LOCK);
		pstMulticastGroup->pMulticastStruct->nDataLen = nReceivedDataLength;

_EXIT_LOCK:
	    UCThreadMutex_Unlock(g_astMulticastGroups[nLoop].pMulticastStruct->hMutex);
		ERRIFGOTO(result, _EXIT);
	}

_EXIT:
	if(result != ERR_UEM_NOERROR && pstUDPMulticastSocket->bExit == FALSE)
	{
		UEM_DEBUG_PRINT("channelReceiverHandlingThread is exited with error: %08x\n", result);
	}
	return NULL;
}


static uem_result createMulticastReceiverThread(SMulticastGroup *pstMulticastGroup)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SUDPMulticast *pstUDPMulticastSocket = NULL;
	int nCommunicationTypeIndex = 0;

	result = MulticastAPI_GetMulticastCommunicationTypeIndex(pstMulticastGroup, PORT_DIRECTION_INPUT, MULTICAST_COMMUNICATION_TYPE_UDP, &nCommunicationTypeIndex);
	ERRIFGOTO(result, _EXIT);

	pstUDPMulticastSocket = (SUDPMulticast *) pstMulticastGroup->pMulticastRecvGateList[nCommunicationTypeIndex];

	result = UCThread_Create(multicastHandlingThread, (void *) pstMulticastGroup, &(pstUDPMulticastSocket->hManagementThread));
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static uem_result destroyMulticastReceiverThread(SUDPMulticast *pstUDPMulticastSocket)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = UCThread_Destroy(&(pstUDPMulticastSocket->hManagementThread), FALSE, 0);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKUDPSocketMulticast_SocketInitialize(SUDPSocket *pstUDPSocket, int nPort, IN uem_bool bIsServer)
{
	SSocketInfo stSocketInfo;
	uem_result result = ERR_UEM_UNKNOWN;

	if (pstUDPSocket->hSocket == NULL)
	{
		stSocketInfo.enSocketType = SOCKET_TYPE_UDP;
		stSocketInfo.nPort = nPort;
		stSocketInfo.pszSocketPath = NULL;
		result = UCDynamicSocket_Create(&stSocketInfo, bIsServer, &pstUDPSocket->hSocket);
		ERRIFGOTO(result, _EXIT);
	}

	result = UCThreadMutex_Create(&(pstUDPSocket->hMutex));
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKUDPSocketMulticast_MakeHeader(IN SMulticastPort *pstMulticastPort, IN OUT unsigned char *pHeader)
{
	uem_result result = ERR_UEM_UNKNOWN;

	pHeader[0] = pstMulticastPort->pMulticastGroup->nMulticastGroupId;

	result = ERR_UEM_NOERROR;

	return result;
}

uem_result UKUDPSocketMulticast_WriteToBuffer(IN SMulticastPort *pstMulticastPort, IN unsigned char *pData, IN int nDataToWrite, OUT int *pnDataWritten)
{
	uem_result result = ERR_UEM_UNKNOWN;

	SUDPMulticast *pstUDPMulticastSocket = NULL;
	SUDPSocket *pstUDPSocket = NULL;
	unsigned char cBuffer[pstMulticastPort->pMulticastGroup->nBufSize + MULTICAST_UDP_HEADER_SIZE];
	int nCommunicationTypeIndex = 0;
	unsigned char *pHeader = cBuffer;
	unsigned char *pBuffer = cBuffer + MULTICAST_UDP_HEADER_SIZE;

	result = UKUDPSocketMulticast_MakeHeader(pstMulticastPort, pHeader);
	ERRIFGOTO(result, _EXIT);

	result = MulticastAPI_GetMulticastCommunicationTypeIndex(pstMulticastPort->pMulticastGroup, PORT_DIRECTION_OUTPUT, MULTICAST_COMMUNICATION_TYPE_UDP, &nCommunicationTypeIndex);
	ERRIFGOTO(result, _EXIT);

	pstUDPMulticastSocket = pstMulticastPort->pMulticastSendGateList[nCommunicationTypeIndex];
	pstUDPSocket = pstUDPMulticastSocket->pstSocket;
	result = pstMulticastPort->pstMemoryAccessAPI->fnCopyToMemory(pBuffer, pData, nDataToWrite);
	ERRIFGOTO(result, _EXIT);

	result = UCUDPSocket_Sendto(pstUDPSocket->hSocket, "255.255.255.255", 10, pHeader, MULTICAST_UDP_HEADER_SIZE + nDataToWrite, pnDataWritten);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKUDPSocketMulticast_Initialize(IN SMulticastGroup *pstMulticastGroup)
{
	uem_result result = ERR_UEM_UNKNOWN;

	SUDPMulticast *pstUDPMulticastSocket = NULL;
	SUDPSocket *pstUDPSocket = NULL;
	int nLoop = 0;
	int nCommunicationTypeIndex = 0;

	result = MulticastAPI_GetMulticastCommunicationTypeIndex(pstMulticastGroup, PORT_DIRECTION_INPUT, MULTICAST_COMMUNICATION_TYPE_UDP, &nCommunicationTypeIndex);
	ERRIFGOTO(result, _EXIT);

	if(nCommunicationTypeIndex >= 0)
	{
		pstUDPMulticastSocket = (SUDPMulticast *) pstMulticastGroup->pMulticastRecvGateList[nCommunicationTypeIndex];

		pstUDPMulticastSocket->pstSocket =(SUDPSocket *) UCAlloc_malloc(sizeof(SUDPSocket));

		pstUDPSocket = pstUDPMulticastSocket->pstSocket;

		result = UKUDPSocketMulticast_AllocBuffer(pstUDPSocket, pstMulticastGroup->nBufSize);
		ERRIFGOTO(result, _EXIT);

		pstUDPMulticastSocket->bExit = 0;
		pstUDPMulticastSocket->pstMulticastManager = pstMulticastGroup;

		result = UKUDPSocketMulticast_SocketInitialize(pstUDPSocket, ((SUDPInfo*)pstMulticastGroup->pstInputCommunicationInfo[nCommunicationTypeIndex].pAdditionalCommunicationInfo)->nPort, TRUE);
		ERRIFGOTO(result, _EXIT);

		result = UCDynamicSocket_Bind(pstUDPSocket->hSocket);
		ERRIFGOTO(result, _EXIT);

		// create receive thread
		result = createMulticastReceiverThread(pstMulticastGroup);
		ERRIFGOTO(result, _EXIT);
	}

	result = MulticastAPI_GetMulticastCommunicationTypeIndex(pstMulticastGroup, PORT_DIRECTION_OUTPUT, MULTICAST_COMMUNICATION_TYPE_UDP, &nCommunicationTypeIndex);
	ERRIFGOTO(result, _EXIT);
	if(nCommunicationTypeIndex >= 0)
	{
		for(nLoop = 0 ; nLoop < pstMulticastGroup->nOutputPortNum ; nLoop++)
		{
			pstUDPMulticastSocket = (SUDPMulticast *) pstMulticastGroup->pstOutputPort[nLoop].pMulticastSendGateList[nCommunicationTypeIndex];

			pstUDPMulticastSocket->pstSocket =(SUDPSocket *) UCAlloc_malloc(sizeof(SUDPSocket));

			pstUDPSocket = pstUDPMulticastSocket->pstSocket;

			pstUDPMulticastSocket->bExit = 0;

			pstUDPMulticastSocket->pstMulticastManager = &(pstMulticastGroup->pstOutputPort[nLoop]);

			result = UKUDPSocketMulticast_SocketInitialize(pstUDPSocket, ((SUDPInfo*)pstMulticastGroup->pstInputCommunicationInfo[nCommunicationTypeIndex].pAdditionalCommunicationInfo)->nPort, FALSE);
			ERRIFGOTO(result, _EXIT);
		}
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKUDPSocketMulticast_Finalize(IN SMulticastGroup *pstMulticastGroup)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;
	int nCommunicationTypeIndex = 0;
	SUDPMulticast *pstUDPMulticastSocket = NULL;

	result = MulticastAPI_GetMulticastCommunicationTypeIndex(pstMulticastGroup, PORT_DIRECTION_INPUT, MULTICAST_COMMUNICATION_TYPE_UDP, &nCommunicationTypeIndex);
	ERRIFGOTO(result, _EXIT);

	if (nCommunicationTypeIndex >= 0)
	{
		pstUDPMulticastSocket = (SUDPMulticast *) pstMulticastGroup->pMulticastRecvGateList[nCommunicationTypeIndex];

		pstUDPMulticastSocket->bExit = EXIT_FLAG_READ | EXIT_FLAG_WRITE;

		if (pstUDPMulticastSocket->hManagementThread != NULL)
		{
			result = destroyMulticastReceiverThread(pstUDPMulticastSocket);
			ERRIFGOTO(result, _EXIT);
		}

		UCAlloc_free(pstUDPMulticastSocket->pstSocket->hSocket);
		UCAlloc_free(pstUDPMulticastSocket->pstSocket);
	}

	result = MulticastAPI_GetMulticastCommunicationTypeIndex(pstMulticastGroup, PORT_DIRECTION_OUTPUT, MULTICAST_COMMUNICATION_TYPE_UDP, &nCommunicationTypeIndex);
	ERRIFGOTO(result, _EXIT);
	if (nCommunicationTypeIndex >= 0)
	{
		for (nLoop = 0; nLoop < pstMulticastGroup->nOutputPortNum; nLoop++)
		{
			pstUDPMulticastSocket = (SUDPMulticast *) pstMulticastGroup->pstOutputPort[nLoop].pMulticastSendGateList[nCommunicationTypeIndex];
			ERRIFGOTO(result, _EXIT);

			pstUDPMulticastSocket->bExit = EXIT_FLAG_READ | EXIT_FLAG_WRITE;

			UCAlloc_free(pstUDPMulticastSocket->pstSocket->hSocket);
			UCAlloc_free(pstUDPMulticastSocket->pstSocket);
		}
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

