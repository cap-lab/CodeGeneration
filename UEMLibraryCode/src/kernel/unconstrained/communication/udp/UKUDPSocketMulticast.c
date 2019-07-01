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
#include <UCThreadMutex.h>

#include <uem_data.h>

#include <UKMulticastMemory.h>
#include <UKTask.h>

#include <uem_udp_data.h>


#define CONNECT_TIMEOUT (3)
#define CONNECT_RETRY_COUNT (100)
#define SECOND_IN_MILLISECOND (1000)

static uem_result UKUDPSocketMulticast_AllocBuffer(SUDPSocket *pstUDPSocket, int nBufSize)
{
	uem_result result = ERR_UEM_UNKNOWN;

	if(pstUDPSocket->pBuffer != NULL)
	{
		ERRASSIGNGOTO(result, ERR_UEM_REALLOCATE_BUFFER, _EXIT);
	}

	pstUDPSocket->pBuffer = UCAlloc_malloc(nBufSize);
	ERRMEMGOTO(pstUDPSocket->pBuffer, result, _EXIT);

	pstUDPSocket->nBufLen = nBufSize;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static uem_result checkMessageOwner(SMulticastGroup *pstMulticastGroup, SUDPSocket *pstUDPSocket, int nReceivedDataLength)
{
	uem_result result = ERR_UEM_UNKNOWN;
	unsigned char ucGroupIndex;

	if(nReceivedDataLength <= 0)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
	}

	memcpy(&ucGroupIndex, pstUDPSocket->pBuffer, 1);

	if (pstMulticastGroup->nMulticastGroupId == (int)ucGroupIndex)
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
	SUDPSocket *pstUDPSocket = NULL;
	int nReceivedDataLength;
	int nWrittenDataLength;
	int nCommunicationTypeIndex = 0;

	pstMulticastGroup = (SMulticastGroup *) pData;

	result = MulticasatAPI_GetMulticastCommunicationTypeIndex(pstMulticastGroup, PORT_DIRECTION_INPUT, MULTICAST_COMMUNICATION_TYPE_UDP, &nCommunicationTypeIndex);
	ERRIFGOTO(result, _EXIT);

	pstUDPSocket = (SUDPSocket *) pstMulticastGroup->pMulticastRecvGateList[nCommunicationTypeIndex];

	while(pstUDPSocket->bExit == FALSE)
	{
		// recieve
		result = UCUDPSocket_RecvFrom(pstUDPSocket->hSocket, "255.255.255.255", 0, pstMulticastGroup->nBufSize, pstUDPSocket->pBuffer, &nReceivedDataLength);
		if(result == ERR_UEM_NET_TIMEOUT)
		{
			continue;
		}
		ERRIFGOTO(result, _EXIT);

		// check the groupName
		result = checkMessageOwner(pstMulticastGroup, pstUDPSocket, nReceivedDataLength);
		if(result == ERR_UEM_SKIP_THIS)
		{
			continue;
		}
		ERRIFGOTO(result, _EXIT);

		// write
		result = UKSharedMemoryChannel_WriteToBuffer(pstMulticastGroup, pstUDPSocket->pBuffer, nReceivedDataLength, nWrittenDataLength);
		if(result != ERR_UEM_NOERROR)
		{
			result = UKUEMProtocol_SetResultMessage(pstUDPSocket, ERR_UEMPROTOCOL_INTERNAL, 0);
		}
		ERRIFGOTO(result, _EXIT);
	}

_EXIT:
	if(result != ERR_UEM_NOERROR && pstUDPSocket->bExit == FALSE)
	{
		UEM_DEBUG_PRINT("channelReceiverHandlingThread is exited with error: %08x\n", result);
	}
	return NULL;
}


static uem_result createMulticastReceiverThread(SMulticastGroup *pstMulticastGroup)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SUDPSocket *pstUDPSocket = NULL;
	int nCommunicationTypeIndex = 0;

	result = MulticasatAPI_GetMulticastCommunicationTypeIndex(pstMulticastGroup, PORT_DIRECTION_INPUT, MULTICAST_COMMUNICATION_TYPE_UDP, &nCommunicationTypeIndex);
	ERRIFGOTO(result, _EXIT);

	pstUDPSocket = (SUDPSocket *) pstMulticastGroup->pMulticastRecvGateList[nCommunicationTypeIndex];

	result = UCThread_Create(multicastHandlingThread, (void *) pstMulticastGroup, &(pstUDPSocket->hManagementThread));
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


static uem_result destroyMulticastReceiverThread(SUDPSocket *pstUDPSocket)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = UCThread_Destroy(&(pstUDPSocket->hManagementThread), FALSE, 0);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKUDPSocketMulticast_SocketInitialize(SUDPSocket *pstUDPSocket, int nPort)
{
	SSocketInfo stSocketInfo;
	uem_result result = ERR_UEM_UNKNOWN;

	if (pstUDPSocket->hSocket == NULL) {
		stSocketInfo.enSocketType = SOCKET_TYPE_UDP;
		stSocketInfo.nPort = nPort;
		stSocketInfo.pszSocketPath = NULL;
		result = UCDynamicSocket_Create(&stSocketInfo, FALSE, pstUDPSocket->hSocket);
		ERRIFGOTO(result, _EXIT);
	}

	result = UCThreadMutex_Create(&(pstUDPSocket->hMutex));
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}
uem_result UKUDPSocketMulticast_WriteToBuffer(IN SMulticastPort *pstMulticastPort, IN unsigned char *pData, IN int nDataToWrite, OUT int *pnDataWritten)
{
	uem_result result = ERR_UEM_UNKNOWN;

	SUDPSocket *pstUDPSocket = NULL;
	char cBuffer[pstMulticastPort->pMulticastGroup->nBufSize] = {(char)pstMulticastPort->pMulticastGroup->nMulticastGroupId,};
	int nCommunicationTypeIndex = 0;

	result = MulticasatAPI_GetMulticastCommunicationTypeIndex(pstMulticastPort->pMulticastGroup, PORT_DIRECTION_OUTPUT, MULTICAST_COMMUNICATION_TYPE_UDP, &nCommunicationTypeIndex);
	ERRIFGOTO(result, _EXIT);
	pstUDPSocket = pstMulticastPort->pMulticastSendGateList[nCommunicationTypeIndex];
	result = pstMulticastPort->pstMemoryAccessAPI->fnCopyToMemory(cBuffer + 1, pData, nDataToWrite);
	ERRIFGOTO(result, _EXIT);
	result = UCUDPSocket_Sendto(pstUDPSocket->hSocket, "255.255.255.255", 0, cBuffer, nDataToWrite, pnDataWritten);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}
uem_result UKUDPSocketMulticast_Initialize(SMulticastGroup *pstMulticastGroup)
{
	uem_result result = ERR_UEM_UNKNOWN;

	SUDPSocket *pstUDPSocket = NULL;
	HSocket hSocket = NULL;
	int nLoop = 0;
	int nCommunicationTypeIndex = 0;

	result = MulticasatAPI_GetMulticastCommunicationTypeIndex(pstMulticastGroup, PORT_DIRECTION_INPUT, MULTICAST_COMMUNICATION_TYPE_UDP, &nCommunicationTypeIndex);
	ERRIFGOTO(result, _EXIT);

	if(nCommunicationTypeIndex >= 0)
	{
		pstUDPSocket = (SUDPSocket *) pstMulticastGroup->pMulticastRecvGateList[nCommunicationTypeIndex];

		if (pstUDPSocket->pBuffer == NULL)
		{
			UKUDPSocketMulticast_AllocBuffer(pstUDPSocket, pstMulticastGroup->nBufSize);
		}
		pstUDPSocket->bExit = 0;
		pstUDPSocket->pstMulticastManager = pstMulticastGroup;

		result = UKUDPSocketMulticast_SocketInitialize(pstUDPSocket, ((SUDPInfo*)pstMulticastGroup->pstInputCommunicationInfo[nCommunicationTypeIndex]->pAdditionalCommunicationInfo)->nPort);
		ERRIFGOTO(result, _EXIT);

		// create receive thread
		result = createMulticastReceiverThread(pstMulticastGroup);
		ERRIFGOTO(result, _EXIT);
	}

	result = GetMulticastCommunicationTypeIndex(pstMulticastGroup, PORT_DIRECTION_OUTPUT, MULTICAST_COMMUNICATION_TYPE_UDP, &nCommunicationTypeIndex);
	ERRIFGOTO(result, _EXIT);
	if(nCommunicationTypeIndex >= 0)
	{
		for(nLoop = 0 ; nLoop < pstMulticastGroup->nOutputPortNum ; nLoop++)
		{

			pstUDPSocket = (SUDPSocket *) pstMulticastGroup->pstOutputPort[nLoop]->pMulticastSendGateList[nCommunicationTypeIndex];
			ERRIFGOTO(result, _EXIT);

			if (pstUDPSocket->pBuffer == NULL)
			{
				UKUDPSocketMulticast_AllocBuffer(pstUDPSocket, pstMulticastGroup->nBufSize);
			}
			pstUDPSocket->bExit = 0;
			pstUDPSocket->pstMulticastManager = pstMulticastGroup->pstOutputPort[nLoop];

			result = UKUDPSocketMulticast_SocketInitialize(pstUDPSocket, ((SUDPInfo*)pstMulticastGroup->pstInputCommunicationInfo[nCommunicationTypeIndex]->pAdditionalCommunicationInfo)->nPort);
			ERRIFGOTO(result, _EXIT);
		}
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKUDPSocket_Finalize(SMulticastGroup *pstMulticastGroup)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;
	int nCommunicationTypeIndex = 0;
	SUDPSocket *pstUDPSocket = NULL;

	result = MulticasatAPI_GetMulticastCommunicationTypeIndex(pstMulticastGroup, PORT_DIRECTION_INPUT, MULTICAST_COMMUNICATION_TYPE_UDP, &nCommunicationTypeIndex);
	ERRIFGOTO(result, _EXIT);

	if (nCommunicationTypeIndex >= 0) {
		pstUDPSocket = (SUDPSocket *) pstMulticastGroup->pMulticastRecvGateList[nCommunicationTypeIndex];

		UKUDPSocketChannel_SetExit(pstMulticastGroup, EXIT_FLAG_READ | EXIT_FLAG_WRITE);

		if (pstUDPSocket->hManagementThread != NULL) {
			result = destroyMulticastReceiverThread(pstUDPSocket);
			ERRIFGOTO(result, _EXIT);
		}
	}

	result = GetMulticastCommunicationTypeIndex(pstMulticastGroup, PORT_DIRECTION_OUTPUT, MULTICAST_COMMUNICATION_TYPE_UDP, &nCommunicationTypeIndex);
	ERRIFGOTO(result, _EXIT);
	if (nCommunicationTypeIndex >= 0) {
		for (nLoop = 0; nLoop < pstMulticastGroup->nOutputPortNum; nLoop++) {
			pstUDPSocket = (SUDPSocket *) pstMulticastGroup->pstOutputPort[nLoop]->pMulticastSendGateList[nCommunicationTypeIndex];
			ERRIFGOTO(result, _EXIT);

			UKUDPSocketChannel_SetExit(pstMulticastGroup, EXIT_FLAG_READ | EXIT_FLAG_WRITE);
		}
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

