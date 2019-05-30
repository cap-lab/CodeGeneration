/*
 * UKTCPCommunication.c
 *
 *  Created on: 2019. 5. 22.
 *      Author: jej
 */


#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <UCDynamicSocket.h>

#include <UKVirtualCommunication.h>

#include <uem_tcp_data.h>

#define UNUSED_PORT_NUM (1)


uem_result UKTCPCommunication_Create(OUT HVirtualSocket *phSocket, void *pSocketInfo)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STCPInfo *pstTCPInfo = NULL;
	SSocketInfo stSocketInfo;
	uem_bool bIsServer = FALSE;
	HSocket hSocket = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(phSocket, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	pstTCPInfo = (STCPInfo *) pSocketInfo;

	stSocketInfo.enSocketType = SOCKET_TYPE_TCP;

	if(pstTCPInfo != NULL)
	{
		stSocketInfo.nPort = pstTCPInfo->nPort;
		stSocketInfo.pszSocketPath = pstTCPInfo->pszIPAddress;

		switch(pstTCPInfo->enType)
		{
		case PAIR_TYPE_SERVER:
			bIsServer = TRUE;
			break;
		case PAIR_TYPE_CLIENT:
			bIsServer = FALSE;
			break;
		default:
			ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
			break;
		}
	}
	else // client socket for accept
	{
		stSocketInfo.nPort = UNUSED_PORT_NUM;
		stSocketInfo.pszSocketPath = NULL;

		bIsServer = FALSE;
	}

	result = UCDynamicSocket_Create(&stSocketInfo, bIsServer, &hSocket);
	ERRIFGOTO(result, _EXIT);

	*phSocket = (HVirtualSocket *) hSocket;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}



