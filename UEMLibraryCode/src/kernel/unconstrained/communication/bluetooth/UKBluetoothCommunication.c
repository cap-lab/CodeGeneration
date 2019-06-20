/*
 * UKBluetoothCommunication.c
 *
 *  Created on: 2019. 5. 23.
 *      Author: jej
 */


#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <UCDynamicSocket.h>

#include <UKVirtualCommunication.h>

#include <uem_bluetooth_data.h>

uem_result UKBluetoothCommunication_Create(OUT HVirtualSocket *phSocket, void *pSocketInfo)
{
	uem_result result = ERR_UEM_UNKNOWN;
	BluetoothAccessInfo *pstBluetoothInfo = NULL;
	SSocketInfo stSocketInfo;
	uem_bool bIsServer = FALSE;
	HSocket hSocket = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(phSocket, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	pstBluetoothInfo = (BluetoothAccessInfo *) pSocketInfo;

	stSocketInfo.enSocketType = SOCKET_TYPE_BLUETOOTH;
	stSocketInfo.nPort = 0;
	stSocketInfo.pszSocketPath = NULL;

	if(pstBluetoothInfo != NULL)
	{
		if(pstBluetoothInfo->pszTargetMacAddress != NULL)
		{
			stSocketInfo.pszSocketPath = pstBluetoothInfo->pszTargetMacAddress;
		}

		switch(pstBluetoothInfo->enType)
		{
		case PAIR_TYPE_SLAVE:
			bIsServer = TRUE;
			break;
		case PAIR_TYPE_MASTER:
			bIsServer = FALSE;
			break;
		default:
			ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
			break;
		}
	}
	else // client socket for accept
	{
		bIsServer = FALSE;
	}

	result = UCDynamicSocket_Create(&stSocketInfo, bIsServer, &hSocket);
	ERRIFGOTO(result, _EXIT);

	*phSocket = (HVirtualSocket *) hSocket;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


