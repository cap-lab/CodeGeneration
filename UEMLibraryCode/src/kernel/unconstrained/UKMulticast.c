/*
 * UKMulticast.c
 *
 *  Created on: 2019. 6. 20.
 *      Author: wecracy
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>
#include <UCString.h>

#include <uem_multicast_data.h>

uem_result MulticastAPI_SetSocketAPIs();

uem_result UKMulticast_Initialize()
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nAPIIndex = 0;
	int nGroupIndex = 0;
	int nPortIndex = 0;

	result = MulticastAPI_SetSocketAPIs();
	ERRIFGOTO(result, _EXIT);

	for(nAPIIndex = 0; nAPIIndex < g_nMulticastAPINum ; nAPIIndex++)
	{
		if(g_astMulticastAPIList[nAPIIndex]->fnAPIInitialize != NULL)
		{
			result = g_astMulticastAPIList[nAPIIndex]->fnAPIInitialize();
			ERRIFGOTO(result, _EXIT);
		}
	}
    
	for(nGroupIndex = 0; nGroupIndex < g_nMulticastGroupNum; nGroupIndex++)
	{
		for(nAPIIndex = 0; nAPIIndex < g_astMulticastGroups[nGroupIndex].nCommunicationTypeNum ; nAPIIndex++)
		{
			if(g_astMulticastGroups[nGroupIndex].astCommunicationList[nAPIIndex].pstMulticastAPI->fnGroupInitialize != NULL)
			{
				g_astMulticastGroups[nGroupIndex].astCommunicationList[nAPIIndex].pstMulticastAPI->fnGroupInitialize(&(g_astMulticastGroups[nGroupIndex]));
				ERRIFGOTO(result, _EXIT);
			}
		}
	}

	for(nGroupIndex = 0; nGroupIndex < g_nMulticastGroupNum; nGroupIndex++)
	{
		for(nPortIndex = 0 ; nPortIndex < g_astMulticastGroups[nGroupIndex].nInputPortNum ; nPortIndex++)
		{
			SMulticastPort *pstInputPort = &g_astMulticastGroups[nGroupIndex].astInputPort[nPortIndex];
			pstInputPort->pstMulticastGroup = &g_astMulticastGroups[nGroupIndex];
			for(nAPIIndex = 0; nAPIIndex < pstInputPort->nCommunicationTypeNum ; nAPIIndex++)
			{
				if(pstInputPort->astCommunicationList[nAPIIndex].pstMulticastAPI->fnPortInitialize != NULL)
				{
					pstInputPort->astCommunicationList[nAPIIndex].pstMulticastAPI->fnPortInitialize(pstInputPort);
					ERRIFGOTO(result, _EXIT);
				}
			}
		}

		for(nPortIndex = 0 ; nPortIndex < g_astMulticastGroups[nGroupIndex].nOutputPortNum ; nPortIndex++)
		{
			SMulticastPort *pstOutputPort = &g_astMulticastGroups[nGroupIndex].astOutputPort[nPortIndex];
			pstOutputPort->pstMulticastGroup = &g_astMulticastGroups[nGroupIndex];
			for(nAPIIndex = 0; nAPIIndex < pstOutputPort->nCommunicationTypeNum ; nAPIIndex++)
			{
				if(pstOutputPort->astCommunicationList[nAPIIndex].pstMulticastAPI->fnPortInitialize != NULL)
				{
					pstOutputPort->astCommunicationList[nAPIIndex].pstMulticastAPI->fnPortInitialize(pstOutputPort);
					ERRIFGOTO(result, _EXIT);
				}
			}
		}
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static uem_result UKMulticast_GetMulticastGroupById(IN int nMulticastGroupId, OUT SMulticastGroup **pstMulticastGroup)
{
	uem_result result = ERR_UEM_NOT_FOUND;
	int nLoop = 0;

	for(nLoop = 0; nLoop < g_nMulticastGroupNum; nLoop++)
	{
		if(g_astMulticastGroups[nLoop].nMulticastGroupId == nMulticastGroupId)
		{
			*pstMulticastGroup = &g_astMulticastGroups[nLoop];
			result = ERR_UEM_NOERROR;
			break;
		}
	}

	return result;
}

static uem_result isMulticastPortTaskIdAndMulticastPortNameEqual(IN SMulticastPort *pstTopMulticastPort, IN uem_string strPortName, IN int nTaskId, OUT uem_bool *bIsMatch)
{
	uem_string_struct stStructPortName;
	uem_result result = ERR_UEM_UNKNOWN;
	SMulticastPort *pstMulticastPort = NULL;

	*bIsMatch = FALSE;
	pstMulticastPort = pstTopMulticastPort;

	if (pstMulticastPort->nTaskId == nTaskId) {
		result = UCString_New(&stStructPortName, (char *) pstMulticastPort->pszPortName, UEMSTRING_CONST);
		ERRIFGOTO(result, _EXIT);
		if (UCString_IsEqual(strPortName, &stStructPortName) == TRUE) {
			*bIsMatch = TRUE;
		}
	}
	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKMulticast_GetMulticastPortByMulticastGroupIdAndMulticastPortId(IN int nMulticastGroupId, IN int nMulticastPortId, IN EPortDirection eDirection, OUT SMulticastPort **pstMulticastPort)
{
	uem_result result = ERR_UEM_NOT_FOUND;
	SMulticastGroup *pstMulticastGroup = NULL;
	SMulticastPort *pstMulticastPortList = NULL;
	int nMulticastPortNum = 0;
	int nMulticastPortIndex = 0;


	result = UKMulticast_GetMulticastGroupById(nMulticastGroupId, &pstMulticastGroup);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOT_FOUND;

	if(eDirection == PORT_DIRECTION_INPUT)
	{
		pstMulticastPortList = pstMulticastGroup->astInputPort;
		nMulticastPortNum = pstMulticastGroup->nInputPortNum;
	}
	else
	{
		pstMulticastPortList = pstMulticastGroup->astOutputPort;
		nMulticastPortNum = pstMulticastGroup->nOutputPortNum;
	}
	for(nMulticastPortIndex = 0 ; nMulticastPortIndex < nMulticastPortNum ; nMulticastPortIndex++)
	{
		if(pstMulticastPortList[nMulticastPortIndex].nMulticastPortId == nMulticastPortId)
		{
			*pstMulticastPort = &pstMulticastPortList[nMulticastPortIndex];
			result = ERR_UEM_NOERROR;
			break;
		}
	}

_EXIT:
	return result;
}

uem_result UKMulticast_GetMulticastGroupIdByTaskAndPortName(IN int nTaskId, IN const char *szPortName, OUT int *pnMulticastGroupId)
{
	uem_result result = ERR_UEM_NOT_FOUND;
	int nLoop = 0;
	int nPortIndex = 0;
	uem_bool bFound = FALSE;
	uem_string_struct stArgPortName;

	*pnMulticastGroupId = INVALID_MULTICAST_GROUP_ID;

	result = UCString_New(&stArgPortName, (char*)szPortName, UEMSTRING_CONST);
	ERRIFGOTO(result, _EXIT);

	for(nLoop = 0; nLoop < g_nMulticastGroupNum; nLoop++)
	{
		for(nPortIndex = 0 ; nPortIndex < g_astMulticastGroups[nLoop].nInputPortNum ; nPortIndex++)
		{
			uem_bool bIsMatch;
			result = isMulticastPortTaskIdAndMulticastPortNameEqual(&(g_astMulticastGroups[nLoop].astInputPort[nPortIndex]), &stArgPortName, nTaskId, &bIsMatch);
			ERRIFGOTO(result, _EXIT);
			if (bIsMatch == TRUE)
			{
				*pnMulticastGroupId = g_astMulticastGroups[nLoop].nMulticastGroupId;
				bFound = TRUE;
				break;
			}
		}
		if(bFound == TRUE)
		{
			break;
		}
		for(nPortIndex = 0 ; nPortIndex < g_astMulticastGroups[nLoop].nOutputPortNum ; nPortIndex++)
		{
			uem_bool bIsMatch;
			result = isMulticastPortTaskIdAndMulticastPortNameEqual(&(g_astMulticastGroups[nLoop].astOutputPort[nPortIndex]),&stArgPortName, nTaskId, &bIsMatch);
			ERRIFGOTO(result, _EXIT);
			if (bIsMatch == TRUE)
			{
				*pnMulticastGroupId = g_astMulticastGroups[nLoop].nMulticastGroupId;
				bFound = TRUE;
				break;
			}
		}
	}
	if(bFound == TRUE)
	{
		result = ERR_UEM_NOERROR;
	}
_EXIT:
	return result;
}

uem_result UKMulticast_GetMulticastPortIdByTaskAndPortName(IN int nTaskId, IN const char *szPortName, OUT int *pnMulticastPortId)
{
	uem_result result = ERR_UEM_NOT_FOUND;
	int nLoop = 0;
	int nPortIndex = 0;
	uem_bool bFound = FALSE;
	uem_string_struct stArgPortName;

	*pnMulticastPortId = INVALID_MULTICAST_GROUP_ID;

	result = UCString_New(&stArgPortName, (char *)szPortName, UEMSTRING_CONST);
	ERRIFGOTO(result, _EXIT);

	for(nLoop = 0; nLoop < g_nMulticastGroupNum; nLoop++)
	{
		for(nPortIndex = 0 ; nPortIndex < g_astMulticastGroups[nLoop].nInputPortNum ; nPortIndex++)
		{
			uem_bool bIsMatch = FALSE;
			result = isMulticastPortTaskIdAndMulticastPortNameEqual(&(g_astMulticastGroups[nLoop].astInputPort[nPortIndex]), &stArgPortName, nTaskId, &bIsMatch);
			ERRIFGOTO(result, _EXIT);
			if (bIsMatch == TRUE)
			{
				*pnMulticastPortId = g_astMulticastGroups[nLoop].astInputPort[nPortIndex].nMulticastPortId;
				bFound = TRUE;
				break;
			}
		}
		if(bFound == TRUE)
		{
			break;
		}
		for(nPortIndex = 0 ; nPortIndex < g_astMulticastGroups[nLoop].nOutputPortNum ; nPortIndex++)
		{
			uem_bool bIsMatch = FALSE;
			result = isMulticastPortTaskIdAndMulticastPortNameEqual(&(g_astMulticastGroups[nLoop].astOutputPort[nPortIndex]),&stArgPortName, nTaskId, &bIsMatch);
			ERRIFGOTO(result, _EXIT);
			if (bIsMatch == TRUE)
			{
				*pnMulticastPortId = g_astMulticastGroups[nLoop].astOutputPort[nPortIndex].nMulticastPortId;
				bFound = TRUE;
				break;
			}
		}
	}
	if(bFound == TRUE)
	{
		result = ERR_UEM_NOERROR;
	}
_EXIT:
	return result;
}

uem_result UKMulticast_GetCommunication(IN SMulticastCommunication *astCommunicationList, IN int nCommunicationTypeNum, IN EMulticastCommunicationType enCommunicationType, OUT SMulticastCommunication **pstCommunication)
{
	uem_result result = ERR_UEM_NOT_FOUND;
	int nLoop = 0;
#ifdef ARGUMENT_CHECK
	if(astCommunicationList == NULL)
	{
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
	if(pstCommunication == NULL)
	{
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
#endif

	for(nLoop = 0 ; nLoop < nCommunicationTypeNum ; nLoop++)
	{
		if(astCommunicationList[nLoop].enCommunicationType == enCommunicationType)
		{
			*pstCommunication = &astCommunicationList[nLoop];
			result = ERR_UEM_NOERROR;
			break;
		}
	}

_EXIT:
	return result;
}

uem_result UKMulticast_WriteToBuffer(IN int nMulticastGroupId, IN int nMulticastPortId, IN unsigned char *pBuffer, IN int nDataToWrite, OUT int *pnDataWritten)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SMulticastPort *pstMulticastPort = NULL;
	int nAPIIndex = 0;
#ifdef ARGUMENT_CHECK
	if(pBuffer == NULL)
	{
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
#endif
	result = UKMulticast_GetMulticastPortByMulticastGroupIdAndMulticastPortId(nMulticastGroupId, nMulticastPortId, PORT_DIRECTION_OUTPUT, &pstMulticastPort);
	ERRIFGOTO(result, _EXIT);

	if(nDataToWrite > pstMulticastPort->pstMulticastGroup->nBufSize)
	{
		nDataToWrite = pstMulticastPort->pstMulticastGroup->nBufSize;
	}

	for(nAPIIndex = 0 ; nAPIIndex < pstMulticastPort->nCommunicationTypeNum ; nAPIIndex++)
	{
		if(pstMulticastPort->astCommunicationList[nAPIIndex].pstMulticastAPI->fnWriteToBuffer != NULL)
		{
			result = pstMulticastPort->astCommunicationList[nAPIIndex].pstMulticastAPI->fnWriteToBuffer(pstMulticastPort, pBuffer, nDataToWrite, pnDataWritten);
			ERRIFGOTO(result, _EXIT);
		}
	}

	// to preserve, ERR_UEM_SUSPEND, do not set UEM_NOERROR here
_EXIT:
	return result;
}

uem_result UKMulticast_ReadFromBuffer(IN int nMulticastGroupId, IN int nMulticastPortId, IN OUT unsigned char *pBuffer, IN int nDataToRead, OUT int *pnDataRead)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SMulticastPort *pstMulticastPort = NULL;
	int nAPIIndex = 0;
#ifdef ARGUMENT_CHECK
	if(pBuffer == NULL)
	{
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
#endif
	result = UKMulticast_GetMulticastPortByMulticastGroupIdAndMulticastPortId(nMulticastGroupId, nMulticastPortId, PORT_DIRECTION_INPUT, &pstMulticastPort);
	ERRIFGOTO(result, _EXIT);

	for(nAPIIndex = 0 ; nAPIIndex < pstMulticastPort->nCommunicationTypeNum ; nAPIIndex++)
	{
		if(pstMulticastPort->astCommunicationList[nAPIIndex].pstMulticastAPI->fnReadFromBuffer != NULL)
		{
			result = pstMulticastPort->astCommunicationList[nAPIIndex].pstMulticastAPI->fnReadFromBuffer(pstMulticastPort, pBuffer, nDataToRead, pnDataRead);
			ERRIFGOTO(result, _EXIT);
		}
	}
_EXIT:

	return result;
}

uem_result UKMulticast_GetMulticastGroupSize(IN int nMulticastGroupId, OUT int *pnMulticastGroupSize)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SMulticastGroup *pstMulticastGroup = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(pnMulticastGroupSize, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	result = UKMulticast_GetMulticastGroupById(nMulticastGroupId, &pstMulticastGroup);
	ERRIFGOTO(result, _EXIT);

	*pnMulticastGroupSize = pstMulticastGroup->nBufSize;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKMulticast_Finalize()
{
	uem_result result = ERR_UEM_NOERROR;
	int nAPIIndex = 0;
	int nGroupIndex = 0;
	int nPortIndex = 0;

	for(nGroupIndex = 0; nGroupIndex < g_nMulticastGroupNum; nGroupIndex++)
	{
		for(nPortIndex = 0 ; nPortIndex < g_astMulticastGroups[nGroupIndex].nInputPortNum ; nPortIndex++)
		{
			SMulticastPort *pstInputPort = &g_astMulticastGroups[nGroupIndex].astInputPort[nPortIndex];
			for(nAPIIndex = 0; nAPIIndex < pstInputPort->nCommunicationTypeNum ; nAPIIndex++)
			{
				if(pstInputPort->astCommunicationList[nAPIIndex].pstMulticastAPI->fnPortFinalize != NULL)
				{
					pstInputPort->astCommunicationList[nAPIIndex].pstMulticastAPI->fnPortFinalize(pstInputPort);
				}
			}
		}
		for(nPortIndex = 0 ; nPortIndex < g_astMulticastGroups[nGroupIndex].nOutputPortNum ; nPortIndex++)
		{
			SMulticastPort *pstOutputPort = &g_astMulticastGroups[nGroupIndex].astOutputPort[nPortIndex];
			for(nAPIIndex = 0; nAPIIndex < pstOutputPort->nCommunicationTypeNum ; nAPIIndex++)
			{
				if(pstOutputPort->astCommunicationList[nAPIIndex].pstMulticastAPI->fnPortFinalize != NULL)
				{
					pstOutputPort->astCommunicationList[nAPIIndex].pstMulticastAPI->fnPortFinalize(pstOutputPort);
				}
			}
		}
	}

	for(nGroupIndex = 0; nGroupIndex < g_nMulticastGroupNum; nGroupIndex++)
	{
		for(nAPIIndex = 0; nAPIIndex < g_astMulticastGroups[nGroupIndex].nCommunicationTypeNum ; nAPIIndex++)
		{
			if(g_astMulticastGroups[nGroupIndex].astCommunicationList[nAPIIndex].pstMulticastAPI->fnGroupFinalize != NULL)
			{
				g_astMulticastGroups[nGroupIndex].astCommunicationList[nAPIIndex].pstMulticastAPI->fnGroupFinalize(&(g_astMulticastGroups[nGroupIndex]));
			}
		}
	}

	for(nAPIIndex = 0; nAPIIndex < g_nMulticastAPINum ; nAPIIndex++)
	{
		if(g_astMulticastAPIList[nAPIIndex]->fnAPIFinalize != NULL)
		{
			g_astMulticastAPIList[nAPIIndex]->fnAPIFinalize();
		}
	}

	return result;
}
