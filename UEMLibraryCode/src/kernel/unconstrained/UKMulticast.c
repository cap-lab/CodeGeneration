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

#include <UKSharedMemoryMulticast.h>
#include <UKTask.h>

// MulticastAPI_xxx functions are generated by UEM Translator
uem_result MulticastAPI_GetAPIStructure(IN SMulticastGroup *pstMulticastGroup, OUT SMulticastAPI **pstMulticastAPI, OUT int *pnAPINum);
uem_result MulticastAPI_GetAPIStructureFromCommunicationType(IN SMulticastGroup *pstMulticastGroup, IN EPortDirection eDirection,  OUT SMulticastAPI **pstMulticastAPI, OUT int *pnAPINum);
uem_result MulticastAPI_GetMulticastCommunicationTypeIndex(IN SMulticastGroup *pstMulticastGroup, IN EPortDirection eDirection, IN EMulticastCommunicationType eMulticastCommunicationType, OUT int *pnCommunicationTypeIndex);
uem_result MulticastAPI_SetSocketAPIs();

uem_result UKMulticast_Initialize()
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;
	int nAPIIndex = 0;
	int nAPINum = 0;
	int nPortNum = 0;
	SMulticastAPI *pstMulticastAPI[g_nMulticastAPINum];

	result = MulticastAPI_SetSocketAPIs();
	ERRIFGOTO(result, _EXIT);

	for(nLoop = 0; nLoop < g_nMulticastAPINum ; nLoop++)
	{
		if(g_astMulticastAPIList[nLoop]->fnAPIInitialize != NULL)
		{
			result = g_astMulticastAPIList[nLoop]->fnAPIInitialize();
			ERRIFGOTO(result, _EXIT);
		}
	}
    
	for(nLoop = 0; nLoop < g_nMulticastGroupNum; nLoop++)
	{
		result = MulticastAPI_GetAPIStructure(&(g_astMulticastGroups[nLoop]), pstMulticastAPI, &nAPINum);
		if (result == ERR_UEM_NOERROR) {
			for (nAPIIndex = 0; nAPIIndex < nAPINum; nAPIIndex++) {
				if (pstMulticastAPI[nAPIIndex]->fnInitialize != NULL) {
					pstMulticastAPI[nAPIIndex]->fnInitialize(&(g_astMulticastGroups[nLoop]));
				}
			}
		}
	}

	for(nLoop = 0; nLoop < g_nMulticastGroupNum; nLoop++)
	{
		SMulticastPort *pstInputPort = g_astMulticastGroups[nLoop].pstInputPort;
		for(nPortNum = 0 ; nPortNum < g_astMulticastGroups[nLoop].nInputPortNum ; nPortNum++)
		{
			pstInputPort[nPortNum].pMulticastGroup = &g_astMulticastGroups[nLoop];
		}

		SMulticastPort *pstOutputPort = g_astMulticastGroups[nLoop].pstOutputPort;
		for(nPortNum = 0 ; nPortNum < g_astMulticastGroups[nLoop].nOutputPortNum ; nPortNum++)
		{
			pstOutputPort[nPortNum].pMulticastGroup = &g_astMulticastGroups[nLoop];
		}
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKMulticast_GetMulticastGroupIndexById(IN int nMulticastGroupId, OUT int *pnMulticastGroupIndex)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;
	*pnMulticastGroupIndex = INVALID_MULTICAST_GROUP_ID;

	for(nLoop = 0; nLoop < g_nMulticastGroupNum; nLoop++)
	{
		if(g_astMulticastGroups[nLoop].nMulticastGroupId == nMulticastGroupId)
		{
			*pnMulticastGroupIndex = nLoop;
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

uem_result UKMulticast_GetMulticastPortIndexByMulticastGroupIdAndMulticastPortId(IN int nMulticastGroupId, IN int nMulticastPortId, IN EPortDirection eDirection, OUT int *pnMulticastPortIndex)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nMulticastPortIndex = 0;
	int nMulticastGroupIndex = 0;
	*pnMulticastPortIndex = INVALID_MULTICAST_PORT_ID;

	result = UKMulticast_GetMulticastGroupIndexById(nMulticastGroupId, &nMulticastGroupIndex);
	ERRIFGOTO(result, _EXIT);
	if(eDirection == PORT_DIRECTION_INPUT)
	{
		for(nMulticastPortIndex = 0 ; nMulticastPortIndex < g_astMulticastGroups[nMulticastGroupIndex].nInputPortNum ; nMulticastPortIndex++)
		{
			if(g_astMulticastGroups[nMulticastGroupIndex].pstInputPort[nMulticastPortIndex].nMulticastPortId == nMulticastPortId)
			{
				*pnMulticastPortIndex = nMulticastPortIndex;
				result = ERR_UEM_NOERROR;
				break;
			}
		}
	}
	else
	{
		for(nMulticastPortIndex = 0 ; nMulticastPortIndex < g_astMulticastGroups[nMulticastGroupIndex].nOutputPortNum ; nMulticastPortIndex++)
		{
			if(g_astMulticastGroups[nMulticastGroupIndex].pstOutputPort[nMulticastPortIndex].nMulticastPortId == nMulticastPortId)
			{
				*pnMulticastPortIndex = nMulticastPortIndex;
				result = ERR_UEM_NOERROR;
				break;
			}
		}
	}
_EXIT:
	return result;
}

uem_result UKMulticast_Clear(IN int nMulticastGroupId)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nIndex = 0;
	SMulticastAPI *pstMulticastAPI[g_nMulticastAPINum];
	int nAPINum = 0;
	int nAPIIndex = 0;

	result = UKMulticast_GetMulticastGroupIndexById(nMulticastGroupId, &nIndex);
	IFVARERRASSIGNGOTO(nIndex, INVALID_MULTICAST_GROUP_ID, result, ERR_UEM_INVALID_PARAM, _EXIT);

	result = MulticastAPI_GetAPIStructureFromCommunicationType(&(g_astMulticastGroups[nIndex]), PORT_DIRECTION_INPUT, pstMulticastAPI, &nAPINum);
	ERRIFGOTO(result, _EXIT);
	for(nAPIIndex = 0 ; nAPIIndex < nAPINum ; nAPIIndex++)
	{
		if (pstMulticastAPI[nAPIIndex]->fnClear == NULL) {
			ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_CONTROL, _EXIT);
		}
		result = pstMulticastAPI[nAPIIndex]->fnClear(&(g_astMulticastGroups[nIndex]));
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKMulticast_GetMulticastGroupIdByTaskAndPortName(IN int nTaskId, IN char *szPortName, OUT int *pnMulticastGroupId)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;
	int nPortIndex = 0;
	uem_bool bFound = FALSE;
	uem_string_struct stArgPortName;

	*pnMulticastGroupId = INVALID_MULTICAST_GROUP_ID;

	result = UCString_New(&stArgPortName, szPortName, UEMSTRING_CONST);
	ERRIFGOTO(result, _EXIT);

	for(nLoop = 0; nLoop < g_nMulticastGroupNum; nLoop++)
	{
		for(nPortIndex = 0 ; nPortIndex < g_astMulticastGroups[nLoop].nInputPortNum ; nPortIndex++)
		{
			uem_bool bIsMatch;
			result = isMulticastPortTaskIdAndMulticastPortNameEqual(&(g_astMulticastGroups[nLoop].pstInputPort[nPortIndex]), &stArgPortName, nTaskId, &bIsMatch);
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
			result = isMulticastPortTaskIdAndMulticastPortNameEqual(&(g_astMulticastGroups[nLoop].pstOutputPort[nPortIndex]),&stArgPortName, nTaskId, &bIsMatch);
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

uem_result UKMulticast_GetMulticastPortIdByTaskAndPortName(IN int nTaskId, IN char *szPortName, OUT int *pnMulticastPortId)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;
	int nPortIndex = 0;
	uem_bool bFound = FALSE;
	uem_string_struct stArgPortName;

	*pnMulticastPortId = INVALID_MULTICAST_GROUP_ID;

	result = UCString_New(&stArgPortName, szPortName, UEMSTRING_CONST);
	ERRIFGOTO(result, _EXIT);

	for(nLoop = 0; nLoop < g_nMulticastGroupNum; nLoop++)
	{
		for(nPortIndex = 0 ; nPortIndex < g_astMulticastGroups[nLoop].nInputPortNum ; nPortIndex++)
		{
			uem_bool bIsMatch = FALSE;
			result = isMulticastPortTaskIdAndMulticastPortNameEqual(&(g_astMulticastGroups[nLoop].pstInputPort[nPortIndex]), &stArgPortName, nTaskId, &bIsMatch);
			ERRIFGOTO(result, _EXIT);
			if (bIsMatch == TRUE)
			{
				*pnMulticastPortId = g_astMulticastGroups[nLoop].pstInputPort[nPortIndex].nMulticastPortId;
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
			result = isMulticastPortTaskIdAndMulticastPortNameEqual(&(g_astMulticastGroups[nLoop].pstOutputPort[nPortIndex]),&stArgPortName, nTaskId, &bIsMatch);
			ERRIFGOTO(result, _EXIT);
			if (bIsMatch == TRUE)
			{
				*pnMulticastPortId = g_astMulticastGroups[nLoop].pstOutputPort[nPortIndex].nMulticastPortId;
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

uem_result UKMulticast_WriteToBuffer(IN int nMulticastGroupId, IN int nMulticastPortId, IN unsigned char *pBuffer, IN int nDataToWrite, OUT int *pnDataWritten)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nMulticastGroupIndex = 0;
	int nMulticastPortIndex = 0;
	int nAPINum = 0;
	int nAPIIndex = 0;
	SMulticastAPI *pstMulticastAPI[g_nMulticastAPINum];
#ifdef ARGUMENT_CHECK
	if(pBuffer == NULL)
	{
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
#endif
	result = UKMulticast_GetMulticastGroupIndexById(nMulticastGroupId, &nMulticastGroupIndex);
	IFVARERRASSIGNGOTO(nMulticastGroupIndex, INVALID_MULTICAST_GROUP_ID, result, ERR_UEM_INVALID_PARAM, _EXIT);

	result = UKMulticast_GetMulticastPortIndexByMulticastGroupIdAndMulticastPortId(nMulticastGroupId, nMulticastPortId, PORT_DIRECTION_OUTPUT, &nMulticastPortIndex);
	ERRIFGOTO(result, _EXIT);

	result = MulticastAPI_GetAPIStructureFromCommunicationType(&(g_astMulticastGroups[nMulticastGroupIndex]), PORT_DIRECTION_OUTPUT, pstMulticastAPI, &nAPINum);
	ERRIFGOTO(result, _EXIT);

	if(nDataToWrite > g_astMulticastGroups[nMulticastGroupIndex].nBufSize)
	{
		nDataToWrite = g_astMulticastGroups[nMulticastGroupIndex].nBufSize;
	}

	for(nAPIIndex = 0 ; nAPIIndex < nAPINum ; nAPIIndex++)
	{
		if (pstMulticastAPI[nAPIIndex]->fnWriteToBuffer == NULL) {
			ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_CONTROL, _EXIT);
		}

		result = pstMulticastAPI[nAPIIndex]->fnWriteToBuffer(&(g_astMulticastGroups[nMulticastGroupIndex].pstOutputPort[nMulticastPortIndex]), pBuffer, nDataToWrite, pnDataWritten);
		ERRIFGOTO(result, _EXIT);
	}

	// to preserve, ERR_UEM_SUSPEND, do not set UEM_NOERROR here
_EXIT:
	return result;
}

uem_result UKMulticast_ReadFromBuffer(IN int nMulticastGroupId, IN int nMulticastPortId, IN OUT unsigned char *pBuffer, IN int nDataToRead, OUT int *pnDataRead)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nMulticastGroupIndex = 0;
	int nMulticastPortIndex = 0;
	int nAPINum = 0;
	int nCommunicationTypeIndex;
	SMulticastAPI *pstMulticastAPI[g_nMulticastAPINum];
#ifdef ARGUMENT_CHECK
	if(pBuffer == NULL)
	{
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
#endif
	result = UKMulticast_GetMulticastGroupIndexById(nMulticastGroupId, &nMulticastGroupIndex);
	IFVARERRASSIGNGOTO(nMulticastGroupIndex, INVALID_CHANNEL_ID, result, ERR_UEM_INVALID_PARAM, _EXIT);

	result = UKMulticast_GetMulticastPortIndexByMulticastGroupIdAndMulticastPortId(nMulticastGroupId, nMulticastPortId, PORT_DIRECTION_INPUT, &nMulticastPortIndex);
	ERRIFGOTO(result, _EXIT);

	result = MulticastAPI_GetAPIStructureFromCommunicationType(&(g_astMulticastGroups[nMulticastGroupIndex]), PORT_DIRECTION_INPUT, pstMulticastAPI, &nAPINum);
	ERRIFGOTO(result, _EXIT);

	result = MulticastAPI_GetMulticastCommunicationTypeIndex(&(g_astMulticastGroups[nMulticastGroupIndex]), PORT_DIRECTION_INPUT, MULTICAST_COMMUNICATION_TYPE_SHARED_MEMORY, &nCommunicationTypeIndex);
	ERRIFGOTO(result, _EXIT);

	if(pstMulticastAPI[nCommunicationTypeIndex]->fnReadFromBuffer == NULL)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_CONTROL, _EXIT);
	}

	if(nDataToRead > g_astMulticastGroups[nMulticastGroupIndex].nBufSize)
	{
		nDataToRead = g_astMulticastGroups[nMulticastGroupIndex].nBufSize;
	}

	result = pstMulticastAPI[nCommunicationTypeIndex]->fnReadFromBuffer(&(g_astMulticastGroups[nMulticastGroupIndex].pstInputPort[nMulticastPortIndex]), pBuffer, nDataToRead, pnDataRead);
	ERRIFGOTO(result, _EXIT);

	// to preserve, ERR_UEM_SUSPEND, do not set UEM_NOERROR here
_EXIT:
	return result;
}

uem_result UKMulticast_GetMulticastGroupSize(IN int nMulticastGroupId, OUT int *pnMulticastGroupSize)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nIndex = 0;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(pnMulticastGroupSize, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	result = UKMulticast_GetMulticastGroupIndexById(nMulticastGroupId, &nIndex);
	IFVARERRASSIGNGOTO(nIndex, INVALID_MULTICAST_GROUP_ID, result, ERR_UEM_INVALID_PARAM, _EXIT);

	*pnMulticastGroupSize = g_astMulticastGroups[nIndex].nBufSize;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKMulticast_Finalize()
{
	uem_result result = ERR_UEM_NOERROR;
	int nLoop = 0;
	int nAPINum = 0;
	int nAPIIndex = 0;
	SMulticastAPI *pstMulticastAPI[g_nMulticastAPINum];

	for(nLoop = 0; nLoop < g_nMulticastGroupNum; nLoop++)
	{
		result = MulticastAPI_GetAPIStructure(&(g_astMulticastGroups[nLoop]), pstMulticastAPI, &nAPINum);
		if (result == ERR_UEM_NOERROR) {
			for (nAPIIndex = 0; nAPIIndex < nAPINum; nAPIIndex++) {
				if (pstMulticastAPI[nAPIIndex]->fnFinalize != NULL) {
					pstMulticastAPI[nAPIIndex]->fnFinalize(&(g_astMulticastGroups[nLoop]));
				}
			}
		}
	}

	for(nLoop = 0; nLoop < g_nMulticastAPINum ; nLoop++)
	{
		if(g_astMulticastAPIList[nLoop]->fnAPIFinalize != NULL)
		{
			g_astMulticastAPIList[nLoop]->fnAPIFinalize();
			// ignore error
		}
	}

	return result;
}
