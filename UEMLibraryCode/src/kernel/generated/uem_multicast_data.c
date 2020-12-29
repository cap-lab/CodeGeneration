/* uem_multicast_data.c made by UEM Translator */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>
#include <uem_multicast_data.h>


#include <UKHostSystem.h>
#include <UKSharedMemoryMulticast.h>
#include <UKMulticast.h>
#include <UCBasic.h>

#include <uem_data.h>

// ##MULTICAST_GROUP_SIZE_DEFINITION_TEMPLATE::START
// ##MULTICAST_GROUP_SIZE_DEFINITION_TEMPLATE::END

// ##MULTICAST_GROUP_BUFFER_DEFINITION_TEMPLATE::START
// ##MULTICAST_GROUP_BUFFER_DEFINITION_TEMPLATE::END

// ##MEMORY_MENAGEMENT_TEMPLATE::START
SGenericMemoryAccess g_stMulticastHostMemory = {
    UKHostSystem_CreateMemory,
    UKHostSystem_CopyToMemory,
    UKHostSystem_CopyInMemory,
    UKHostSystem_CopyFromMemory,
    UKHostSystem_DestroyMemory,
};

// ##MEMORY_MENAGEMENT_TEMPLATE::END

// ##MULTICAST_INPUT_PORT_LIST_TEMPLATE::START
// ##MULTICAST_INPUT_PORT_LIST_TEMPLATE::END

// ##MULTICAST_OUTPUT_PORT_LIST_TEMPLATE::START
// ##MULTICAST_OUTPUT_PORT_LIST_TEMPLATE::END

// ##MULTICAST_GROUP_LIST_TEMPLATE::START
SMulticastGroup g_astMulticastGroups[] = {
};
// ##MULTICAST_GROUP_LIST_TEMPLATE::START

SMulticastAPI g_stSharedMemoryMulticast = {
    UKSharedMemoryMulticast_Initialize, // fnInitialize
    UKSharedMemoryMulticast_ReadFromBuffer, // fnReadFromBuffer
    UKSharedMemoryMulticast_WriteToBuffer, // fnWriteToBuffer
    UKSharedMemoryMulticast_Clear, // fnClear
    UKSharedMemoryMulticast_Finalize, // fnFinalize
    (FnMulticastAPIInitialize) NULL,
    (FnMulticastAPIFinalize) NULL,
};


SMulticastAPI *g_astMulticastAPIList[] = {
    &g_stSharedMemoryMulticast,
};


#ifdef __cplusplus
extern "C"
{
#endif

uem_result MulticastAPI_SetSocketAPIs()
{
    uem_result result = ERR_UEM_UNKNOWN;


    result = ERR_UEM_NOERROR;
    return result;
}
uem_result MulticastAPI_GetAPIStructure(IN SMulticastGroup *pstMulticastGroup, OUT SMulticastAPI **pstMulticastAPI, OUT int *pnAPINum)
{
    uem_result result = ERR_UEM_UNKNOWN;
    int arrCommunicationTypeList[MULTICAST_COMMUNICATION_TYPE_END];
    int nAPINum = 0;
    int nLoop = 0;
    int nInerLoop = 0;
    
    UC_memset(arrCommunicationTypeList, 0, MULTICAST_COMMUNICATION_TYPE_END);

    for(nLoop = 0 ; nLoop < pstMulticastGroup->nInputCommunicationTypeNum ; nLoop++)
    {
        arrCommunicationTypeList[pstMulticastGroup->pstInputCommunicationInfo[nLoop].eCommunicationType] = 1;
    }
    for(nLoop = 0 ; nLoop < pstMulticastGroup->nOutputCommunicationTypeNum ; nLoop++)
    {
        arrCommunicationTypeList[pstMulticastGroup->pstOutputCommunicationInfo[nLoop].eCommunicationType] = 1;
    }
    
    for(nLoop = 0 ; nLoop < MULTICAST_COMMUNICATION_TYPE_END ; nLoop++)
    {
    	if(arrCommunicationTypeList[nLoop]==0)
    	{
    		continue;
    	}
        switch(nLoop)
        {
            case MULTICAST_COMMUNICATION_TYPE_SHARED_MEMORY:
                pstMulticastAPI[nLoop] = &g_stSharedMemoryMulticast;
                nAPINum++;
                break;
            case MULTICAST_COMMUNICATION_TYPE_UDP:
                ERRIFGOTO(result, _EXIT);
                break;
            default:
                ERRIFGOTO(result, _EXIT);
        }
    }
    *pnAPINum = nAPINum;

    result = ERR_UEM_NOERROR;
_EXIT:
    return result;
}

uem_result MulticastAPI_GetAPIStructureFromCommunicationType(IN SMulticastGroup *pstMulticastGroup, IN EPortDirection eDirection, OUT SMulticastAPI **pstMulticastAPI, OUT int *pnAPINum)
{
    uem_result result = ERR_UEM_UNKNOWN;
    int nAPINum = 0;
    SMulticastCommunicationInfo *pstCommunicationInfo;
    int nCommunicationTypeNum = 0;
    
    if(eDirection == PORT_DIRECTION_INPUT)
    {
        pstCommunicationInfo = pstMulticastGroup->pstInputCommunicationInfo;
    	nCommunicationTypeNum = pstMulticastGroup->nInputCommunicationTypeNum;
    }
    else if(eDirection == PORT_DIRECTION_OUTPUT)
    {
    	pstCommunicationInfo = pstMulticastGroup->pstOutputCommunicationInfo;
    	nCommunicationTypeNum = pstMulticastGroup->nOutputCommunicationTypeNum;
    }

    for(nAPINum = 0 ; nAPINum < nCommunicationTypeNum ;nAPINum++)
    {
        switch(pstCommunicationInfo[nAPINum].eCommunicationType)
        {
            case MULTICAST_COMMUNICATION_TYPE_SHARED_MEMORY:
                pstMulticastAPI[nAPINum] = &g_stSharedMemoryMulticast;
                break;
            case MULTICAST_COMMUNICATION_TYPE_UDP:
                ERRIFGOTO(result, _EXIT);
                break;
            default:
                ERRIFGOTO(result, _EXIT);
        }
    }
    
    *pnAPINum = nAPINum;
    
    result = ERR_UEM_NOERROR;
_EXIT:
    return result;
}

uem_result MulticastAPI_GetMulticastCommunicationTypeIndex(IN SMulticastGroup *pstMulticastGroup, IN EPortDirection eDirection, IN EMulticastCommunicationType eMulticastCommunicationType, OUT int *pnCommunicationTypeIndex)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop;
	uem_bool bFound = FALSE;
	SMulticastCommunicationInfo *pstCommunicationInfo;
    int nCommunicationTypeNum = 0;
    	
	if(eDirection == PORT_DIRECTION_INPUT)
    {
    	pstCommunicationInfo = pstMulticastGroup->pstInputCommunicationInfo;
    	nCommunicationTypeNum = pstMulticastGroup->nInputCommunicationTypeNum;
    }
    else
    {
    	pstCommunicationInfo = pstMulticastGroup->pstOutputCommunicationInfo;
    	nCommunicationTypeNum = pstMulticastGroup->nOutputCommunicationTypeNum;
    }

	for(nLoop = 0 ; nLoop < nCommunicationTypeNum ; nLoop++)
	{
		if(eMulticastCommunicationType == pstCommunicationInfo[nLoop].eCommunicationType)
		{
			*pnCommunicationTypeIndex = nLoop;
			bFound = TRUE;
			break;
		}
	}
	
	IFVARERRASSIGNGOTO(bFound, FALSE, result, ERR_UEM_NOT_FOUND, _EXIT);
	
	result = ERR_UEM_NOERROR;
_EXIT:
    return result;
}

#ifdef __cplusplus
}
#endif

int g_nMulticastGroupNum = ARRAYLEN(g_astMulticastGroups);
int g_nMulticastAPINum = ARRAYLEN(g_astMulticastAPIList);
