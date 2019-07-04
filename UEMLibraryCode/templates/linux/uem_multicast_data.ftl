/* uem_multicast_data.c made by UEM Translator */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>
#include <uem_multicast_data.h>
<#if communication_used == true>
#include <UCDynamicSocket.h>
    <#if used_communication_list?seq_contains("udp")>
#include <UCUDPSocket.h>
#include <UKUDPSocketMulticast.h>
    </#if>
</#if>

<#if gpu_used == true>
#include <UKGPUSystem.h>
</#if>

#include <UKHostSystem.h>
#include <UKSharedMemoryMulticast.h>
#include <UKMulticast.h>

#include <uem_data.h>

// ##MULTICAST_GROUP_SIZE_DEFINITION_TEMPLATE::START
<#list multicast_group_list as multicast>
    <#if multicast.inputPortNum gt 0>
#define MULTICAST_${multicast.groupName}_SIZE (${multicast.size?c})
    </#if>
</#list>
// ##MULTICAST_GROUP_SIZE_DEFINITION_TEMPLATE::END

// ##MULTICAST_GROUP_BUFFER_DEFINITION_TEMPLATE::START
<#list multicast_group_list as multicast>
    <#if multicast.inputPortNum gt 0>
char s_pMulticastGroup_${multicast.groupName}_buffer[MULTICAST_${multicast.groupName}_SIZE];
    </#if>
</#list>
// ##MULTICAST_GROUP_BUFFER_DEFINITION_TEMPLATE::END

// ##MEMORY_MENAGEMENT_TEMPLATE::START
SGenericMemoryAccess g_stMulticastHostMemory = {
    UKHostSystem_CreateMemory,
    UKHostSystem_CopyToMemory,
    UKHostSystem_CopyInMemory,
    UKHostSystem_CopyFromMemory,
    UKHostSystem_DestroyMemory,
};

<#if gpu_used == true>
SGenericMemoryAccess g_stMulticastDeviceMemory = {
    UKHostSystem_CreateMemory,
    UKGPUSystem_CopyDeviceToHostMemory,
    UKHostSystem_CopyInMemory,
    UKGPUSystem_CopyHostToDeviceMemory,
    UKHostSystem_DestroyMemory,
};
</#if>
// ##MEMORY_MENAGEMENT_TEMPLATE::END

// ##MULTICAST_INPUT_PORT_LIST_TEMPLATE::START
<#list multicast_group_list as multicast>
    <#if multicast.inputPortNum gt 0>
SSharedMemoryMulticast g_stSharedMemoryMulticast_${multicast.groupName} = {
    s_pMulticastGroup_${multicast.groupName}_buffer, // pBuffer
    s_pMulticastGroup_${multicast.groupName}_buffer, // pDataStart
    s_pMulticastGroup_${multicast.groupName}_buffer, // pDataEnd
    0, // nDataLen
    0, // nReadReferenceCount
    0, // nWriteReferenceCount
    (HThreadMutex) NULL, // Mutex
};
    </#if>
    
    <#list multicast.inputCommunicationTypeList as communicationType>
        <#switch communicationType>
            <#case "MULTICAST_COMMUNICATION_TYPE_UDP">
SUDPMulticast g_stUDPMulticastSocket_${multicast.groupName} = {
    (SUDPSocket *) NULL, // pstSocket
    (HThread) NULL,  // hManagementThread
    FALSE, // bExit
    NULL // pstMulticastManager
};
            <#break>
        </#switch>
    </#list>

void *g_astMulticastRecvGateList_${multicast.groupName}[] = {
    <#list multicast.inputCommunicationTypeList as communicationType>
        <#switch communicationType>
            <#case "MULTICAST_COMMUNICATION_TYPE_SHARED_MEMORY">
    NULL,
            <#break>
            <#case "MULTICAST_COMMUNICATION_TYPE_UDP">
    (SUDPMulticast *) &g_stUDPMulticastSocket_${multicast.groupName},
            <#break>
            <#default>
        </#switch>
    </#list>
};

	<#list multicast.inputCommunicationTypeList as communicationType>
		<#switch communicationType>
			<#case "MULTICAST_COMMUNICATION_TYPE_UDP">
SUDPInfo g_astUDPInputCommunicationInfo_${multicast.groupName} = {
    "${udp_server_list[0].IP}", // pszIP
	${udp_server_list[0].port?c} // nPort
};
			<#break>
			<#default>
		</#switch>
	</#list>

SMulticastCommunicationInfo g_astMulticastInputCommunicationInfo_${multicast.groupName}[] = {
    <#list multicast.inputCommunicationTypeList as communicationType>
    {
        ${communicationType},
		<#switch communicationType>
			<#case "MULTICAST_COMMUNICATION_TYPE_UDP">
		&g_astUDPInputCommunicationInfo_${multicast.groupName},
			<#break>
            <#default>
        NULL,
        </#switch>
    },
    </#list>
};

SMulticastPort g_astMulticastInputPortList_${multicast.groupName}[] = {
    <#list multicast.inputPortList as inputPort>
    {
        ${inputPort.taskId}, // nTaskId
        ${inputPort.portId}, // nPortId
        "${inputPort.portName}", // pszPortName
        PORT_DIRECTION_INPUT, // eDirection
        <#switch inputPort.inMemoryAccessType>
        	<#case "CPU_ONLY">
        &g_stMulticastHostMemory, // pstMemoryAccessAPI
        	<#break>
        	<#case "GPU_ONLY">
        &g_stMulticastDeviceMemory, // pstMemoryAccessAPI
        	<#break>
        	<#default>
        </#switch>
        (SMulticastGroup *) NULL, // pMulticastGroup		
        (void **) NULL, // pMulticastSendGateList
    },
    </#list>
};
</#list>
// ##MULTICAST_INPUT_PORT_LIST_TEMPLATE::END

// ##MULTICAST_OUTPUT_PORT_LIST_TEMPLATE::START
<#list multicast_group_list as multicast>
    <#list multicast.outputPortList as outputPort>
        <#list multicast.outputCommunicationTypeList as communicationType>
            <#switch communicationType>
                <#case "MULTICAST_COMMUNICATION_TYPE_UDP">
SUDPMulticast g_stUDPMulticastSocket_${multicast.groupName}_${outputPort.portName} = {
    (SUDPSocket *) NULL, // pstSocket
    (HThread) NULL,  // hManagementThread
    FALSE, // bExit
    NULL // pstMulticastManager
};
                <#break>
            </#switch>
        </#list>

void *g_astMulticastSendGateList_${multicast.groupName}_${outputPort.portName}[] = {
        <#list multicast.outputCommunicationTypeList as communicationType>
            <#switch communicationType>
        		<#case "MULTICAST_COMMUNICATION_TYPE_SHARED_MEMORY">
    NULL,
            	<#break>
                <#case "MULTICAST_COMMUNICATION_TYPE_UDP">
    (SUDPMulticast *) &g_stUDPMulticastSocket_${multicast.groupName}_${outputPort.portName},
                <#break>
                <#default>
            </#switch>
        </#list>
};
    </#list>
    
	<#list multicast.outputCommunicationTypeList as communicationType>
		<#switch communicationType>
			<#case "MULTICAST_COMMUNICATION_TYPE_UDP">
SUDPInfo g_astUDPOutputCommunicationInfo_${multicast.groupName} = {
	"${udp_client_list[0].IP}", // pszIP
	${udp_client_list[0].port?c} // nPort
};
			<#break>
			<#default>
		</#switch>
	</#list>

SMulticastCommunicationInfo g_astMulticastOutputCommunicationInfo_${multicast.groupName}[] = {
    <#list multicast.outputCommunicationTypeList as communicationType>
    {
        ${communicationType},
		<#switch communicationType>
			<#case "MULTICAST_COMMUNICATION_TYPE_UDP">
		&g_astUDPOutputCommunicationInfo_${multicast.groupName},
			<#break>
            <#default>
        NULL,
        </#switch>
    },
    </#list>
};

SMulticastPort g_astMulticastOutputPortList_${multicast.groupName}[] = {
    <#list multicast.outputPortList as outputPort>
    {
        ${outputPort.taskId}, // nTaskId
        ${outputPort.portId}, // nMulticastPortId
        "${outputPort.portName}", // pszPortName
        PORT_DIRECTION_OUTPUT, // eDirection
        <#switch outputPort.inMemoryAccessType>
        	<#case "CPU_ONLY">
        &g_stMulticastHostMemory, // pstMemoryAccessAPI
        	<#break>
        	<#case "GPU_ONLY">
        &g_stMulticastDeviceMemory, // pstMemoryAccessAPI
        	<#break>
        	<#default>
        </#switch>
        (SMulticastGroup *) NULL, // pMulticastGroup
        g_astMulticastSendGateList_${multicast.groupName}_${outputPort.portName}, // pMulticastSendGateList
    },
    </#list>
};
</#list>
// ##MULTICAST_OUTPUT_PORT_LIST_TEMPLATE::END

// ##MULTICAST_GROUP_LIST_TEMPLATE::START
SMulticastGroup g_astMulticastGroups[] = {
<#list multicast_group_list as multicast>
    {
        ${multicast.multicastGroupId}, // Multicast group ID
        "${multicast.groupName}", // pszGroupName
        MULTICAST_${multicast.groupName}_SIZE, // Multicast group buffer size
        g_astMulticastInputPortList_${multicast.groupName}, // pstInputPort
        ${multicast.inputPortNum}, // nInputPortNum
        g_astMulticastInputCommunicationInfo_${multicast.groupName}, // pstInputCommunicationInfo
        ARRAYLEN(g_astMulticastInputCommunicationInfo_${multicast.groupName}), // nInputCommunicationTypeNum
        g_astMulticastOutputPortList_${multicast.groupName}, // pstOutputPort
        ${multicast.outputPortNum}, // nOutputPortNum
        g_astMulticastOutputCommunicationInfo_${multicast.groupName}, // pstOutputCommunicationInfo
        ARRAYLEN(g_astMulticastOutputCommunicationInfo_${multicast.groupName}), // nOutputCommunicationTypeNum
        <#if multicast.inputPortNum gt 0>
        &(g_stSharedMemoryMulticast_${multicast.groupName}), // pMulticastStruct
        <#else>
        NULL, // pMulticastStruct
        </#if>
        g_astMulticastRecvGateList_${multicast.groupName},
    },
</#list>
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

<#if used_communication_list?seq_contains("udp")>
SMulticastAPI g_stUDPSocketMulticast = {
    UKUDPSocketMulticast_Initialize, // fnInitialize
    (FnMulticastReadFromBuffer) NULL, // fnReadFromBuffer
    UKUDPSocketMulticast_WriteToBuffer, // fnWriteToBuffer
    (FnMulticastClear) NULL, // fnClear
    UKUDPSocketMulticast_Finalize, // fnFinalize
    (FnMulticastAPIInitialize) NULL,
    (FnMulticastAPIFinalize) NULL,
};
</#if>

SMulticastAPI *g_astMulticastAPIList[] = {
    &g_stSharedMemoryMulticast,
    <#if used_communication_list?seq_contains("udp")>
    &g_stUDPSocketMulticast,
    </#if>
};

<#if used_communication_list?seq_contains("udp")>
SSocketAPI stUDPAPI = {
    UCUDPSocket_Bind,
    (FnSocketAccept) NULL,
    (FnSocketConnect) NULL,
    UCUDPSocket_Create,
    (FnSocketDestroy) NULL,
};		
</#if>

#ifdef __cplusplus
extern "C"
{
#endif

<#assign printed=false />
uem_result MulticastAPI_SetSocketAPIs()
{
    uem_result result = ERR_UEM_UNKNOWN;

<#if used_communication_list?seq_contains("udp")>
    result = UCDynamicSocket_SetAPIList(SOCKET_TYPE_UDP, &stUDPAPI);
    ERRIFGOTO(result, _EXIT);
    <#assign printed=true />
</#if>

    result = ERR_UEM_NOERROR;
<#if (printed == true)>
_EXIT:
</#if>
    return result;
}
uem_result MulticastAPI_GetAPIStructure(IN SMulticastGroup *pstMulticastGroup, OUT SMulticastAPI **pstMulticastAPI, OUT int *pnAPINum)
{
    EMulticastCommunicationType aeCommunicationTypeList[g_nMulticastAPINum];
    int nAPINum = 0;
    int nLoop = 0;
    int nInerLoop = 0;

    for(nAPINum = 0, nLoop = 0 ; nLoop < pstMulticastGroup->nInputCommunicationTypeNum ; nAPINum++, nLoop = 0)
    {
        aeCommunicationTypeList[nAPINum] = pstMulticastGroup->nInputCommunicationTypeNum[nLoop].eCommunicationType;
    }
    for(nLoop = 0 ; nLoop < pstMulticastGroup->nOutputCommunicationTypeNum ; nLoop++)
    {
        for(nInerLoop = 0 ; nInerLoop < nAPINum ; nInerLoop++)
        {
            if(aeCommunicationTypeList[nInerLoop] == pstMulticastGroup->nOutputCommunicationTypeNum[nLoop].eCommunicationType)
            {
                break;
            }
        }
        if(nInerLoop != nAPINum)
        {
            aeCommunicationTypeList[nAPINum] = pstMulticastGroup->nOutputCommunicationTypeNum[nLoop].eCommunicationType;
            nAPINum++;
        }
    }
    for(nLoop = 0 ; nLoop < nAPINum ; nLoop++)
    {
        switch(aeCommunicationTypeList[nLoop])
        {
            case MULTICAST_COMMUNICATION_TYPE_SHARED_MEMORY:
                pstMulticastAPI[nAPINum] = &g_stSharedMemoryMulticast;
                break;
            case MULTICAST_COMMUNICATION_TYPE_UDP:
<#if used_communication_list?seq_contains("udp")>
                pstMulticastAPI[nAPINum] = &g_stUDPSocketMulticast;
<#else>
                ERRIFGOTO(result, _EXIT);
</#if>
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
<#if used_communication_list?seq_contains("udp")>
                pstMulticastAPI[nAPINum] = &g_stUDPSocketMulticast;
<#else>
                ERRIFGOTO(result, _EXIT);
</#if>
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
	if(bFound == FALSE)
	{
		ERRIFGOTO(result, _EXIT);
	}
	
	result = ERR_UEM_NOERROR;
_EXIT:
    return result;
}

#ifdef __cplusplus
}
#endif

int g_nMulticastGroupNum = ARRAYLEN(g_astMulticastGroups);
int g_nMulticastAPINum = ARRAYLEN(g_astMulticastAPIList);
