/* uem_multicast_data.c made by UEM Translator */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

<#if communication_used == true>
#include <UCDynamicSocket.h>
    <#if used_communication_list?seq_contains("udp")>
#include <UCUDPSocket.h>
#include <UKUDPServerManager.h>
#include <UKUDPSocketMulticast.h>

#include <uem_udp_data.h>
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
// ##CHANNEL_SIZE_DEFINITION_TEMPLATE::END

// ##MEMORY_MENAGEMENT_TEMPLATE::START
SGenericMemoryAccess g_stHostMemory = {
    UKHostSystem_CreateMemory,
    UKHostSystem_CopyToMemory,
    UKHostSystem_CopyInMemory,
    UKHostSystem_CopyFromMemory,
    UKHostSystem_DestroyMemory,
};

<#if gpu_used == true>
SGenericMemoryAccess g_stDeviceMemory = {
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
    &g_stHostMemory, // pstMemoryAccessAPI
};
    </#if>
    <#list multicast.inputCommunicationTypeList as communicationType>
        <#switch communicationType>
            <#case "MULTICAST_COMMUNICATION_TYPE_UDP">
SUDPSocket g_stUDPSocket_${multicast.groupName} = {
    (HThread) NULL, // hThread
    NULL, // pBuffer
    0, // nBufLen
    (HThreadMutex) NULL, // hMutex
    FALSE, // bExit
    NULL, // pSocketManager
    (SGenericMemoryAccess*) NULL, // pstReaderAccess
    (HSocket*) NULL, // hSocket
};
            <#break>
        </#switch>
    </#list>

void *g_astMulticastRecvGateList_${multicast.groupName}[]{
    <#list multicast.inputCommunicationTypeList as communicationType>
        <#switch communicationType>
            <#case "MULTICAST_COMMUNICATION_TYPE_UDP">
    g_stUDPSocket_${multicast.groupName},
            <#break>
            <#default>
    NULL,
        </#switch>
    </#list>
};

	<#list multicast.inputCommunicationTypeList as communicationType>
		<#switch communicationType>
			<#case "MULTICAST_COMMUNICATION_TYPE_UDP">
SUDPInfo g_astUDPInputCommunicationInfo_${multicast.groupName} = {
	${udp_server_list[0].port}
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
        ${inputPort.portName}, // pszPortName
        PORT_DIRECTION_OUTPUT, // eDirection
        <#switch inputPort.inMemoryAccessType>
        	<#case "CPU_ONLY">
        &g_stHostMemory, // pstMemoryAccessAPI
        	<#break>
        	<#case "GPU_ONLY">
        &g_stDeviceMemory, // pstMemoryAccessAPI
        	<#break>
        	<#default>
        </#switch>
        NULL, // pMulticastGroup		
        NULL, // pMulticastSendGateList
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
SUDPSocket g_astUDPSocket_${multicast.groupName}_&{multicast.portName} = {
    (HThread) NULL, // hThread
    NULL, // pBuffer
    0, // nBufLen
    (HThreadMutex) NULL, // hMutex
    FALSE, // bExit
    NULL, // pSocketManager
    (SGenericMemoryAccess*) NULL, // pstReaderAccess
    (HSocket*) NULL, // hSocket
};
                <#break>
            </#switch>
        </#list>

void *g_astMulticastSendGateList_${multicast.groupName}_${outputPort.portName}[] = {
        <#list multicast.outputCommunicationTypeList as communicationType>
            <#switch communicationType>
                <#case "MULTICAST_COMMUNICATION_TYPE_UDP">
    g_astUDPSocket_${multicast.groupName}_${outputPort.portName},
                <#break>
                <#default>
    NULL,
            </#switch>
        </#list>
};
    </#list>
    
	<#list multicast.outputCommunicationTypeList as communicationType>
		<#switch communicationType>
			<#case "MULTICAST_COMMUNICATION_TYPE_UDP">
SUDPInfo g_astUDPOutputCommunicationInfo_${multicast.groupName} = {
	${udp_client_list[0].port}
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
        ${outputPort.portName}, // pszPortName
        PORT_DIRECTION_INPUT, // eDirection
        <#switch outputPort.inMemoryAccessType>
        	<#case "CPU_ONLY">
        &g_stHostMemory, // pstMemoryAccessAPI
        	<#break>
        	<#case "GPU_ONLY">
        &g_stDeviceMemory, // pstMemoryAccessAPI
        	<#break>
        	<#default>
        </#switch>
        NULL, // pMulticastGroup
        &g_astMulticastSendGateList_${multicast.groupName}_${outputPort.portName}, // pMulticastSendGateList
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
        MULTICAST_${multicast.groupName}_SIZE, // Multicast group buffer size
        &(g_astMulticastInputPortList_${multicast.groupName}), // pstInputPort
        ${multicast.inputPortNum}, // nInputPortNum
        &(g_astMulticastInputCommunicationInfo_${multicast.groupName}), // pstInputCommunicationInfo
        ARRAYLEN(g_astMulticastInputCommunicationInfo_${multicast.groupName}), // nInputCommunicationTypeNum
        &(g_astMulticastOutputPortList_${multicast.groupName}), // pstOutputPort
        ${multicast.outputPortNum}, // nOutputPortNum
        &(g_astMulticastOutputCommunicationInfo_${multicast.groupName}), // pstOutputCommunicationInfo
        ARRAYLEN(g_astMulticastOutputCommunicationInfo_${multicast.groupName}), // nOutputCommunicationTypeNum
        <#if multicast.inputPortNum gt 0>
        g_stSharedMemoryMulticast_${multicast.groupName}, // pMulticastStruct
        <#else>
        NULL, // pMulticastStruct
        </#if>
        &g_astMulticastRecvGateList_${multicast.groupName},
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
    (fnReadFromBuffer) NULL, // fnReadFromBuffer
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
    (FnSocketCreate) NULL,
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

uem_result MulticastAPI_GetAPIStructureFromCommunicationType(IN SMulticastGroup *pstMulticastGroup, IN EPortDirection eDirection, OUT SMulticastAPI **pstMulticastAPI, OUT int *pnAPINum)
{
    uem_result result = ERR_UEM_UNKNOWN;
    int nAPINum = 0;
    if(eDirection == PORT_DIRECTION_INPUT)
    {
        for(nAPINum = 0 ; nAPINum < pstMulticastGroup->nInputCommunicationTypeNum ;nAPINum++)
        {
            switch(pstMulticastGroup->pstInputCommunicationInfo->eCommunicationType)
            {
                case COMMUNICATION_TYPE_SHARED_MEMORY:
                    pstMulticastAPI[nAPINum] = &g_stSharedMemoryMulticast;
                    break;
                case COMMUNICATION_TYPE_UDP:
                    pstMulticastAPI[nAPINum] = &g_stUDPSocketMulticast;
                    break;
                default:
                    ERRIFGOTO(result, _EXIT);
            }
        }
    }
    else
    {
        for(nAPINum = 0 ; nAPINum < pstMulticastGroup->nOutputCommunicationTypeNum ;nAPINum++)
        {
            switch(pstMulticastGroup->pstOutputCommunicationInfo->eCommunicationType)
            {
                case COMMUNICATION_TYPE_SHARED_MEMORY:
                    pstMulticastAPI[nAPINum] = &g_stSharedMemoryMulticast;
                    break;
                case COMMUNICATION_TYPE_UDP:
                    pstMulticastAPI[nAPINum] = &g_stUDPSocketMulticast;
                    break;
                default:
                    ERRIFGOTO(result, _EXIT);
            }
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
	
	if(eDirection == PORT_DIRECTION_INPUT)
	{
		for(nLoop = 0 ; nLoop < pstMulticastGroup->nInputCommunicationTypeNum ; nLoop++)
		{
			if(eMulticastCommunicationType == pstMulticastGroup->pstInputCommunicationInfo[nLoop]->eCommunicationType)
			{
				*pnCommunicationTypeIndex = nLoop;
				break;
			}
		}
	}
	else
	{
		for(nLoop = 0 ; nLoop < pstMulticastGroup->nOutputCommunicationTypeNum ; nLoop++)
		{
			if(eMulticastCommunicationType == pstMulticastGroup->pstOutputCommunicationInfo[nLoop]->eCommunicationType)
			{
				*pnCommunicationTypeIndex = nLoop;
				break;
			}
		}
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
