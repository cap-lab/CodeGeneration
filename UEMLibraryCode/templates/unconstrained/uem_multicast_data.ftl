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
#include <UCBasic.h>

#include <uem_data.h>

// ##MULTICAST_GROUP_SIZE_DEFINITION_TEMPLATE::START
<#list multicast_group_list as multicast>
#define MULTICAST_${multicast.groupName}_SIZE (${multicast.bufferSize?c})
</#list>
// ##MULTICAST_GROUP_SIZE_DEFINITION_TEMPLATE::END

// ##MULTICAST_API_DEFINITION_TEMPLATE::START
SMulticastAPI g_stSharedMemoryMulticast = {
	(FnMulticastAPIInitialize) NULL, // fnAPIInitialize
	(FnMulticastAPIFinalize) NULL, // fnAPIFinalize
	UKSharedMemoryMulticastGroup_Initialize, // fnGroupInitialize
	UKSharedMemoryMulticastGroup_Finalize, // fnGroupFinalize
	(FnMulticastPortInitialize) NULL, // fnPortInitialize
	(FnMulticastPortFinalize) NULL, // fnPortFinalize
	UKSharedMemoryMulticast_ReadFromBuffer, // fnReadFromBuffer
	UKSharedMemoryMulticast_WriteToBuffer, // fnWriteToBuffer
};

<#if used_communication_list?seq_contains("udp")>
SMulticastAPI g_stUDPSocketMulticast = {
	UKUDPSocketMulticastAPI_Initialize, // fnAPIInitialize
	UKUDPSocketMulticastAPI_Finalize, // fnAPIFinalize
	(FnMulticastGroupInitialize) NULL, // fnInitialize
	(FnMulticastGroupFinalize) NULL, // fnFinalize
	UKUDPSocketMulticastPort_Initialize, // fnInitialize
	UKUDPSocketMulticastPort_Finalize, // fnFinalize
	(FnMulticastReadFromBuffer) NULL, // fnReadFromBuffer
	UKUDPSocketMulticast_WriteToBuffer, // fnWriteToBuffer
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
// ##MULTICAST_API_DEFINITION_TEMPLATE::END

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

// ##MULTICAST_GROUP_BUFFER_DEFINITION_TEMPLATE::START
<#list multicast_group_list as multicast>
	<#if multicast.inputPortNum gt 0>
char s_pMulticastGroup_${multicast.groupName}_buffer[MULTICAST_${multicast.groupName}_SIZE];
	</#if>
</#list>
// ##MULTICAST_GROUP_BUFFER_DEFINITION_TEMPLATE::END

// ##MULTICAST_CONNECTION_TEMPLATE::START
<#if used_communication_list?seq_contains("udp")>
<#list udp_list as udp>
int g_anMulticastUDPReceivers_${udp.getUDPId()}[] = {
	<#list udp.getMulticastReceivers() as receiver>
	${receiver?c},
	</#list>
};
int g_anMulticastUDPSenders_${udp.getUDPId()}[] = {
	<#list udp.getMulticastSenders() as sender>
	${sender?c},
	</#list>
};
</#list>

SUDPMulticast g_astMulticastUDPList[] = {
	<#list udp_list as udp>
	{
		{
			"${udp.IP}",
			${udp.port?c},
		},
		g_anMulticastUDPReceivers_${udp.getUDPId()},
		${udp.getMulticastReceivers()?size},
		g_anMulticastUDPSenders_${udp.getUDPId()},
		${udp.getMulticastSenders()?size},
		(SUDPMulticastReceiver *) NULL,
	},
	</#list>
};
</#if>
// ##MULTICAST_CONNECTION_TEMPLATE::END

// ##MULTICAST_SHARED_MEMORY_TEMPLATE::START
<#list multicast_group_list as multicast>
	<#if multicast.getInputPortNum() gt 0>
SSharedMemoryMulticast g_stSharedMemoryMulticast_${multicast.groupName} = {
	s_pMulticastGroup_${multicast.groupName}_buffer, // pData
	0, // nDataLen
	(HThreadMutex) NULL, // Mutex
};
	</#if>
</#list>
// ##MULTICAST_SHARED_MEMORY_TEMPLATE::END

// ##MULTICAST_INPUT_PORT_LIST_TEMPLATE::START
<#list multicast_group_list as multicast>
SMulticastCommunication g_astMulticastInputCommunicationList_${multicast.groupName}[] = {
	<#list multicast.getInputCommunicationType() as inputCommunicationType>
	{
		<#switch inputCommunicationType>
			<#case "SHARED_MEMORY">
		SHARED_MEMORY,
		&g_stSharedMemoryMulticast,
		&g_stSharedMemoryMulticast_${multicast.groupName},
				<#break>
			<#case "UDP">
		UDP,
		&g_stUDPSocketMulticast,
		(void *) NULL,
				<#break>
			<#default>  
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
			<#case "CPU_GPU">
		&g_stMulticastDeviceMemory, // pstMemoryAccessAPI
				<#break>
			<#default>
		</#switch>
		(SMulticastGroup *) NULL, // pMulticastGroup
		g_astMulticastInputCommunicationList_${multicast.groupName}, // astCommunicationList
		${multicast.getInputCommunicationType()?size}, // nCommunicationTypeNum
	},
	</#list>
};
</#list>
// ##MULTICAST_INPUT_PORT_LIST_TEMPLATE::END

// ##MULTICAST_OUTPUT_PORT_LIST_TEMPLATE::START
<#list multicast_group_list as multicast>
SMulticastCommunication g_astMulticastOutputCommunicationList_${multicast.groupName}[] = {
	 <#list multicast.getOutputCommunicationType() as outputCommunicationType>
	{
		<#switch outputCommunicationType>
			<#case "SHARED_MEMORY">
		SHARED_MEMORY,
		&g_stSharedMemoryMulticast,
		&g_stSharedMemoryMulticast_${multicast.groupName},
				<#break>
			<#case "UDP">
		UDP,
		&g_stUDPSocketMulticast,
		(void *) NULL,
				<#break>
			<#default>  
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
			<#case "GPU_CPU">
		&g_stMulticastDeviceMemory, // pstMemoryAccessAPI
				<#break>
			<#default>
		</#switch>
		(SMulticastGroup *) NULL, // pMulticastGroup
		g_astMulticastOutputCommunicationList_${multicast.groupName}, // astCommunicationList
		${multicast.getOutputCommunicationType()?size}, // nCommunicationTypeNum
	},
	</#list>
};
</#list>
// ##MULTICAST_OUTPUT_PORT_LIST_TEMPLATE::END

// ##MULTICAST_GROUP_LIST_TEMPLATE::START
<#list multicast_group_list as multicast>
SMulticastCommunication g_astMulticastCommunicationList_${multicast.groupName}[] = {
	 <#list multicast.getCommunicationTypeList() as communicationType>
	{
		<#switch communicationType>
			<#case "SHARED_MEMORY">
		SHARED_MEMORY,
		&g_stSharedMemoryMulticast,
		&g_stSharedMemoryMulticast_${multicast.groupName},
				<#break>
			<#case "UDP">
		UDP,
		&g_stUDPSocketMulticast,
		(void *) NULL,
				<#break>
			<#default>  
		</#switch>
	},
	</#list>
};
</#list>
SMulticastGroup g_astMulticastGroups[] = {
<#list multicast_group_list as multicast>
	{
		${multicast.multicastGroupId}, // Multicast group ID
		"${multicast.groupName}", // pszGroupName
		MULTICAST_${multicast.groupName}_SIZE, // Multicast group buffer size
		g_astMulticastInputPortList_${multicast.groupName}, // pstInputPort
		${multicast.inputPortList?size}, // nInputPortNum
		g_astMulticastOutputPortList_${multicast.groupName}, // pstOutputPort
		${multicast.outputPortList?size}, // nOutputPortNum
		g_astMulticastCommunicationList_${multicast.groupName}, // astCommunicationList
		${multicast.getCommunicationTypeList()?size}, // nCommunicationTypeNum
	},
</#list>
<#if platform == "windows" && (multicast_group_list?size == 0)>
	0, 
</#if>
};
// ##MULTICAST_GROUP_LIST_TEMPLATE::START

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

#ifdef __cplusplus
}
#endif

<#if used_communication_list?seq_contains("udp")>
int g_nMulticastUDPNum = ARRAYLEN(g_astMulticastUDPList);
</#if>
<#if (multicast_group_list?size > 0) >
int g_nMulticastGroupNum = ARRAYLEN(g_astMulticastGroups);
int g_nMulticastAPINum = ARRAYLEN(g_astMulticastAPIList);
<#else>
int g_nMulticastGroupNum = 0;
int g_nMulticastAPINum = 0;
</#if>
