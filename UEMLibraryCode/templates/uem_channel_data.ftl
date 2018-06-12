/* uem_channel_data.c made by UEM Translator */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

<#if communication_used == true>
#include <UCDynamicSocket.h>
</#if>

#include <UKHostMemorySystem.h>

#include <uem_data.h>

<#if communication_used == true>
#include <UKUEMProtocol.h>

#include <uem_tcp_data.h>
</#if>

<#if gpu_used == true>
#include <UKGPUMemorySystem.h>
</#if>


// ##CHANNEL_SIZE_DEFINITION_TEMPLATE::START
<#list channel_list as channel>
#define CHANNEL_${channel.index}_SIZE (${channel.size?c})
</#list>
// ##CHANNEL_SIZE_DEFINITION_TEMPLATE::END


// ##CHANNEL_BUFFER_DEFINITION_TEMPLATE::START
<#list channel_list as channel>
char s_pChannel_${channel.index}_buffer[CHANNEL_${channel.index}_SIZE];
</#list>
// ##CHANNEL_BUFFER_DEFINITION_TEMPLATE::END


// ##CHUNK_DEFINITION_TEMPLATE::START
<#list channel_list as channel>
SChunk g_astChunk_channel_${channel.index}_out[] = {
<#list 0..(channel.outputPort.maximumChunkNum-1) as chunk_id>
	{
		s_pChannel_${channel.index}_buffer, // Chunk start pointer
		s_pChannel_${channel.index}_buffer, // Data start pointer
		s_pChannel_${channel.index}_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
</#list>
};

SChunk g_astChunk_channel_${channel.index}_in[] = {
<#list 0..(channel.inputPort.maximumChunkNum-1) as chunk_id>
	{
		s_pChannel_${channel.index}_buffer, // Chunk start pointer
		s_pChannel_${channel.index}_buffer, // Data start pointer
		s_pChannel_${channel.index}_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
</#list>
};

</#list>
// ##CHUNK_DEFINITION_TEMPLATE::END


//portSampleRateList
// ##PORT_SAMPLE_RATE_TEMPLATE::START
<#list port_info as port>
SPortSampleRate g_astPortSampleRate_${port.taskName}_${port.portName}[] = {
	<#list port.portSampleRateList as sample_rate>
	{ 	"${sample_rate.modeName}", // Mode name
		${sample_rate.sampleRate?c}, // Sample rate
		${sample_rate.maxAvailableNum}, // Available number of data
	},
	</#list>	
};

</#list>
// ##PORT_SAMPLE_RATE_TEMPLATE::END


// ##PORT_ARRAY_TEMPLATE::START
SPort g_astPortInfo[] = {
<#list port_info as port>
	{
		${port.taskId}, // Task ID
		"${port.portName}", // Port name
		PORT_SAMPLE_RATE_${port.portSampleRateType}, // Port sample rate type
		g_astPortSampleRate_${port.taskName}_${port.portName}, // Array of sample rate list
		${port.portSampleRateList?size}, // Array element number of sample rate list
		0, //Selected sample rate index
		${port.sampleSize?c}, // Sample size
		PORT_TYPE_${port.portType}, // Port type
		<#if port.subgraphPort??>&g_astPortInfo[${port_key_to_index[port.subgraphPort.portKey]}]<#else>(SPort *) NULL</#if>, // Pointer to Subgraph port
	}, // Port information		
</#list>
};
// ##PORT_ARRAY_TEMPLATE::END


// ##AVAILABLE_CHUNK_LIST_TEMPLATE::START
<#list channel_list as channel>
SAvailableChunk g_astAvailableInputChunk_channel_${channel.index}[] = {
<#list 0..(channel.inputPort.maximumChunkNum-1) as chunk_id>
	{ ${chunk_id}, 0, (SAvailableChunk *) NULL, (SAvailableChunk *) NULL, },
</#list>
};
</#list>
// ##AVAILABLE_CHUNK_LIST_TEMPLATE::END

<#if communication_used == true>
// ##TCP_CLIENT_GENERATION_TEMPLATE::START
STCPClientInfo g_astTCPClientInfo[] = {
	<#list tcp_client_list as client>
	{
		"${client.IP}",
		${client.port?c},
	},
	</#list>
};
// ##TCP_CLIENT_GENERATION_TEMPLATE::END


// ##TCP_SERVER_GENERATION_TEMPLATE::START
STCPServerInfo g_astTCPServerInfo[] = {
	<#list tcp_server_list as server>
	{
		${server.port?c},
		(HSocket) NULL,
		(HThread) NULL,
	},
	</#list>
};
// ##TCP_SERVER_GENERATION_TEMPLATE::END

// ##TCP_COMMUNICATION_GENERATION_TEMPLATE::START
SExternalCommunicationInfo g_astExternalCommunicationInfo[] = {
<#list channel_list as channel>
	<#switch channel.communicationType>
		<#case "TCP_CLIENT_WRITER">
		<#case "TCP_CLIENT_READER">
		<#case "TCP_SERVER_WRITER">
		<#case "TCP_SERVER_READER">
	{
		${channel.index},
		COMMUNICATION_TYPE_${channel.communicationType},
		(HSocket) NULL,
		(HUEMProtocol) NULL,
	},
			<#break>
	</#switch>
</#list>
};
// ##TCP_COMMUNICATION_GENERATION_TEMPLATE::END
</#if>

SGenericMemoryAccess g_stHostMemory = {
	UKHostMemorySystem_CreateMemory,
	UKHostMemorySystem_CopyToMemory,
	UKHostMemorySystem_CopyFromMemory,
	UKHostMemorySystem_DestroyMemory,
};

<#if gpu_used == true>
SGenericMemoryAccess g_stHostToDeviceMemory = {
	UKHostMemorySystem_CreateMemory,
	UKHostMemorySystem_CopyToMemory,
	UKGPUMemorySystem_CopyHostToDeviceMemory,
	UKHostMemorySystem_DestroyMemory,
};

SGenericMemoryAccess g_stDeviceToHostMemory = {
	UKHostMemorySystem_CreateMemory,
	UKGPUMemorySystem_CopyDeviceToHostMemory,
	UKHostMemorySystem_CopyFromMemory,
	UKHostMemorySystem_DestroyMemory,
};

SGenericMemoryAccess g_stDeviceItSelfMemory = {
	UKGPUMemorySystem_CreateMemory,
	UKGPUMemorySystem_CopyDeviceToDeviceMemory,
	UKGPUMemorySystem_CopyDeviceToDeviceMemory,
	UKGPUMemorySystem_DestroyMemory,
};

SGenericMemoryAccess g_stDeviceToDeviceMemory = {
	UKGPUMemorySystem_CreateHostAllocMemory,
	UKGPUMemorySystem_CopyDeviceToHostMemory,
	UKGPUMemorySystem_CopyHostToDeviceMemory,
	UKGPUMemorySystem_DestroyHostAllocMemory,
};
</#if>


// ##SPECIFIC_CHANNEL_LIST_TEMPLATE::START
<#list channel_list as channel>

<#switch channel.communicationType>
	<#case "SHARED_MEMORY">
	<#case "TCP_SERVER_WRITER">
	<#case "TCP_CLIENT_WRITER">
	<#case "CPU_GPU">
	<#case "GPU_CPU">
	<#case "GPU_GPU">
	<#case "GPU_GPU_DIFFERENT">
SSharedMemoryChannel g_stSharedMemoryChannel_${channel.index} = {
		<#switch channel.communicationType>
			<#case "SHARED_MEMORY">
			<#case "TCP_CLIENT_WRITER">
			<#case "TCP_SERVER_WRITER">
		s_pChannel_${channel.index}_buffer, // Channel buffer pointer
		s_pChannel_${channel.index}_buffer, // Channel data start
		s_pChannel_${channel.index}_buffer, // Channel data end
				<#break>
			<#case "CPU_GPU">
			<#case "GPU_CPU">
			<#case "GPU_GPU">
			<#case "GPU_GPU_DIFFERENT">
		NULL, // Channel buffer pointer
		NULL, // Channel data start
		NULL, // Channel data end
				<#break>
		</#switch>
		0, // Channel data length
		0, // Read reference count
		0, // Write reference count
		FALSE, // Read exit setting
		FALSE, // Write exit setting
		(HThreadMutex) NULL, // Mutex
		(HThreadEvent) NULL, // Read available notice event
		(HThreadEvent) NULL, // Write available notice event
		{
			g_astChunk_channel_${channel.index}_in, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Input chunk information
		{
			g_astChunk_channel_${channel.index}_out, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Output chunk information
		CHUNK_NUM_NOT_INITIALIZED, // Written output chunk number
		g_astAvailableInputChunk_channel_${channel.index}, // Available chunk list
		${channel.inputPort.maximumChunkNum}, // maximum input port chunk size for all port sample rate cases
		(SAvailableChunk *) NULL, // Chunk list head
		(SAvailableChunk *) NULL, // Chunk list tail
		<#switch channel.communicationType>
			<#case "SHARED_MEMORY">
			<#case "TCP_CLIENT_WRITER">
			<#case "TCP_SERVER_WRITER">
		&g_stHostMemory, // Host memory access API
		TRUE, // memory is statically allocated
				<#break>
			<#case "CPU_GPU">
		&g_stHostToDeviceMemory, // Host memory access API
		FALSE, // memory is statically allocated
				<#break>
			<#case "GPU_CPU">
		&g_stDeviceToHostMemory, // Host memory access API
		FALSE, // memory is statically allocated
				<#break>
			<#case "GPU_GPU">
		&g_stDeviceItSelfMemory, // Host memory access API
		FALSE, // memory is statically allocated
				<#break>
			<#case "GPU_GPU_DIFFERENT">
		&g_stDeviceToDeviceMemory, // Host memory access API
		FALSE, // memory is statically allocated
				<#break>
		</#switch>
};
		<#break>
	</#switch>

	<#switch channel.communicationType>
		<#case "TCP_CLIENT_WRITER">
		<#case "TCP_CLIENT_READER">
		<#case "TCP_SERVER_WRITER">
		<#case "TCP_SERVER_READER">
STCPSocketChannel g_stTCPSocketChannel_${channel.index} = {
		<#switch channel.communicationType>
			<#case "TCP_CLIENT_WRITER">
			<#case "TCP_CLIENT_READER">
	(STCPClientInfo *) &g_astTCPClientInfo[${channel.tcpClientIndex}], // STCPClientInfo *pstClientInfo;
				<#break>
			<#default>
	(STCPClientInfo *) NULL, // STCPClientInfo *pstClientInfo;
		</#switch>
		break; 
	(SExternalCommunicationInfo *) NULL, // SExternalCommunicationInfo *pstCommunicationInfo;
	(HThread) NULL, // HThread hReceivingThread;
	NULL, // char *pBuffer;
	0, // int nBufLen;
	(HThreadMutex) NULL, // HThreadMutex hMutex;
	FALSE, // uem_bool bChannelExit;
		<#switch channel.communicationType>
			<#case "TCP_CLIENT_WRITER">
			<#case "TCP_SERVER_WRITER">
	&g_stSharedMemoryChannel_${channel.index}, // SSharedMemoryChannel *pstInternalChannel;
				<#break>
			<#default>
	(SSharedMemoryChannel *) NULL, // SSharedMemoryChannel *pstInternalChannel;
		</#switch>
};
		<#break>
	</#switch>
</#list>
// ##SPECIFIC_CHANNEL_LIST_TEMPLATE::END


// ##CHANNEL_LIST_TEMPLATE::START
SChannel g_astChannels[] = {
<#list channel_list as channel>
	{
		${channel.index}, // Channel ID
		${channel.nextChannelIndex}, // Next channel index (which is used for single port is connecting to multiple channels)
		COMMUNICATION_TYPE_${channel.communicationType}, // Channel communication type
		CHANNEL_TYPE_${channel.channelType}, // Channel type
		CHANNEL_${channel.index}_SIZE, // Channel size
		{
			${channel.inputPort.taskId}, // Task ID
			"${channel.inputPort.portName}", // Port name
			PORT_SAMPLE_RATE_${channel.inputPort.portSampleRateType}, // Port sample rate type
			g_astPortSampleRate_${channel.inputPort.taskName}_${channel.inputPort.portName}, // Array of sample rate list
			${channel.inputPort.portSampleRateList?size}, // Array element number of sample rate list
			0, //Selected sample rate index
			${channel.inputPort.sampleSize?c}, // Sample size
			PORT_TYPE_${channel.inputPort.portType}, // Port type
			<#if channel.inputPort.subgraphPort??>&g_astPortInfo[${port_key_to_index[channel.inputPort.subgraphPort.portKey]}]<#else>(SPort *) NULL</#if>, // Pointer to Subgraph port
		}, // Input port information
		{
			${channel.outputPort.taskId}, // Task ID
			"${channel.outputPort.portName}", // Port name
			PORT_SAMPLE_RATE_${channel.outputPort.portSampleRateType}, // Port sample rate type
			g_astPortSampleRate_${channel.outputPort.taskName}_${channel.outputPort.portName}, // Array of sample rate list
			${channel.outputPort.portSampleRateList?size}, // Array element number of sample rate list
			0, //Selected sample rate index
			${channel.outputPort.sampleSize?c}, // Sample size
			PORT_TYPE_${channel.outputPort.portType}, // Port type
			<#if channel.outputPort.subgraphPort??>&g_astPortInfo[${port_key_to_index[channel.outputPort.subgraphPort.portKey]}]<#else>(SPort *) NULL</#if>, // Pointer to Subgraph port
		}, // Output port information
		${channel.initialDataLen?c}, // Initial data length
	<#switch channel.communicationType>
		<#case "SHARED_MEMORY">
		<#case "CPU_GPU">
		<#case "GPU_CPU">
		<#case "GPU_GPU">
		<#case "GPU_GPU_DIFFERENT">
		&g_stSharedMemoryChannel_${channel.index}, // specific shared memory channel structure pointer
			<#break>
		<#case "TCP_CLIENT_WRITER">
		<#case "TCP_CLIENT_READER">
		<#case "TCP_SERVER_WRITER">
		<#case "TCP_SERVER_READER">
		&g_stTCPSocketChannel_${channel.index}, // specific TCP socket channel structure pointer
			<#break>
	</#switch>
	},
</#list>
};
// ##CHANNEL_LIST_TEMPLATE::END


int g_nChannelNum = ARRAYLEN(g_astChannels);
<#if (tcp_server_list?size > 0) >
int g_nTCPServerInfoNum = ARRAYLEN(g_astTCPServerInfo);
</#if>
<#if communication_used == true>
int g_nExternalCommunicationInfoNum = ARRAYLEN(g_astExternalCommunicationInfo);
</#if>
