/* uem_channel_data.c made by UEM Translator */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

<#if communication_used == true>
#include <UCDynamicSocket.h>
</#if>

#include <UKHostMemorySystem.h>
#include <UKSharedMemoryChannel.h>
#include <UKChannel.h>

#include <uem_data.h>

<#if communication_used == true>
#include <UKUEMProtocol.h>
#include <UKTCPServerManager.h>
#include <UKTCPSocketChannel.h>

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
SChunk g_astChunk_channel_${channel.index}_out[${channel.outputPort.maximumChunkNum}];
SChunk g_astChunk_channel_${channel.index}_in[${channel.inputPort.maximumChunkNum}];

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
SAvailableChunk g_astAvailableInputChunk_channel_${channel.index}[${channel.inputPort.maximumChunkNum}];
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
SSharedMemoryChannel g_stSharedMemoryChannel_${channel.index} = {
		ACCESS_TYPE_${channel.accessType},
		<#switch channel.accessType>
			<#case "CPU_ONLY">
			<#case "CPU_GPU">
			<#case "GPU_CPU">
		s_pChannel_${channel.index}_buffer, // Channel buffer pointer
		s_pChannel_${channel.index}_buffer, // Channel data start
		s_pChannel_${channel.index}_buffer, // Channel data end
				<#break>
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
		${channel.outputPort.maximumChunkNum?c}, // maximum chunk size for all port sample rate cases (output port)
		${channel.inputPort.maximumChunkNum?c}, // maximum input port chunk size for all port sample rate cases (input port)
		(SAvailableChunk *) NULL, // Chunk list head
		(SAvailableChunk *) NULL, // Chunk list tail
		<#switch channel.accessType>
			<#case "CPU_ONLY">
		&g_stHostMemory, // Host memory access API
		TRUE, // memory is statically allocated
				<#break>
			<#case "CPU_GPU">
		&g_stHostToDeviceMemory, // Host memory access API
		TRUE, // memory is statically allocated
				<#break>
			<#case "GPU_CPU">
		&g_stDeviceToHostMemory, // Host memory access API
		TRUE, // memory is statically allocated
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
	(SGenericMemoryAccess *) NULL, // SGenericMemoryAccess *pstReaderAccess - READER-part channel memory access API
				<#break>
			<#case "TCP_CLIENT_READER">
			<#case "TCP_SERVER_READER">
	(SSharedMemoryChannel *) NULL, // SSharedMemoryChannel *pstInternalChannel;
				<#switch channel.accessType>
					<#case "CPU_ONLY">
	&g_stHostMemory, // SGenericMemoryAccess *pstReaderAccess - READER-part channel memory access API
						<#break>
					<#case "CPU_GPU">
	&g_stHostToDeviceMemory, // SGenericMemoryAccess *pstReaderAccess - READER-part channel memory access API
						<#break>
					<#default>
	error (cannot generate other access type)
				</#switch>
				<#break>
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


SChannelAPI g_stSharedMemoryChannel = {
	UKSharedMemoryChannel_Initialize, // fnInitialize
	UKSharedMemoryChannel_ReadFromQueue, // fnReadFromQueue
	UKSharedMemoryChannel_ReadFromBuffer, // fnReadFromBuffer
	UKSharedMemoryChannel_WriteToQueue, // fnWriteToQueue
	UKSharedMemoryChannel_WriteToBuffer, // fnWriteToBuffer
	UKSharedMemoryChannel_GetAvailableChunk, // fnGetAvailableChunk
	UKSharedMemoryChannel_GetNumOfAvailableData, // fnGetNumOfAvailableData
	UKSharedMemoryChannel_Clear, // fnClear
	UKSharedMemoryChannel_SetExit,
	UKSharedMemoryChannel_ClearExit,
	UKSharedMemoryChannel_Finalize, // fnFinalize
	NULL,
	NULL,
};

<#if communication_used == true>
SChannelAPI g_stTCPSocketChannelWriter = {
	UKTCPSocketChannel_Initialize, // fnInitialize
	NULL, // fnReadFromQueue
	NULL, // fnReadFromBuffer
	UKTCPSocketChannel_WriteToQueue, // fnWriteToQueue
	UKTCPSocketChannel_WriteToBuffer, // fnWriteToBuffer
	NULL, // fnGetAvailableChunk
	NULL, // fnGetNumOfAvailableData
	UKTCPSocketChannel_Clear, // fnClear
	UKTCPSocketChannel_SetExit,
	UKTCPSocketChannel_ClearExit,
	UKTCPSocketChannel_Finalize, // fnFinalize
	UKTCPServerManager_Initialize,
	UKTCPServerManager_Finalize,
};


SChannelAPI g_stTCPSocketChannelReader = {
	UKTCPSocketChannel_Initialize, // fnInitialize
	UKTCPSocketChannel_ReadFromQueue, // fnReadFromQueue
	UKTCPSocketChannel_ReadFromBuffer, // fnReadFromBuffer
	NULL, // fnWriteToQueue
	NULL, // fnWriteToBuffer
	UKTCPSocketChannel_GetAvailableChunk, // fnGetAvailableChunk
	UKTCPSocketChannel_GetNumOfAvailableData, // fnGetNumOfAvailableData
	UKTCPSocketChannel_Clear, // fnClear
	UKTCPSocketChannel_SetExit,
	UKTCPSocketChannel_ClearExit,
	UKTCPSocketChannel_Finalize, // fnFinalize
	NULL,
	NULL,
};
</#if>


SChannelAPI *g_astChannelAPIList[] = {
		&g_stSharedMemoryChannel,
<#if communication_used == true>
		&g_stTCPSocketChannelWriter,
		&g_stTCPSocketChannelReader,
</#if>
};


uem_result ChannelAPI_GetAPIStructureFromCommunicationType(IN ECommunicationType enType, OUT SChannelAPI **ppstChannelAPI)
{
	uem_result result = ERR_UEM_UNKNOWN;
	switch(enType)
	{
	case COMMUNICATION_TYPE_SHARED_MEMORY:
		*ppstChannelAPI = &g_stSharedMemoryChannel;
		break;
	case COMMUNICATION_TYPE_TCP_SERVER_READER:
	case COMMUNICATION_TYPE_TCP_CLIENT_READER:
<#if communication_used == true>
		*ppstChannelAPI = &g_stTCPSocketChannelReader;
<#else>
		ERRASSIGNGOTO(result, ERR_UEM_NOT_SUPPORTED_YET, _EXIT)
</#if>
		break;
	case COMMUNICATION_TYPE_TCP_SERVER_WRITER:
	case COMMUNICATION_TYPE_TCP_CLIENT_WRITER:
<#if communication_used == true>
		*ppstChannelAPI = &g_stTCPSocketChannelWriter;
<#else>
		ERRASSIGNGOTO(result, ERR_UEM_NOT_SUPPORTED_YET, _EXIT)
</#if>
		break;
	default:
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT)
		break;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


int g_nChannelNum = ARRAYLEN(g_astChannels);
int g_nChannelAPINum = ARRAYLEN(g_astChannelAPIList);
<#if communication_used == true>
<#if (tcp_server_list?size > 0) >
int g_nTCPServerInfoNum = ARRAYLEN(g_astTCPServerInfo);
<#else>
int g_nTCPServerInfoNum = 0;
</#if>
int g_nExternalCommunicationInfoNum = ARRAYLEN(g_astExternalCommunicationInfo);
</#if>





