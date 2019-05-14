/* uem_channel_data.c made by UEM Translator */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

<#if communication_used == true>
#include <UCDynamicSocket.h>
#include <UCFixedSizeQueue.h>
	<#if used_communication_list?seq_contains("tcp")>
#include <UCTCPSocket.h>
	</#if>
	<#if used_communication_list?seq_contains("bluetooth")>
#include <UCBluetoothSocket.h>
	</#if>
	<#if used_communication_list?seq_contains("serial")>
#include <UCSerialPort.h>
	</#if>
</#if>

#include <UKHostSystem.h>
#include <UKSharedMemoryChannel.h>
#include <UKChannel.h>

#include <uem_data.h>

<#if communication_used == true>
#include <UKUEMProtocol.h>

	<#if used_communication_list?seq_contains("tcp")>
#include <UKTCPServerManager.h>
#include <UKTCPSocketChannel.h>

#include <uem_tcp_data.h>
	</#if>
	
	<#if used_communication_list?seq_contains("bluetooth")>
#include <UKBluetoothModule.h>
#include <UKBluetoothChannel.h>

#include <uem_bluetooth_data.h>
	</#if>
	<#if used_communication_list?seq_contains("serial")>
#include <UKSerialModule.h>
#include <UKSerialChannel.h>

#include <uem_bluetooth_data.h>
	</#if>
</#if>
<#if gpu_used == true>
#include <UKGPUSystem.h>
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
SChunk g_astChunk_channel_${channel.index}_out[${channel.outputPort.maximumChunkNum?c}];
SChunk g_astChunk_channel_${channel.index}_in[${channel.inputPort.maximumChunkNum?c}];

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
	{ // index number: ${port?index}
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
SAvailableChunk g_astAvailableInputChunk_channel_${channel.index}[${channel.inputPort.maximumChunkNum?c}];

</#list>
// ##AVAILABLE_CHUNK_LIST_TEMPLATE::END

<#if used_communication_list?seq_contains("tcp")>
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

<#if used_communication_list?seq_contains("bluetooth")>
SBluetoothInfo g_astBluetoothMasterInfo[] = {
	<#list bluetooth_master_list as master>
	{
		"${master.portAddress}", // target mac address
		(HSocket) NULL, // socket handle
		(HThread) NULL, // thread handle
		(HConnector) NULL, // connector handle
		${master.channelAccessNum}, // max channel access number
		(HSerialCommunicationManager) NULL, // Serial communication manager handle
		FALSE, // initialized or not
	},
	</#list>
};

SBluetoothInfo g_astBluetoothSlaveInfo[] = {
	<#list bluetooth_slave_list as slave>
	{
		"${slave.portAddress}", // slave mac address
		(HSocket) NULL, // socket handle
		(HThread) NULL, // thread handle
		(HConnector) NULL, // connector handle
		${slave.channelAccessNum}, // max channel access number
		(HSerialCommunicationManager) NULL, // Serial communication manager handle
		FALSE, // initialized or not
	},
	</#list>
};

</#if>

<#if used_communication_list?seq_contains("serial")>
SSerialInfo g_astSerialMasterInfo[] = {
	<#list serial_master_list as master>
	{
		"${master.portAddress}", // serial port path
		(HSerialPort) NULL, // hSerialPort handle
		(HThread) NULL, // thread handle
		(HConnector) NULL, // connector handle
		${master.channelAccessNum}, // max channel access number
		(HSerialCommunicationManager) NULL, // Serial communication manager handle	
		FALSE, // initialized or not
	},
	</#list>
};


SSerialInfo g_astSerialSlaveInfo[] = {
	<#list serial_slave_list as slave>
	{
		"${slave.portAddress}", // serial port path
		(HSerialPort) NULL, // hSerialPort handle
		(HThread) NULL, // thread handle
		(HConnector) NULL, // connector handle
		${slave.channelAccessNum}, // max channel access number
		(HSerialCommunicationManager) NULL, // Serial communication manager handle		
		FALSE, // initialized or not
	},
	</#list>
};


</#if>


SGenericMemoryAccess g_stHostMemory = {
	UKHostSystem_CreateMemory,
	UKHostSystem_CopyToMemory,
	UKHostSystem_CopyInMemory,
	UKHostSystem_CopyFromMemory,
	UKHostSystem_DestroyMemory,
};

<#if gpu_used == true>
SGenericMemoryAccess g_stHostToDeviceMemory = {
	UKHostSystem_CreateMemory,
	UKHostSystem_CopyToMemory,
	UKHostSystem_CopyInMemory,
	UKGPUSystem_CopyHostToDeviceMemory,
	UKHostSystem_DestroyMemory,
};

SGenericMemoryAccess g_stDeviceToHostMemory = {
	UKHostSystem_CreateMemory,
	UKGPUSystem_CopyDeviceToHostMemory,
	UKHostSystem_CopyInMemory,
	UKHostSystem_CopyFromMemory,
	UKHostSystem_DestroyMemory,
};

SGenericMemoryAccess g_stDeviceItSelfMemory = {
	UKGPUSystem_CreateMemory,
	UKGPUSystem_CopyDeviceToDeviceMemory,
	UKGPUSystem_CopyDeviceToDeviceMemory,
	UKGPUSystem_CopyDeviceToDeviceMemory,
	UKGPUSystem_DestroyMemory,
};

SGenericMemoryAccess g_stDeviceToDeviceMemory = {
	UKGPUSystem_CreateHostAllocMemory,
	UKGPUSystem_CopyDeviceToHostMemory,
	UKHostSystem_CopyInMemory, // host alloc memory can use host memcpy function
	UKGPUSystem_CopyHostToDeviceMemory,
	UKGPUSystem_DestroyHostAllocMemory,
};
</#if>


// ##SPECIFIC_CHANNEL_LIST_TEMPLATE::START
<#list channel_list as channel>

	<#switch channel.communicationType>
		<#case "SHARED_MEMORY">
		<#case "TCP_SERVER_WRITER">
		<#case "TCP_CLIENT_WRITER">
		<#case "BLUETOOTH_MASTER_WRITER">
		<#case "BLUETOOTH_SLAVE_WRITER">
		<#case "SERIAL_MASTER_WRITER">
		<#case "SERIAL_SLAVE_WRITER">
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
	(void*) NULL, // Channel buffer pointer
	(void*) NULL, // Channel data start
	(void*) NULL, // Channel data end
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
			<#if (channel.initialDataLen > 0)>
	FALSE, // initial data is updated
			<#else>
	TRUE, // initial data is updated
			</#if>
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
	(STCPClientInfo *) &g_astTCPClientInfo[${channel.socketInfoIndex}], // STCPClientInfo *pstClientInfo;
				<#break>
			<#default>
	(STCPClientInfo *) NULL, // STCPClientInfo *pstClientInfo;
		</#switch>
	(SExternalCommunicationInfo *) NULL, // SExternalCommunicationInfo *pstCommunicationInfo;
	(HThread) NULL, // HThread hReceivingThread;
	(char *) NULL, // char *pBuffer;
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
	
	<#switch channel.communicationType>
		<#case "BLUETOOTH_MASTER_WRITER">
		<#case "BLUETOOTH_SLAVE_WRITER">
		<#case "SERIAL_MASTER_WRITER">
		<#case "SERIAL_SLAVE_WRITER">
		
SSerialWriterChannel g_stSerialWriterChannel_${channel.index} = {
		<#switch channel.communicationType>
			<#case "BLUETOOTH_MASTER_WRITER">
	(void *) &g_astBluetoothMasterInfo[${channel.socketInfoIndex}],
				<#break>
			<#case "BLUETOOTH_SLAVE_WRITER">
	(void *) &g_astBluetoothSlaveInfo[${channel.socketInfoIndex}],
				<#break>
			<#case "SERIAL_MASTER_WRITER">
	(void *) &g_astSerialMasterInfo[${channel.socketInfoIndex}],			
				<#break>
			<#case "SERIAL_SLAVE_WRITER">
	(void *) &g_astSerialSlaveInfo[${channel.socketInfoIndex}],			
				<#break>
		</#switch>
	(HFixedSizeQueue) NULL,
	(HThread) NULL,
	(char*) NULL,
	0,
	(HThreadMutex) NULL,
	FALSE,
	&g_stSharedMemoryChannel_${channel.index},	
};
			<#break>
	</#switch>

	
	<#switch channel.communicationType>
		<#case "BLUETOOTH_MASTER_READER">
		<#case "BLUETOOTH_SLAVE_READER">
		<#case "SERIAL_MASTER_READER">
		<#case "SERIAL_SLAVE_READER">
SSerialReaderChannel g_stSerialReaderChannel_${channel.index} = {
			<#switch channel.communicationType>
				<#case "BLUETOOTH_MASTER_READER">
	(void *) &g_astBluetoothMasterInfo[${channel.socketInfoIndex}],
					<#break>
				<#case "BLUETOOTH_SLAVE_READER">
	(void *) &g_astBluetoothSlaveInfo[${channel.socketInfoIndex}],
					<#break>
				<#case "SERIAL_MASTER_READER">
	(void *) &g_astSerialMasterInfo[${channel.socketInfoIndex}],
					<#break>
				<#case "SERIAL_SLAVE_READER">
	(void *) &g_astSerialSlaveInfo[${channel.socketInfoIndex}],
					<#break>
			</#switch>
	(HFixedSizeQueue) NULL, // response queue
	(HThreadMutex) NULL, // mutex variable
	FALSE, // channel exit flag
			<#switch channel.accessType>
				<#case "CPU_ONLY">
	&g_stHostMemory, // SGenericMemoryAccess *pstReaderAccess - READER-part channel memory access API
					<#break>
				<#case "CPU_GPU">
	&g_stHostToDeviceMemory, // SGenericMemoryAccess *pstReaderAccess - READER-part channel memory access API
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
		&(g_astPortInfo[${channel.inputPortIndex}]), // Outer-most input port information (port name: ${channel.inputPort.portName})
		&(g_astPortInfo[${channel.outputPortIndex}]), // Outer-most output port information (port name: ${channel.outputPort.portName})
		${channel.initialDataLen?c}, // Initial data length
		${channel.processerId?c}, // Processor ID
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
		<#case "BLUETOOTH_MASTER_WRITER">
		<#case "BLUETOOTH_SLAVE_WRITER">
		<#case "SERIAL_MASTER_WRITER">
		<#case "SERIAL_SLAVE_WRITER">
			&g_stSerialWriterChannel_${channel.index}, // specific bluetooth/serial channel structure pointer
			<#break>
		<#case "BLUETOOTH_MASTER_READER">
		<#case "BLUETOOTH_SLAVE_READER">
		<#case "SERIAL_MASTER_READER">
		<#case "SERIAL_SLAVE_READER">
			&g_stSerialReaderChannel_${channel.index}, // specific bluetooth/serial channel structure pointer
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
	UKSharedMemoryChannel_FillInitialData,
	UKSharedMemoryChannel_Finalize, // fnFinalize
	(FnChannelAPIInitialize) NULL,
	(FnChannelAPIFinalize) NULL,
};

<#if used_communication_list?seq_contains("tcp")>
SChannelAPI g_stTCPSocketChannelWriter = {
	UKTCPSocketChannel_Initialize, // fnInitialize
	(FnChannelReadFromQueue) NULL, // fnReadFromQueue
	(FnChannelReadFromBuffer) NULL, // fnReadFromBuffer
	UKTCPSocketChannel_WriteToQueue, // fnWriteToQueue
	UKTCPSocketChannel_WriteToBuffer, // fnWriteToBuffer
	(FnChannelGetAvailableChunk) NULL, // fnGetAvailableChunk
	UKTCPSocketChannel_GetNumOfAvailableData, // fnGetNumOfAvailableData
	UKTCPSocketChannel_Clear, // fnClear
	UKTCPSocketChannel_SetExit,
	UKTCPSocketChannel_ClearExit,
	UKTCPSocketChannel_FillInitialData,
	UKTCPSocketChannel_Finalize, // fnFinalize
	UKTCPServerManager_Initialize,
	UKTCPServerManager_Finalize,
};


SChannelAPI g_stTCPSocketChannelReader = {
	UKTCPSocketChannel_Initialize, // fnInitialize
	UKTCPSocketChannel_ReadFromQueue, // fnReadFromQueue
	UKTCPSocketChannel_ReadFromBuffer, // fnReadFromBuffer
	(FnChannelWriteToQueue) NULL, // fnWriteToQueue
	(FnChannelWriteToBuffer) NULL, // fnWriteToBuffer
	UKTCPSocketChannel_GetAvailableChunk, // fnGetAvailableChunk
	UKTCPSocketChannel_GetNumOfAvailableData, // fnGetNumOfAvailableData
	UKTCPSocketChannel_Clear, // fnClear
	UKTCPSocketChannel_SetExit,
	UKTCPSocketChannel_ClearExit,
	(FnChannelFillInitialData) NULL,
	UKTCPSocketChannel_Finalize, // fnFinalize
	(FnChannelAPIInitialize) NULL,
	(FnChannelAPIFinalize) NULL,
};
</#if>


<#if used_communication_list?seq_contains("bluetooth")>
SChannelAPI g_stBluetoothChannelWriter = {
	UKBluetoothChannel_Initialize, // fnInitialize
	(FnChannelReadFromQueue) NULL, // fnReadFromQueue
	(FnChannelReadFromBuffer) NULL, // fnReadFromBuffer
	UKBluetoothChannel_WriteToQueue, // fnWriteToQueue
	UKBluetoothChannel_WriteToBuffer, // fnWriteToBuffer
	(FnChannelGetAvailableChunk) NULL, // fnGetAvailableChunk
	UKBluetoothChannel_GetNumOfAvailableData, // fnGetNumOfAvailableData
	UKBluetoothChannel_Clear, // fnClear
	UKBluetoothChannel_SetExit,
	UKBluetoothChannel_ClearExit,
	UKBluetoothChannel_FillInitialData,
	UKBluetoothChannel_Finalize, // fnFinalize
	UKBluetoothModule_Initialize,
	UKBluetoothModule_Finalize,
};


SChannelAPI g_stBluetoothChannelReader = {
	UKBluetoothChannel_Initialize, // fnInitialize
	UKBluetoothChannel_ReadFromQueue, // fnReadFromQueue
	UKBluetoothChannel_ReadFromBuffer, // fnReadFromBuffer
	(FnChannelWriteToQueue) NULL, // fnWriteToQueue
	(FnChannelWriteToBuffer) NULL, // fnWriteToBuffer
	UKBluetoothChannel_GetAvailableChunk, // fnGetAvailableChunk
	UKBluetoothChannel_GetNumOfAvailableData, // fnGetNumOfAvailableData
	UKBluetoothChannel_Clear, // fnClear
	UKBluetoothChannel_SetExit,
	UKBluetoothChannel_ClearExit,
	(FnChannelFillInitialData) NULL,
	UKBluetoothChannel_Finalize, // fnFinalize
	(FnChannelAPIInitialize) NULL,
	(FnChannelAPIFinalize) NULL,
};
</#if>

<#if used_communication_list?seq_contains("serial")>
SChannelAPI g_stSerialChannelWriter = {
	UKSerialChannel_Initialize, // fnInitialize
	(FnChannelReadFromQueue) NULL, // fnReadFromQueue
	(FnChannelReadFromBuffer) NULL, // fnReadFromBuffer
	UKSerialChannel_WriteToQueue, // fnWriteToQueue
	UKSerialChannel_WriteToBuffer, // fnWriteToBuffer
	(FnChannelGetAvailableChunk) NULL, // fnGetAvailableChunk
	UKSerialChannel_GetNumOfAvailableData, // fnGetNumOfAvailableData
	UKSerialChannel_Clear, // fnClear
	UKSerialChannel_SetExit,
	UKSerialChannel_ClearExit,
	UKSerialChannel_FillInitialData,
	UKSerialChannel_Finalize, // fnFinalize
	UKSerialModule_Initialize,
	UKSerialModule_Finalize,
};


SChannelAPI g_stSerialChannelReader = {
	UKSerialChannel_Initialize, // fnInitialize
	UKSerialChannel_ReadFromQueue, // fnReadFromQueue
	UKSerialChannel_ReadFromBuffer, // fnReadFromBuffer
	(FnChannelWriteToQueue) NULL, // fnWriteToQueue
	(FnChannelWriteToBuffer) NULL, // fnWriteToBuffer
	UKSerialChannel_GetAvailableChunk, // fnGetAvailableChunk
	UKSerialChannel_GetNumOfAvailableData, // fnGetNumOfAvailableData
	UKSerialChannel_Clear, // fnClear
	UKSerialChannel_SetExit,
	UKSerialChannel_ClearExit,
	(FnChannelFillInitialData) NULL,
	UKSerialChannel_Finalize, // fnFinalize
	(FnChannelAPIInitialize) NULL,
	(FnChannelAPIFinalize) NULL,
};
</#if>



SChannelAPI *g_astChannelAPIList[] = {
	&g_stSharedMemoryChannel,
<#if used_communication_list?seq_contains("tcp")>
	&g_stTCPSocketChannelWriter,
	&g_stTCPSocketChannelReader,
</#if>
<#if used_communication_list?seq_contains("bluetooth")>
	&g_stBluetoothChannelWriter,
	&g_stBluetoothChannelReader,
</#if>

<#if used_communication_list?seq_contains("serial")>
	&g_stSerialChannelWriter,
	&g_stSerialChannelReader,
</#if>

};

<#if used_communication_list?seq_contains("bluetooth")>
SSocketAPI stBluetoothAPI = {
	UCBluetoothSocket_Bind,
	UCBluetoothSocket_Accept,
	UCBluetoothSocket_Connect,
	(FnSocketCreate) NULL,
	(FnSocketDestroy) NULL,
};
</#if>


<#if used_communication_list?seq_contains("tcp")>
SSocketAPI stTCPAPI = {
	UCTCPSocket_Bind,
	UCTCPSocket_Accept,
	UCTCPSocket_Connect,
	(FnSocketCreate) NULL,
	(FnSocketDestroy) NULL,
};		
</#if>


/*
SSocketAPI stUnixDomainSocketAPI = {
	UCUnixDomainSocket_Bind,
	UCUnixDomainSocket_Accept,
	UCUnixDomainSocket_Connect,
	(FnSocketCreate) NULL,
	UCUnixDomainSocket_Destroy,
};
*/

#ifdef __cplusplus
extern "C"
{
#endif

<#assign printed=false />
uem_result ChannelAPI_SetSocketAPIs()
{
	uem_result result = ERR_UEM_UNKNOWN;

<#if used_communication_list?seq_contains("bluetooth")>
	result = UCDynamicSocket_SetAPIList(SOCKET_TYPE_BLUETOOTH, &stBluetoothAPI);
	ERRIFGOTO(result, _EXIT);
	<#assign printed=true />
</#if>
<#if used_communication_list?seq_contains("tcp")>
	result = UCDynamicSocket_SetAPIList(SOCKET_TYPE_TCP, &stTCPAPI);
	ERRIFGOTO(result, _EXIT);
	<#assign printed=true />
</#if>

	result = ERR_UEM_NOERROR;
<#if (printed == true)>
_EXIT:
</#if>
	return result;
}

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
<#if used_communication_list?seq_contains("tcp")>
		*ppstChannelAPI = &g_stTCPSocketChannelReader;
<#else>
		ERRASSIGNGOTO(result, ERR_UEM_NOT_SUPPORTED_YET, _EXIT)
</#if>
		break;
	case COMMUNICATION_TYPE_TCP_SERVER_WRITER:
	case COMMUNICATION_TYPE_TCP_CLIENT_WRITER:
<#if used_communication_list?seq_contains("tcp")>
		*ppstChannelAPI = &g_stTCPSocketChannelWriter;
<#else>
		ERRASSIGNGOTO(result, ERR_UEM_NOT_SUPPORTED, _EXIT)
</#if>
		break;
		
	case COMMUNICATION_TYPE_BLUETOOTH_MASTER_WRITER:
	case COMMUNICATION_TYPE_BLUETOOTH_SLAVE_WRITER:
<#if used_communication_list?seq_contains("bluetooth")>
		*ppstChannelAPI = &g_stBluetoothChannelWriter;	
<#else>
		ERRASSIGNGOTO(result, ERR_UEM_NOT_SUPPORTED, _EXIT)
</#if>
		break;
	case COMMUNICATION_TYPE_BLUETOOTH_MASTER_READER:
	case COMMUNICATION_TYPE_BLUETOOTH_SLAVE_READER:
<#if used_communication_list?seq_contains("bluetooth")>
		*ppstChannelAPI = &g_stBluetoothChannelReader;	
<#else>
		ERRASSIGNGOTO(result, ERR_UEM_NOT_SUPPORTED, _EXIT)
</#if>
		break;

	case COMMUNICATION_TYPE_SERIAL_MASTER_WRITER:
	case COMMUNICATION_TYPE_SERIAL_SLAVE_WRITER:
<#if used_communication_list?seq_contains("serial")>
		*ppstChannelAPI = &g_stSerialChannelWriter;	
<#else>
		ERRASSIGNGOTO(result, ERR_UEM_NOT_SUPPORTED, _EXIT)
</#if>
		break;
	case COMMUNICATION_TYPE_SERIAL_MASTER_READER:
	case COMMUNICATION_TYPE_SERIAL_SLAVE_READER:
<#if used_communication_list?seq_contains("serial")>
		*ppstChannelAPI = &g_stSerialChannelReader;	
<#else>
		ERRASSIGNGOTO(result, ERR_UEM_NOT_SUPPORTED, _EXIT)
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

#ifdef __cplusplus
}
#endif

int g_nChannelNum = ARRAYLEN(g_astChannels);
int g_nChannelAPINum = ARRAYLEN(g_astChannelAPIList);
<#if communication_used == true>
	<#if used_communication_list?seq_contains("tcp")>
int g_nExternalCommunicationInfoNum = ARRAYLEN(g_astExternalCommunicationInfo);
	</#if>

	<#if (tcp_server_list?size > 0) >
int g_nTCPServerInfoNum = ARRAYLEN(g_astTCPServerInfo);
	<#else>
int g_nTCPServerInfoNum = 0;
	</#if>
	
	<#if (bluetooth_master_list?size > 0) >
int g_nBluetoothMasterNum = ARRAYLEN(g_astBluetoothMasterInfo);	
	<#else>
int g_nBluetoothMasterNum = 0;
	</#if>

	<#if (bluetooth_slave_list?size > 0) >
int g_nBluetoothSlaveNum = ARRAYLEN(g_astBluetoothSlaveInfo);	
	<#else>
int g_nBluetoothSlaveNum = 0;
	</#if>

	<#if (serial_master_list?size > 0) >
int g_nSerialMasterInfoNum = ARRAYLEN(g_astSerialMasterInfo);	
	<#else>
int g_nSerialMasterInfoNum = 0;
	</#if>
			
	<#if (serial_slave_list?size > 0) >
int g_nSerialSlaveInfoNum = ARRAYLEN(g_astSerialSlaveInfo);	
	<#else>
int g_nSerialSlaveInfoNum = 0;
	</#if>				
</#if>



