/* uem_channel_data.c made by UEM Translator */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

<#if communication_used == true>
#include <UCSocket.h>
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
	<#if used_communication_list?seq_contains("secure_tcp")>
#include <UCSecureTCPSocket.h>
	</#if>
</#if>

#include <UKHostSystem.h>
#include <UKSharedMemoryChannel.h>
#include <UKChannel.h>

#include <uem_data.h>

<#if communication_used == true>
#include <UKUEMProtocol.h>
#include <UKVirtualCommunication.h>
#include <UKVirtualEncryption.h>
#include <UKRemoteChannel.h>
#include <uem_remote_data.h>
#include <UKSocketCommunication.h>

	<#if used_communication_list?seq_contains("tcp")>
#include <UKTCPServerManager.h>
#include <UKTCPCommunication.h>
#include <uem_tcp_data.h>
	</#if>
	<#if used_communication_list?seq_contains("bluetooth")>
#include <UKBluetoothModule.h>
#include <UKBluetoothCommunication.h>
#include <uem_bluetooth_data.h>
	</#if>
	<#if used_communication_list?seq_contains("serial")>
#include <UKSerialModule.h>
#include <UKSerialCommunication.h>
#include <uem_serial_data.h>
	</#if>
	<#if used_communication_list?seq_contains("secure_tcp")>
#include <UKSecureTCPServerManager.h>
#include <UKSecureTCPCommunication.h>
#include <uem_secure_tcp_data.h>
	</#if>
</#if>
<#if gpu_used == true>
#include <UKGPUSystem.h>
</#if>

<#if encryption_used == true>
<#if used_encryption_list?seq_contains("lea")>
#include <UKEncryptionLEA.h>
</#if>
<#if used_encryption_list?seq_contains("hight")>
#include <UKEncryptionHIGHT.h>
</#if>
<#if used_encryption_list?seq_contains("seed")>
#include <UKEncryptionSEED.h>
</#if>
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
		<#if sample_rate.modeName == "Default">
	{	DEFAULT_STRING_NAME, // Mode name
		<#else>
	{	"${sample_rate.modeName}", // Mode name
		</#if>
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
<#if platform == "windows" && (port_info?size == 0)>
	0, 
</#if>
};
// ##PORT_ARRAY_TEMPLATE::END


// ##AVAILABLE_CHUNK_LIST_TEMPLATE::START
<#list channel_list as channel>
SAvailableChunk g_astAvailableInputChunk_channel_${channel.index}[${channel.inputPort.maximumChunkNum?c}];

</#list>
// ##AVAILABLE_CHUNK_LIST_TEMPLATE::END


<#if used_communication_list?seq_contains("tcp")>
SVirtualCommunicationAPI g_stTCPCommunication = {
	UKTCPCommunication_Create,
	UKSocketCommunication_Destroy,
	UKSocketCommunication_Connect,
	UKSocketCommunication_Disconnect,
	UKSocketCommunication_Listen,
	UKSocketCommunication_Accept,
	UKSocketCommunication_Send,
	UKSocketCommunication_Receive,
};
</#if>

<#if used_communication_list?seq_contains("bluetooth")>
SVirtualCommunicationAPI g_stBluetoothCommunication = {
	UKBluetoothCommunication_Create,
	UKSocketCommunication_Destroy,
	UKSocketCommunication_Connect,
	UKSocketCommunication_Disconnect,
	UKSocketCommunication_Listen,
	UKSocketCommunication_Accept,
	UKSocketCommunication_Send,
	UKSocketCommunication_Receive,
};

</#if>
<#if used_communication_list?seq_contains("serial")>
SVirtualCommunicationAPI g_stSerialCommunication = {
	UKSerialCommunication_Create,
	UKSerialCommunication_Destroy,
	UKSerialCommunication_Connect,
	UKSerialCommunication_Disconnect,
	UKSerialCommunication_Listen,
	UKSerialCommunication_Accept,
	UKSerialCommunication_Send,
	UKSerialCommunication_Receive,
};

</#if>
<#if used_communication_list?seq_contains("secure_tcp")>
SVirtualCommunicationAPI g_stSecureTCPCommunication = {
	UKSecureTCPCommunication_Create,
	UKSecureTCPCommunication_Destroy,
	UKSecureTCPCommunication_Connect,
	UKSecureTCPCommunication_Disconnect,
	UKSecureTCPCommunication_Listen,	
	UKSecureTCPCommunication_Accept,
	UKSecureTCPCommunication_Send,
	UKSecureTCPCommunication_Receive,
};

SSecurityKeyInfo g_astSSLKeyInfoList[] = {
	<#list ssl_key_info_list as key_info>
	{
		<#if key_info.caPublicKey?has_content == false>
		NULL,
		<#else>
		"${key_info.caPublicKey}",
		</#if>		
		<#if key_info.publicKey?has_content == false>
		NULL,
		<#else>
		"${key_info.publicKey}",
		</#if>		
		<#if key_info.privateKey?has_content == false>
		NULL,
		<#else>
		"${key_info.privateKey}",
		</#if>
	},
	</#list>
};
</#if>

<#if encryption_used == true>
<#if used_encryption_list?seq_contains("lea")>
SVirtualEncryptionAPI g_stEncryptionLEA = {
	UKEncryptionLEA_Initialize,
	UKEncryptionLEA_EncodeOnCTRMode,
	UKEncryptionLEA_EncodeOnCTRMode,
	UKEncryptionLEA_Finalize
};
</#if>

<#if used_encryption_list?seq_contains("hight")>
SVirtualEncryptionAPI g_stEncryptionHIGHT = {
	UKEncryptionHIGHT_Initialize,
	UKEncryptionHIGHT_EncodeOnCTRMode,
	UKEncryptionHIGHT_EncodeOnCTRMode,
	UKEncryptionHIGHT_Finalize
};
</#if>

<#if used_encryption_list?seq_contains("seed")>
SVirtualEncryptionAPI g_stEncryptionSEED = {
	UKEncryptionSEED_Initialize,
	UKEncryptionSEED_EncodeOnCTRMode,
	UKEncryptionSEED_EncodeOnCTRMode,
	UKEncryptionSEED_Finalize
};
</#if>

SEncryptionKeyInfo g_astEncryptionKeyInfoList[] = {
	<#list encryption_list as encryption>
	{
		(unsigned char*)"${encryption.userKey}", // User Key
		(unsigned char*)"${encryption.initializationVector}", // Initialization vector
		${encryption.userKeyLen}, // User Key len
		&(g_stEncryption${encryption.encryptionType}),
	},
	</#list>
};
</#if>

<#if used_communication_list?seq_contains("tcp")>
// #TCP_COMMUNICATION_GENERATION_TEMPLATE::START
#ifndef AGGREGATE_TCP_CONNECTION
// ##TCP_CLIENT_GENERATION_TEMPLATE::START
STCPInfo g_astTCPClientInfo[] = {
	<#list tcp_client_list as client>
	{
		${client.port?c},
		"${client.IP}",
		PAIR_TYPE_CLIENT,
	},
	</#list>
	<#if platform == "windows" && (tcp_client_list?size == 0)>
	0, 
	</#if>
};
// ##TCP_CLIENT_GENERATION_TEMPLATE::END

// ##TCP_SERVER_GENERATION_TEMPLATE::START
STCPServerInfo g_astTCPServerInfo[] = {
	<#list tcp_server_list as server>
	{
		{
			${server.port?c},
			(char *) NULL,
			PAIR_TYPE_SERVER,
		},
		{
			(HVirtualSocket) NULL,
			(HThread) NULL,
			&g_stTCPCommunication,
			<#if encryption_used == true>
			<#if server.usedEncryption == true>
			&(g_astEncryptionKeyInfoList[${server.encryptionListIndex}])
			<#else>
			(SEncryptionKeyInfo *) NULL
			</#if>
			</#if>
		},		
	},
	</#list>
	<#if platform == "windows" && (tcp_server_list?size == 0)>
	0, 
	</#if>
};
// ##TCP_SERVER_GENERATION_TEMPLATE::END

#else
STCPAggregatedServiceInfo g_astTCPAggregateClientInfo[] = {
	<#list tcp_client_list as client>
	{
		{
			${client.port?c},
			"${client.IP}",
			PAIR_TYPE_CLIENT,
		},
		{
			(HThread) NULL, // thread handle
			${client.channelAccessNum}, // max channel access number
			(HVirtualSocket) NULL, // socket handle
			&g_stTCPCommunication, // bluetooth communication API
			(HSerialCommunicationManager) NULL, // Serial communication manager handle
			FALSE, // initialized or not
			<#if encryption_used == true>
			<#if client.usedEncryption == true>
			&(g_astEncryptionKeyInfoList[${client.encryptionListIndex}])
			<#else>
			(SEncryptionKeyInfo *) NULL
			</#if>
			</#if>
		},
	},
	</#list>
	<#if platform == "windows" && (tcp_client_list?size == 0)>
	0, 
	</#if>
};

STCPAggregatedServiceInfo g_astTCPAggregateServerInfo[] = {
	<#list tcp_server_list as server>
	{
		{
			${server.port?c},
			(char *) NULL,
			PAIR_TYPE_SERVER,
		},
		{
			(HThread) NULL, // thread handle
			${server.channelAccessNum}, // max channel access number
			(HVirtualSocket) NULL, // socket handle
			&g_stTCPCommunication, // bluetooth communication API
			(HSerialCommunicationManager) NULL, // Serial communication manager handle
			FALSE, // initialized or not
			<#if encryption_used == true>
			<#if server.usedEncryption == true>
			&(g_astEncryptionKeyInfoList[${server.encryptionListIndex}])
			<#else>
			(SEncryptionKeyInfo *) NULL
			</#if>
			</#if>
		},
	},
	</#list>
	<#if platform == "windows" && (tcp_server_list?size == 0)>
	0, 
	</#if>
};
#endif
// ##TCP_COMMUNICATION_GENERATION_TEMPLATE::END
</#if>

<#if used_communication_list?seq_contains("secure_tcp")>
// #SECURE_TCP_COMMUNICATION_GENERATION_TEMPLATE::START
#ifndef AGGREGATE_SECURETCP_CONNECTION
// ##SECURE_TCP_CLIENT_GENERATION_TEMPLATE::START
SSecureTCPInfo g_astSecureTCPClientInfo[] = {
	<#list secure_tcp_client_list as client>
	{
		{
			${client.port?c},
			"${client.IP}",
			PAIR_TYPE_CLIENT,
		},
		&g_astSSLKeyInfoList[${client.keyInfoIndex}],
	},
	</#list>
	<#if platform == "windows" && (secure_tcp_client_list?size == 0)>
	0, 
	</#if>
};
// ##SECURE_TCP_CLIENT_GENERATION_TEMPLATE::END

// ##SECURE_TCP_SERVER_GENERATION_TEMPLATE::START
SSecureTCPServerInfo g_astSecureTCPServerInfo[] = {
	<#list secure_tcp_server_list as server>
	{
		{
			{
				${server.port?c},
				(char *) NULL,
				PAIR_TYPE_SERVER,
			},
			&g_astSSLKeyInfoList[${server.keyInfoIndex}],
		},
		{
			(HVirtualSocket) NULL,
			(HThread) NULL,
			&g_stSecureTCPCommunication,
		},		
	},
	</#list>
	<#if platform == "windows" && (secure_tcp_server_list?size == 0)>
	0, 
	</#if>
};
// ##SECURE_TCP_SERVER_GENERATION_TEMPLATE::END

#else
SSecureTCPAggregatedServiceInfo g_astSecureTCPAggregateClientInfo[] = {
	<#list secure_tcp_client_list as client>
	{
		{
			{
				${client.port?c},
				"${client.IP}",
				PAIR_TYPE_CLIENT,
			},
			&g_astSSLKeyInfoList[${client.keyInfoIndex}],
		},
		{
			(HThread) NULL, // thread handle
			${client.channelAccessNum}, // max channel access number
			(HVirtualSocket) NULL, // socket handle
			&g_stSecureTCPCommunication, // bluetooth communication API
			(HSerialCommunicationManager) NULL, // Serial communication manager handle
			FALSE, // initialized or not
			<#if encryption_used == true>
			<#if client.usedEncryption == true>
			&(g_astEncryptionKeyInfoList[${client.encryptionListIndex}])
			<#else>
			(SEncryptionKeyInfo *) NULL
			</#if>
			</#if>
		},
	},
	</#list>
	<#if platform == "windows" && (secure_tcp_client_list?size == 0)>
	0, 
	</#if>
};

SSecureTCPAggregatedServiceInfo g_astSecureTCPAggregateServerInfo[] = {
	<#list secure_tcp_server_list as server>
	{
		{
			{
				${server.port?c},
				(char *) NULL,
				PAIR_TYPE_SERVER,
			},
			&g_astSSLKeyInfoList[${server.keyInfoIndex}],
		},
		{
			(HThread) NULL, // thread handle
			${server.channelAccessNum}, // max channel access number
			(HVirtualSocket) NULL, // socket handle
			&g_stSecureTCPCommunication, // bluetooth communication API
			(HSerialCommunicationManager) NULL, // Serial communication manager handle
			FALSE, // initialized or not
			<#if encryption_used == true>
			<#if server.usedEncryption == true>
			&(g_astEncryptionKeyInfoList[${server.encryptionListIndex}])
			<#else>
			(SEncryptionKeyInfo *) NULL
			</#if>
			</#if>
		},
	},
	</#list>
	<#if platform == "windows" && (secure_tcp_server_list?size == 0)>
	0, 
	</#if>
};
#endif
// ##SECURE_TCP_COMMUNICATION_GENERATION_TEMPLATE::END
</#if>

<#if communication_used == true>
// ##INDIVIDUAL_CONNECTION_GENERATION_TEMPLATE::START
SIndividualConnectionInfo g_astIndividualConnectionInfo[] = {
	<#list channel_list as channel>
		<#switch channel.remoteMethodType>
			<#case "TCP">
			<#case "SecureTCP">
#ifndef AGGREGATE_${channel.remoteMethodType?upper_case}_CONNECTION
	{
		${channel.index},
		COMMUNICATION_METHOD_${channel.remoteMethodType?upper_case},
		<#switch channel.connectionRoleType>
			<#case "CLIENT">
		(S${channel.remoteMethodType}Info *) &g_ast${channel.remoteMethodType}ClientInfo[${channel.socketInfoIndex}],
		PAIR_TYPE_CLIENT,
			<#break>
			<#case "SERVER">
		(void *) NULL,
		PAIR_TYPE_SERVER,
			<#break>
		</#switch>
		&g_st${channel.remoteMethodType}Communication,
		(HVirtualSocket) NULL,
		(HUEMProtocol) NULL,
		<#if encryption_used == true>
		<#if channel.usedEncryption == true>
		&(g_astEncryptionKeyInfoList[${channel.encryptionListIndex}])
		<#else>
		(SEncryptionKeyInfo *) NULL
		</#if>
		</#if>		
	},
#endif
			<#break>
		</#switch>
	</#list>
};
// ##INDIVIDUAL_CONNECTION_GENERATION_TEMPLATE::END
</#if>

<#if used_communication_list?seq_contains("bluetooth")>
// ##BLUETOOTH_COMMUNICATION_GENERATION_TEMPLATE::START
SBluetoothInfo g_astBluetoothMasterInfo[] = {
	<#list bluetooth_master_list as master>
	{
		{
			(HThread) NULL, // thread handle
			${master.channelAccessNum}, // max channel access number
			(HVirtualSocket) NULL, // socket handle
			&g_stBluetoothCommunication, // bluetooth communication API
			(HSerialCommunicationManager) NULL, // Serial communication manager handle
			FALSE, // initialized or not
			<#if encryption_used == true>
			<#if master.usedEncryption == true>
			&(g_astEncryptionKeyInfoList[${master.encryptionListIndex}])
			<#else>
			(SEncryptionKeyInfo *) NULL
			</#if>
			</#if>
		},
		{
			"${master.portAddress}", // target mac address
			PAIR_TYPE_MASTER,
		},
	},
	</#list>
};

SBluetoothInfo g_astBluetoothSlaveInfo[] = {
	<#list bluetooth_slave_list as slave>
	{
		{
			(HThread) NULL, // thread handle
			${slave.channelAccessNum}, // max channel access number
			(HVirtualSocket) NULL, // socket handle
			&g_stBluetoothCommunication, // bluetooth communication API
			(HSerialCommunicationManager) NULL, // Serial communication manager handle
			FALSE, // initialized or not
			<#if encryption_used == true>
			<#if slave.usedEncryption == true>
			&(g_astEncryptionKeyInfoList[${slave.encryptionListIndex}])
			<#else>
			(SEncryptionKeyInfo *) NULL
			</#if>
			</#if>
		},
		{
			"${slave.portAddress}", // slave mac address
			PAIR_TYPE_SLAVE,
		},
	},
	</#list>
};
// ##BLUETOOTH_COMMUNICATION_GENERATION_TEMPLATE::END
</#if>

<#if used_communication_list?seq_contains("serial")>
// ##SERIAL_COMMUNICATION_GENERATION_TEMPLATE::START
SSerialInfo g_astSerialMasterInfo[] = {
	<#list serial_master_list as master>
	{
		{
			(HThread) NULL, // thread handle
			${master.channelAccessNum}, // max channel access number
			(HVirtualSocket) NULL, // socket handle
			&g_stSerialCommunication, // bluetooth communication API
			(HSerialCommunicationManager) NULL, // Serial communication manager handle
			FALSE, // initialized or not
			<#if encryption_used == true>
			<#if master.usedEncryption == true>
			&(g_astEncryptionKeyInfoList[${master.encryptionListIndex}])
			<#else>
			(SEncryptionKeyInfo *) NULL
			</#if>
			</#if>
		},
		{
			"${master.portAddress}", // serial port path
			PAIR_TYPE_MASTER,
		},
	},
	</#list>
};


SSerialInfo g_astSerialSlaveInfo[] = {
	<#list serial_slave_list as slave>
	{
		{
			(HThread) NULL, // thread handle
			${slave.channelAccessNum}, // max channel access number
			(HVirtualSocket) NULL, // socket handle
			&g_stSerialCommunication, // bluetooth communication API
			(HSerialCommunicationManager) NULL, // Serial communication manager handle
			FALSE, // initialized or not
			<#if encryption_used == true>
			<#if slave.usedEncryption == true>
			&(g_astEncryptionKeyInfoList[${slave.encryptionListIndex}])
			<#else>
			(SEncryptionKeyInfo *) NULL
			</#if>
			</#if>
		},
		{
			"${slave.portAddress}", // serial port path
			PAIR_TYPE_SLAVE,
		},
	},
	</#list>
};
// ##SERIAL_COMMUNICATION_GENERATION_TEMPLATE::END
</#if>


<#if communication_used == true>
SAggregateConnectionInfo g_astAggregateConnectionInfo[] = {
	<#list channel_list as channel>
		<#switch channel.remoteMethodType>
			<#case "BLUETOOTH">
	{
		${channel.index},
		{
			(HFixedSizeQueue) NULL,
		},
				<#switch channel.connectionRoleType>
					<#case "MASTER">
		&(g_astBluetoothMasterInfo[${channel.socketInfoIndex}].stAggregateInfo),
						<#break>
					<#case "SLAVE">
		&(g_astBluetoothSlaveInfo[${channel.socketInfoIndex}].stAggregateInfo),
						<#break>
				</#switch>
	},
					<#break>
			<#case "SERIAL">
	{
		${channel.index},
		{
			(HFixedSizeQueue) NULL,
		},			
				<#switch channel.connectionRoleType>
					<#case "MASTER">
		&(g_astSerialMasterInfo[${channel.socketInfoIndex}].stAggregateInfo),
						<#break>
					<#case "SLAVE">
		&(g_astSerialSlaveInfo[${channel.socketInfoIndex}].stAggregateInfo),
						<#break>
				</#switch>
	},
				<#break>
			<#case "TCP">
#ifdef AGGREGATE_TCP_CONNECTION
	{
		${channel.index},
		{
			(HFixedSizeQueue) NULL,
		},
				<#switch channel.connectionRoleType>
					<#case "CLIENT">
		&(g_astTCPAggregateClientInfo[${channel.socketInfoIndex}].stServiceInfo),
						<#break>
					<#case "SERVER">
		&(g_astTCPAggregateServerInfo[${channel.socketInfoIndex}].stServiceInfo),
						<#break>
				</#switch>	
	},
#else
	<#if platform == "windows">
	0,
	</#if>
#endif				
				<#break>
			<#case "SecureTCP">
#ifdef AGGREGATE_SECURETCP_CONNECTION
	{
		${channel.index},
		{
			(HFixedSizeQueue) NULL,
		},
				<#switch channel.connectionRoleType>
					<#case "CLIENT">
		&(g_astSecureTCPAggregateClientInfo[${channel.socketInfoIndex}].stServiceInfo),
						<#break>
					<#case "SERVER">
		&(g_astSecureTCPAggregateServerInfo[${channel.socketInfoIndex}].stServiceInfo),
						<#break>
				</#switch>	
	},
#else
	<#if platform == "windows">
	0,
	</#if>
#endif				
				<#break>
		</#switch>
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


<#macro printRemoteChannelCommonInfo channel>
	{ 
			<#switch channel.remoteMethodType>
				<#case "TCP">
				<#case "SecureTCP">
#ifndef AGGREGATE_${channel.remoteMethodType?upper_case}_CONNECTION
		CONNECTION_METHOD_INDIVIDUAL,
#else
		CONNECTION_METHOD_AGGREGATE,
#endif
					<#break>
				<#case "BLUETOOTH">
				<#case "SERIAL">
		CONNECTION_METHOD_AGGREGATE,
					<#break>
			</#switch>
		NULL, // will be set to SIndividualServiceInfo or SAggregateServiceInfo
		(HThreadMutex) NULL,
		FALSE, // bChannelExit
	},
</#macro>

// ##SPECIFIC_CHANNEL_LIST_TEMPLATE::START
<#list channel_list as channel>

	<#switch channel.communicationType>
		<#case "SHARED_MEMORY">
		<#case "REMOTE_WRITER">
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
		<#case "REMOTE_WRITER">
SRemoteWriterChannel g_stRemoteWriterChannel_${channel.index} = {
			<@printRemoteChannelCommonInfo channel />
	(HThread) NULL, // receive handling thread
	(char *) NULL,
	0,
	&g_stSharedMemoryChannel_${channel.index}, // SSharedMemoryChannel *pstInternalChannel;
};
			<#break>
		<#case "REMOTE_READER">
SRemoteReaderChannel g_stRemoteReaderChannel_${channel.index} = {
			<@printRemoteChannelCommonInfo channel />
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
		<#case "REMOTE_WRITER">
		&g_stRemoteWriterChannel_${channel.index}, // specific TCP socket channel structure pointer
			<#break>
		<#case "REMOTE_READER">
		&g_stRemoteReaderChannel_${channel.index}, // specific bluetooth/serial channel structure pointer
			<#break>
	</#switch>
	},
</#list>
};
// ##CHANNEL_LIST_TEMPLATE::END

<#if communication_used == true>
SChannelAPI g_stRemoteWriterChannel = {
	UKRemoteChannel_Initialize,
	(FnChannelReadFromQueue) NULL,
	(FnChannelReadFromBuffer) NULL,
	UKRemoteChannel_WriteToQueue,
	UKRemoteChannel_WriteToBuffer,
	(FnChannelGetAvailableChunk) NULL,
	UKRemoteChannel_GetNumOfAvailableData,
	UKRemoteChannel_Clear,
	UKRemoteChannel_SetExit,
	UKRemoteChannel_ClearExit,
	UKRemoteChannel_FillInitialData,
	UKRemoteChannel_Finalize,
	UKRemoteChannel_APIInitialize, 
	UKRemoteChannel_APIFinalize,
};


SChannelAPI g_stRemoteReaderChannel = {
	UKRemoteChannel_Initialize,
	UKRemoteChannel_ReadFromQueue,
	UKRemoteChannel_ReadFromBuffer,
	(FnChannelWriteToQueue) NULL,
	(FnChannelWriteToBuffer) NULL,
	UKRemoteChannel_GetAvailableChunk,
	UKRemoteChannel_GetNumOfAvailableData,
	UKRemoteChannel_Clear,
	UKRemoteChannel_SetExit,
	UKRemoteChannel_ClearExit,
	(FnChannelFillInitialData) NULL,
	UKRemoteChannel_Finalize,
	(FnChannelAPIInitialize) NULL, 
	(FnChannelAPIFinalize) NULL,
};
</#if>

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


SChannelAPI *g_astChannelAPIList[] = {
	&g_stSharedMemoryChannel,
<#if communication_used == true>
	&g_stRemoteReaderChannel,
	&g_stRemoteWriterChannel,
</#if>
};



<#if communication_used == true>
FnChannelAPIInitialize g_aFnRemoteCommunicationModuleIntializeList[] = {
	UCSocket_Initialize, 
	<#if used_communication_list?seq_contains("tcp")>
	UKTCPServerManager_Initialize,
	</#if>
	<#if used_communication_list?seq_contains("bluetooth")>
	UKBluetoothModule_Initialize,
	</#if>
	<#if used_communication_list?seq_contains("serial")>
	UKSerialModule_Initialize,
	</#if>
	<#if used_communication_list?seq_contains("secure_tcp")>
	UKSecureTCPServerManager_Initialize,
	</#if>
};

FnChannelAPIFinalize g_aFnRemoteCommunicationModuleFinalizeList[] = {
	<#if used_communication_list?seq_contains("tcp")>
	UKTCPServerManager_Finalize,
	</#if>
	<#if used_communication_list?seq_contains("bluetooth")>
	UKBluetoothModule_Finalize,
	</#if>
	<#if used_communication_list?seq_contains("serial")>
	UKSerialModule_Finalize,
	</#if>
	<#if used_communication_list?seq_contains("secure_tcp")>
	UKSecureTCPServerManager_Finalize,
	</#if>
	UCSocket_Finalize, 
};
</#if>

<#if used_communication_list?seq_contains("bluetooth")>
SSocketAPI stBluetoothAPI = {
	UCBluetoothSocket_Bind,
	UCBluetoothSocket_Accept,
	UCBluetoothSocket_Connect,
	(FnSocketCreate) NULL,
	(FnSocketDestroy) NULL,
};
</#if>


<#if used_communication_list?seq_contains("tcp") || used_communication_list?seq_contains("secure_tcp")>
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
<#if used_communication_list?seq_contains("tcp") || used_communication_list?seq_contains("secure_tcp")>
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
	case COMMUNICATION_TYPE_REMOTE_READER:
<#if communication_used == true>
		*ppstChannelAPI = &g_stRemoteReaderChannel;
<#else>
		ERRASSIGNGOTO(result, ERR_UEM_NOT_SUPPORTED, _EXIT)
</#if>
		break;
	case COMMUNICATION_TYPE_REMOTE_WRITER:
<#if communication_used == true>
		*ppstChannelAPI = &g_stRemoteWriterChannel;
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
#ifndef AGGREGATE_TCP_CONNECTION
	<#if used_communication_list?seq_contains("tcp")>
int g_nIndividualConnectionInfoNum = ARRAYLEN(g_astIndividualConnectionInfo);
	<#else>
int g_nIndividualConnectionInfoNum = 0;
	</#if>
#else 
int g_nIndividualConnectionInfoNum = 0;
#endif

#ifdef AGGREGATE_TCP_CONNECTION
int g_nAggregateConnectionInfoNum = ARRAYLEN(g_astAggregateConnectionInfo);
#else
	<#if used_communication_list?seq_contains("bluetooth") || used_communication_list?seq_contains("serial")>
int g_nAggregateConnectionInfoNum = ARRAYLEN(g_astAggregateConnectionInfo);
	<#else>
int g_nAggregateConnectionInfoNum = 0;
	</#if>
#endif
	

#ifndef AGGREGATE_TCP_CONNECTION
	<#if (tcp_server_list?size > 0) >
int g_nTCPServerInfoNum = ARRAYLEN(g_astTCPServerInfo);
	<#else>
int g_nTCPServerInfoNum = 0;
	</#if>
#else
	<#if (tcp_server_list?size > 0) >
int g_nTCPAggregateServerInfoNum = ARRAYLEN(g_astTCPAggregateServerInfo);
	<#else>
int g_nTCPAggregateServerInfoNum = 0;
	</#if>
	
	<#if (tcp_client_list?size > 0) >
int g_nTCPAggregateClientInfoNum = ARRAYLEN(g_astTCPAggregateClientInfo);
	<#else>
int g_nTCPAggregateClientInfoNum = 0;
	</#if>
#endif
	
#ifndef AGGREGATE_SECURETCP_CONNECTION
	<#if (secure_tcp_server_list?size > 0) >
int g_nSecureTCPServerInfoNum = ARRAYLEN(g_astSecureTCPServerInfo);
	<#else>
int g_nSecureTCPServerInfoNum = 0;
	</#if>
#else
	<#if (secure_tcp_server_list?size > 0) >
int g_nSecureTCPAggregateServerInfoNum = ARRAYLEN(g_astSecureTCPAggregateServerInfo);
	<#else>
int g_nSecureTCPAggregateServerInfoNum = 0;
	</#if>
	
	<#if (secure_tcp_client_list?size > 0) >
int g_nSecureTCPAggregateClientInfoNum = ARRAYLEN(g_astSecureTCPAggregateClientInfo);
	<#else>
int g_nSecureTCPAggregateClientInfoNum = 0;
	</#if>
#endif
	
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

int g_nRemoteCommunicationModuleNum = ARRAYLEN(g_aFnRemoteCommunicationModuleIntializeList);

</#if>


