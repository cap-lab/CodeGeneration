/* uem_channel_data.c made by UEM Translator */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif


<#--  default board_tag name to avoid error from unassigned board_tag value -->
<#assign board_tag= "uno">

<#list env_var_info as envVar>
  <#if envVar.name = "BOARD_TAG">
    <#assign board_tag= envVar.value>
    <#break>
  </#if>
</#list>

<#if communication_used == true && board_tag != "OpenCR" >
#include <Arduino.h>
#include <SoftwareSerial.h>
</#if>

<#if communication_used == true && board_tag == "OpenCR" >
#include <Arduino.h>
#include <USBSerial.h>
#include <HardwareSerial.h>
</#if>

#include <uem_common.h>

<#if communication_used == true>
#include <UCSerial.h>
#include <UCSerial_data.hpp>
</#if>

#include <uem_enum.h>

#include <uem_channel_data.h>
<#if communication_used == true>
#include <uem_bluetooth_data.h>
#include <UKSerialChannel.h>
#include <UKSerialModule.h>
#include <UKVirtualEncryption.h>
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

#include <UKAddOnHandler.h>
#include <UKSharedMemoryChannel.h>

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


// ##PORT_ARRAY_TEMPLATE::START
SPort g_astPortInfo[] = {
<#list port_info as port>
	{
		${port.taskId}, // Task ID
		"${port.portName}", // Port name
		<#if port.subgraphPort??>&g_astPortInfo[${port_key_to_index[port.subgraphPort.portKey]}]<#else>(SPort *) NULL</#if>, // Pointer to Subgraph port
	}, // Port information
</#list>
};
// ##PORT_ARRAY_TEMPLATE::END


// ##SOFTWARESERIAL_GENERATION_TEMPLATE::START
<#if communication_used == true && board_tag != "OpenCR" >

<#list serial_slave_list as slave>
<#switch slave.connectionType>
	<#case "BLUETOOTH">
	<#case "WIRE">
static SoftwareSerial s_clsSerial_${slave.name}(${slave.boardRXPinNumber}, ${slave.boardTXPinNumber});

SSerialHandle g_stSerial_${slave.name}  = { &s_clsSerial_${slave.name}, SoftwareSerial_Initialize };
		<#break>
	<#case "USB">
SSerialHandle g_stSerial_${slave.name}  = { &Serial, HardwareSerial_Initialize };	
		<#break>
</#switch>
</#list>

<#list serial_master_list as master>
<#switch master.connectionType>
	<#case "BLUETOOTH">
	<#case "WIRE">
static SoftwareSerial s_clsSerial_${master.name}(${master.boardRXPinNumber}, ${master.boardTXPinNumber});

SSerialHandle g_stSerial_${master.name}  = { &s_clsSerial_${master.name}, SoftwareSerial_Initialize };
		<#break>
	<#case "USB">
SSerialHandle g_stSerial_${master.name}  = { &Serial, HardwareSerial_Initialize };
		<#break>
</#switch>
</#list>

<#list serial_slave_list as slave>
SChannel *g_pastAccessChannel_${slave.name}[${slave.channelAccessNum}];
</#list>
	
<#list serial_master_list as master>
SChannel *g_pastAccessChannel_${master.name}[${master.channelAccessNum}];
</#list>
	
<#if encryption_used == true>
<#if used_encryption_list?seq_contains("lea")>
SVirtualEncryptionAPI g_stEncryptionLEA = {
	UKEncryptionLEA_Initialize,
	UKEncryptionLEA_EncodeOnCTRMode,
	UKEncryptionLEA_EncodeOnCTRMode,
	//NULL
};
</#if>

<#if used_encryption_list?seq_contains("hight")>
SVirtualEncryptionAPI g_stEncryptionHIGHT = {
	UKEncryptionHIGHT_Initialize,
	UKEncryptionHIGHT_EncodeOnCTRMode,
	UKEncryptionHIGHT_EncodeOnCTRMode,
	//NULL
};
</#if>

<#if used_encryption_list?seq_contains("seed")>
SVirtualEncryptionAPI g_stEncryptionSEED = {
	UKEncryptionSEED_Initialize,
	UKEncryptionSEED_EncodeOnCTRMode,
	UKEncryptionSEED_EncodeOnCTRMode,
	//NULL
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

SSerialInfo g_astSerialSlaveInfo[] = {
	<#list serial_slave_list as slave>
	{
		&g_stSerial_${slave.name},
		${slave.channelAccessNum},
		0,
		g_pastAccessChannel_${slave.name},
		<#if encryption_used == true>
		<#if slave.usedEncryption == true>
		&(g_astEncryptionKeyInfoList[${slave.encryptionListIndex}])
		<#else>
		(SEncryptionKeyInfo *) NULL
		</#if>
		</#if>
	},
	</#list>
};

SSerialInfo g_astSerialMasterInfo[] = {
	<#list serial_master_list as master>
	{
		&g_stSerial_${master.name},
		${master.channelAccessNum},
		0,
		g_pastAccessChannel_${master.name},
		<#if encryption_used == true>
		<#if master.usedEncryption == true>
		&(g_astEncryptionKeyInfoList[${master.encryptionListIndex}])
		<#else>
		(SEncryptionKeyInfo *) NULL
		</#if>
		</#if>		
	},
	</#list>
};
</#if>
// ##SOFTWARESERIAL_GENERATION_TEMPLATE::END

//OPENCRSERIAL_GENERATION_TEMPLATE::START
<#if communication_used == true && board_tag == "OpenCR" >
	<#list serial_slave_list as slave> //suppose only one Serial used.
SSerialHandle g_stSerial_${slave.name}  = { &Serial, USBSerial_Initialize };

SChannel *g_pastAccessChannel_${slave.name}[${slave.channelAccessNum}];
	</#list>
	
SSerialInfo g_astSerialSlaveInfo[] = {
	<#list serial_slave_list as slave>
	{
		&g_stSerial_${slave.name},
		${slave.channelAccessNum},
		0,
		g_pastAccessChannel_${slave.name},
		<#if encryption_used == true>
		<#if slave.usedEncryption == true>
		&(g_astEncryptionKeyInfoList[${slave.encryptionListIndex}])
		<#else>
		(SEncryptionKeyInfo *) NULL
		</#if>
		</#if>		
	},
	</#list>
};
</#if>
//OPENCRSERIAL_GENERATION_TEMPLATE::END

// ##SPECIFIC_CHANNEL_LIST_TEMPLATE::START
<#list channel_list as channel>
SSharedMemoryChannel g_stSharedMemoryChannel_${channel.index} = {
	s_pChannel_${channel.index}_buffer, // Channel buffer pointer
	s_pChannel_${channel.index}_buffer, // Channel data start
	s_pChannel_${channel.index}_buffer, // Channel data end
	0, // Channel data length
	<#if (channel.initialDataLen > 0)>FALSE<#else>TRUE</#if>, // initial data is updated
};

	<#switch channel.communicationType>
		<#case "REMOTE_WRITER">
		<#case "REMOTE_READER">	
			<#switch channel.remoteMethodType>
				<#case "BLUETOOTH">
				<#case "SERIAL">
SSerialChannel g_stSerialChannel_${channel.index} = {
					<#switch channel.connectionRoleType>
						<#case "MASTER">
	&g_astSerialMasterInfo[${channel.socketInfoIndex}],
							<#break>
						<#case "SLAVE">
	&g_astSerialSlaveInfo[${channel.socketInfoIndex}],
							<#break>
				 	</#switch>
	{
		MESSAGE_TYPE_NONE,
		0,
	},
	&g_stSharedMemoryChannel_${channel.index},
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
		//CHANNEL_TYPE_${channel.channelType}, // Channel type (not used in constrained device)
		CHANNEL_${channel.index}_SIZE, // Channel size
		&(g_astPortInfo[${channel.inputPortIndex}]), // Outer-most input port information
		&(g_astPortInfo[${channel.outputPortIndex}]), // Outer-most output port information
		${channel.initialDataLen?c}, // Initial data length
	<#switch channel.communicationType>
		<#case "SHARED_MEMORY">
		&g_stSharedMemoryChannel_${channel.index}, // specific shared memory channel structure pointer
			<#break>
		<#case "REMOTE_WRITER">
		<#case "REMOTE_READER">
			<#switch channel.remoteMethodType>
				<#case "BLUETOOTH">
				<#case "SERIAL">
		&g_stSerialChannel_${channel.index}, // specific serial channel structure pointer
					<#break>
			</#switch>
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
	(FnChannelAPIInitialize) NULL,
};

<#if communication_used == true>
SChannelAPI g_stSerialChannel = {
	UKSerialChannel_Initialize, // fnInitialize
	UKSerialChannel_ReadFromQueue, // fnReadFromQueue
	UKSerialChannel_ReadFromBuffer, // fnReadFromBuffer
	UKSerialChannel_WriteToQueue, // fnWriteToQueue
	UKSerialChannel_WriteToBuffer, // fnWriteToBuffer
	UKSerialChannel_GetAvailableChunk, // fnGetAvailableChunk
	UKSerialChannel_GetNumOfAvailableData, // fnGetNumOfAvailableData
	UKSerialChannel_Clear, // fnClear
	UKSerialChannel_SetExit,
	UKSerialChannel_ClearExit,
	UKSerialChannel_FillInitialData,
	UKSerialChannel_Finalize, // fnFinalize
	UKSerialModule_Initialize,
	UKSerialModule_Finalize,
};
</#if>

SChannelAPI *g_astChannelAPIList[] = {
		&g_stSharedMemoryChannel,
<#if communication_used == true>
		&g_stSerialChannel,
</#if>
};


SAddOnFunction g_astAddOns[] = {
	<#if communication_used == true>
	{
		(FuncAddOnFunction) NULL,
		UKSerialModule_Run,
		(FuncAddOnFunction) NULL,
	},
	</#if>
};


#ifdef __cplusplus
extern "C"
{
#endif

uem_result ChannelAPI_SetSocketAPIs()
{
	return ERR_UEM_NOERROR;
}


uem_result ChannelAPI_GetAPIStructureFromCommunicationType(IN ECommunicationType enType, OUT SChannelAPI **ppstChannelAPI)
{
	uem_result result = ERR_UEM_UNKNOWN;
	switch(enType)
	{
	case COMMUNICATION_TYPE_SHARED_MEMORY:
		*ppstChannelAPI = &g_stSharedMemoryChannel;
		break;
	// For unconstrained device, only serial communication (serial or bluetooth) is supported for remote communication
	case COMMUNICATION_TYPE_REMOTE_WRITER:
	case COMMUNICATION_TYPE_REMOTE_READER:
<#if communication_used == true>
		*ppstChannelAPI = &g_stSerialChannel;	
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

int g_nChannelAPINum = ARRAYLEN(g_astChannelAPIList);
int g_nChannelNum = ARRAYLEN(g_astChannels);

<#if communication_used == true>
int g_nAddOnNum = ARRAYLEN(g_astAddOns);

	<#if (serial_master_list?size > 0) >
int g_nSerialMasterNum = ARRAYLEN(g_astSerialMasterInfo);
	<#else>
int g_nSerialMasterNum = 0;
	</#if>
	
	<#if (serial_slave_list?size > 0) >
int g_nSerialSlaveNum = ARRAYLEN(g_astSerialSlaveInfo);
	<#else>
int g_nSerialSlaveNum = 0;
	</#if>
<#else>
int g_nAddOnNum = 0;
</#if>

