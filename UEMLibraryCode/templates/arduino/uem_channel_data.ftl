/* uem_channel_data.c made by UEM Translator */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <uem_enum.h>

#include <uem_channel_data.h>

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
		PORT_SAMPLE_RATE_${port.portSampleRateType}, // Port sample rate type
		${port.portSampleRateList[0].sampleRate?c}, // Port sample rate (for fixed type)
		${port.sampleSize?c}, // Sample size
		PORT_TYPE_${port.portType}, // Port type
		<#if port.subgraphPort??>&g_astPortInfo[${port_key_to_index[port.subgraphPort.portKey]}]<#else>(SPort *) NULL</#if>, // Pointer to Subgraph port
	}, // Port information
</#list>
};
// ##PORT_ARRAY_TEMPLATE::END


// ##SPECIFIC_CHANNEL_LIST_TEMPLATE::START
<#list channel_list as channel>

	<#switch channel.communicationType>
		<#case "SHARED_MEMORY">
SSharedMemoryChannel g_stSharedMemoryChannel_${channel.index} = {
	s_pChannel_${channel.index}_buffer, // Channel buffer pointer
	s_pChannel_${channel.index}_buffer, // Channel data start
	s_pChannel_${channel.index}_buffer, // Channel data end
	0, // Channel data length
	<#if (channel.initialDataLen > 0)>FALSE<#else>TRUE</#if>, // initial data is updated
};

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
			${channel.inputPort.portSampleRateList[0].sampleRate?c}, // Port sample rate (for fixed type)
			${channel.inputPort.sampleSize?c}, // Sample size
			PORT_TYPE_${channel.inputPort.portType}, // Port type
			<#if channel.inputPort.subgraphPort??>&g_astPortInfo[${port_key_to_index[channel.inputPort.subgraphPort.portKey]}]<#else>(SPort *) NULL</#if>, // Pointer to Subgraph port
		}, // Input port information
		{
			${channel.outputPort.taskId}, // Task ID
			"${channel.outputPort.portName}", // Port name
			PORT_SAMPLE_RATE_${channel.outputPort.portSampleRateType}, // Port sample rate type
			${channel.outputPort.portSampleRateList[0].sampleRate?c}, // Port sample rate (for fixed type)
			${channel.outputPort.sampleSize?c}, // Sample size
			PORT_TYPE_${channel.outputPort.portType}, // Port type
			<#if channel.outputPort.subgraphPort??>&g_astPortInfo[${port_key_to_index[channel.outputPort.subgraphPort.portKey]}]<#else>(SPort *) NULL</#if>, // Pointer to Subgraph port
		}, // Output port information
		${channel.initialDataLen?c}, // Initial data length
	<#switch channel.communicationType>
		<#case "SHARED_MEMORY">
		&g_stSharedMemoryChannel_${channel.index}, // specific shared memory channel structure pointer
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
	NULL,
	NULL,
};

SChannelAPI *g_astChannelAPIList[] = {
		&g_stSharedMemoryChannel,
};



uem_result ChannelAPI_GetAPIStructureFromCommunicationType(IN ECommunicationType enType, OUT SChannelAPI **ppstChannelAPI)
{
	uem_result result = ERR_UEM_UNKNOWN;
	switch(enType)
	{
	case COMMUNICATION_TYPE_SHARED_MEMORY:
		*ppstChannelAPI = &g_stSharedMemoryChannel;
		break;
	default:
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT)
		break;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

int g_nChannelAPINum = ARRAYLEN(g_astChannelAPIList);
int g_nChannelNum = ARRAYLEN(g_astChannels);


