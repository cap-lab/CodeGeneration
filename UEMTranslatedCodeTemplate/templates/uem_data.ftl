/* uem_data.c made by UEM Translator */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_data.h>

SExecutionTime g_stExecutionTime = { ${execution_time.value}, TIME_METRIC_${execution_time.metric} } ;

// ##TASK_CODE_TEMPLATE::START
<#list flat_task as task_name, task>
	<#if !task.childTaskGraphName??>
		<#list 0..(task.taskFuncNum-1) as task_func_id>
void ${task.name}_Init${task_func_id}(int nTaskId);
void ${task.name}_Go${task_func_id}();
void ${task.name}_Wrapup${task_func_id}();
		</#list>
	</#if>

</#list>
// ##TASK_CODE_TEMPLATE::END

// ##CHANNEL_SIZE_DEFINITION_TEMPLATE::START
<#list channel_list as channel>
#define CHANNEL_${channel.index}_SIZE (${channel.size})
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
<#list 0..(channel.maximumChunkNum-1) as chunk_id>
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
<#list 0..(channel.maximumChunkNum-1) as chunk_id>
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
<#list channel_list as channel>
SPortSampleRate g_astPortSampleRate_${channel.inputPort.taskName}_${channel.inputPort.portName}[] = {
	<#list channel.inputPort.portSampleRateList as sample_rate>
	{ 	"${sample_rate.modeName}", // Mode name
		${sample_rate.sampleRate}, // Sample rate
		${sample_rate.maxAvailableNum}, // Available number of data
	},
	</#list>	
};

SPortSampleRate g_astPortSampleRate_${channel.outputPort.taskName}_${channel.outputPort.portName}[] = {
	<#list channel.outputPort.portSampleRateList as sample_rate>
	{ 	"${sample_rate.modeName}", // Mode name
		${sample_rate.sampleRate}, // Sample rate
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
		${port.sampleSize}, // Sample size
		PORT_TYPE_${port.portType}, // Port type
		<#if port.subgraphPort??>&g_astPortInfo[${port_key_to_index[port.subgraphPort.portKey]}]<#else>NULL</#if>, // Pointer to Subgraph port
	}, // Port information		
		
</#list>
};
// ##PORT_ARRAY_TEMPLATE::END


// ##LOOP_STRUCTURE_TEMPLATE::START
<#list flat_task as task_name, task>
	<#if task.loopStruct??>
SLoopInfo g_stLoopStruct_${task.name} = {
	LOOP_TYPE_${task.loopType},
	${task.loopCount},
	"${task.designatedTaskId}",
};

	</#if>
</#list>
// ##LOOP_STRUCTURE_TEMPLATE::END

// ##VARIABLE_INT_MAP_TEMPLATE::START


// ##VARIABLE_INT_MAP_TEMPLATE::END


// ##MODE_TRANSITION_TEMPLATE::START
<#list flat_task as task_name, task>
	<#if task.modeTransition??>
		<#list task.modeTransition.modeMap as task_name, task_mode>
STask *g_pastRelatedChildTasks_${task.name}_${task_mode.name}[] = {
			<#list task_mode.relatedChildTaskSet as child_task>
	&g_astTasks_${child_task.parentTaskGraphName}[${child_task.inGraphIndex}],
			</#list>
};
		</#list>
		
SModeMap g_astModeMap_${task.name}[] = {
		<#list task.modeTransition.modeMap as mode_name, task_mode>
	{
		${task_mode.id},
		"${task_mode.name}",
		&g_pastRelatedChildTasks_${task.name}_${task_mode.name},
		${task_mode.relatedChildTaskSet?size},
	},
		</#list>
};

SVariableIntMap g_astVariableIntMap_${task.name}[] = {
		<#list task.modeTransition.variableMap as var_name, var_type>
	{
		${var_name?index},
		"${var_name}",
		0, 
	},
		</#list>
}
		

SModeTransitionMachine g_stModeTransition_${task.name} = {
	${task.id},
	&g_astModeMap_${task.name},
	&g_astVariableIntMap_${task.name},
	NULL;
	0,
};
	</#if>
</#list>
// ##MODE_TRANSITION_TEMPLATE::END


// ##AVAILABLE_CHUNK_LIST_TEMPLATE::START
<#list channel_list as channel>
SAvailableChunk g_astAvailableInputChunk_channel_${channel.index}[] = {
<#list 0..(channel.maximumChunkNum-1) as chunk_id>
	{ ${chunk_id}, 0, NULL, NULL, },
</#list>
};
</#list>
// ##AVAILABLE_CHUNK_LIST_TEMPLATE::END

// ##TASK_PARAMETER_TEMPLATE::START
<#list flat_task as task_name, task>
	<#if (task.taskParamList?size > 0)>
STaskParameter g_astTaskParameter_${task.name}[] = {
		<#list task.taskParamList as task_param>
	{
		${task_param.id},
		PARAMETER_TYPE_${task_param.type},
		${task_param.name},
		{ <#if task_param.type == "INT">.nParam = 0,<#else>.dbParam = 0</#if> },
	},
		</#list>
};
	</#if>
</#list>
// ##TASK_PARAMETER_TEMPLATE::END

// ##TASK_FUNCTION_LIST::START
<#list flat_task as task_name, task>
STaskFunctions g_ast_${task.name}_functions[] = {
	<#if !task.childTaskGraphName??>
		<#list 0..(task.taskFuncNum-1) as task_func_id>
	{
		${task.name}_Init${task_func_id}, // Task init function
		${task.name}_Go${task_func_id}, // Task go function
		${task.name}_Wrapup${task_func_id}, // Task wrapup function
	},
		</#list>
	</#if>
};

</#list>
// ##TASK_FUNCTION_LIST::END


// ##CHANNEL_LIST_TEMPLATE::START
SChannel g_astChannels[] = {
<#list channel_list as channel>
	{
		${channel.index}, // Channel ID
		COMMUNICATION_TYPE_${channel.communicationType}, // Channel communication type
		CHANNEL_TYPE_${channel.channelType}, // Channel type
		s_pChannel_${channel.index}_buffer, // Channel buffer pointer
		CHANNEL_${channel.index}_SIZE, // Channel size
		s_pChannel_${channel.index}_buffer, // Channel data start
		s_pChannel_${channel.index}_buffer, // Channel data end
		0, // Channel data length
		0, // Read reference count
		0, // Write reference count
		FALSE, // Exit setting
		NULL, // Mutex
		NULL, // Read available notice event
		NULL, // Write available notice event
		{
			${channel.inputPort.taskId}, // Task ID
			"${channel.inputPort.portName}", // Port name
			PORT_SAMPLE_RATE_${channel.inputPort.portSampleRateType}, // Port sample rate type
			g_astPortSampleRate_${channel.inputPort.taskName}_${channel.inputPort.portName}, // Array of sample rate list
			${channel.inputPort.portSampleRateList?size}, // Array element number of sample rate list
			0, //Selected sample rate index
			${channel.inputPort.sampleSize}, // Sample size
			PORT_TYPE_${channel.inputPort.portType}, // Port type
			<#if channel.inputPort.subgraphPort??>&g_astPortInfo[${port_key_to_index[channel.inputPort.subgraphPort.portKey]}]<#else>NULL</#if>, // Pointer to Subgraph port
		}, // Input port information
		{
			${channel.outputPort.taskId}, // Task ID
			"${channel.outputPort.portName}", // Port name
			PORT_SAMPLE_RATE_${channel.outputPort.portSampleRateType}, // Port sample rate type
			g_astPortSampleRate_${channel.outputPort.taskName}_${channel.outputPort.portName}, // Array of sample rate list
			${channel.outputPort.portSampleRateList?size}, // Array element number of sample rate list
			0, //Selected sample rate index
			${channel.outputPort.sampleSize}, // Sample size
			PORT_TYPE_${channel.outputPort.portType}, // Port type
			<#if channel.outputPort.subgraphPort??>&g_astPortInfo[${port_key_to_index[channel.outputPort.subgraphPort.portKey]}]<#else>NULL</#if>, // Pointer to Subgraph port
		}, // Output port information
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
		${channel.maximumChunkNum},
		NULL, // Chunk list head
		NULL, // Chunk list tail
	},
</#list>
};
// ##CHANNEL_LIST_TEMPLATE::END

// ##TASK_LIST_TEMPLATE::START
<#list task_graph as graph_name, task_graph>
STask g_astTasks_${task_graph.name}[] = {
	<#list task_graph.taskList as task>
	{ 	${task.id}, // Task ID
		"${task.name}", // Task name
		TASK_TYPE_${task.type}, // Task Type
		g_ast_${task.name}_functions, // Task function array
		${task.taskFuncNum}, // Task function array number
		RUN_CONDITION_${task.runCondition}, // Run condition
		1, // Run rate
		${task.period}, // Period
		TIME_METRIC_${task.periodMetric}, // Period metric
		NULL, // Subgraph
		&g_stGraph_${task.parentTaskGraphName}, // Parent task graph
		<#if task.modeTransition??>&g_stModeTransition_${task.name}<#else>NULL</#if>, // MTM information
		<#if task.loopStruct??>&g_stLoopStruct_${task.name}<#else>NULL</#if>, // Loop information
		<#if (task.taskParamList?size > 0)>&g_astTaskParameter_${task.name}<#else>NULL</#if>, // Task parameter information
		<#if task.staticScheduled == true>TRUE<#else>FALSE</#if>, // Statically scheduled or not
		NULL, // Mutex
		NULL, // Conditional variable
	},
	</#list>
};
</#list>

// ##TASK_LIST_TEMPLATE::END

// ##TASK_GRAPH_TEMPLATE::START
<#list task_graph as graph_name, task_graph>
STaskGraph g_stGraph_${task_graph.name} = {
		GRAPH_TYPE_PROCESS_NETWORK, // TODO: Task graph type (not used now)
		g_astTasks_${task_graph.name}, // current task graph's task list
		NULL, // parent task
};
</#list>
// ##TASK_GRAPH_TEMPLATE::END

// ##TASK_ID_TO_TASK_MAP_TEMPLATE::START
STaskIdToTaskMap g_astTaskIdToTask[] = {
<#list flat_task as task_name, task>
	{ 	${task.id}, // Task ID
		"${task.name}", // Task name
		&g_astTasks_${task.parentTaskGraphName}[${task.inGraphIndex}], // Task structure pointer
	},
</#list>
};
// ##TASK_ID_TO_TASK_MAP_TEMPLATE::END


// ##PROCESSOR_INFO_TEMPLATE::START
SProcessor g_astProcessorInfo[] = {

<#list device_info as device_name, device>
	<#list device.processorList as processor>
	{ 	${processor.id}, // Processor ID
		<#if processor.isCPU == true>TRUE<#else>FALSE</#if>, // Processor is CPU?			
		"${processor.name}", // Processor name
		${processor.poolSize}, // Processor pool size
	},
	</#list>
</#list>
};
// ##PROCESSOR_INFO_TEMPLATE::END


<#macro printScheduledCode scheduleItem space>
	<#if scheduleItem.itemType == "LOOP">
		<#if (scheduleItem.repetition > 1) >
${space}for(${scheduleItem.variableName} = 0 ; ${scheduleItem.variableName} < ${scheduleItem.repetition} ; ${scheduleItem.variableName}++)
${space}{
		</#if>
		<#list scheduleItem.scheduleItemList as loop_schedule_item>
			<#if (scheduleItem.repetition <= 1) >
				<#assign newspace="${space}" />
			<#else>
				<#assign newspace="${space}	" />
			</#if>
			<@printScheduledCode loop_schedule_item newspace />
		</#list>
		<#if (scheduleItem.repetition > 1) >		
${space}}

		</#if>
	<#else>
		<#if (scheduleItem.repetition > 1) >
${space}for(${scheduleItem.variableName} = 0 ; ${scheduleItem.variableName} < ${scheduleItem.repetition} ; ${scheduleItem.variableName}++)
${space}{
			<#assign innerspace="${space}	" />
		<#else>
			<#assign innerspace="${space}" />
		</#if>
${innerspace}${scheduleItem.taskName}_Go${scheduleItem.taskFuncId}();
		<#if (scheduleItem.repetition > 1) >		
${space}}

		</#if>
	</#if>
</#macro>

// ##SCHEDULED_COMPOSITE_TASK_FUNCTION_IMPLEMENTATION::START
<#list schedule_info as task_name, mapped_schedule>
	<#list mapped_schedule.mappedProcessorList as compositeMappedProcessor>
		<#list compositeMappedProcessor.compositeTaskScheduleList as task_schedule>
void ${mapped_schedule.parentTaskName}_${compositeMappedProcessor.modeId}_${compositeMappedProcessor.processorId}_${compositeMappedProcessor.processorLocalId}_${task_schedule.scheduleId}_Go() 
{
<#if (task_schedule.maxLoopVariableNum > 0) >
	<#list 0..(task_schedule.maxLoopVariableNum-1) as variable_id>
	int depth${variable_id};
	</#list>

</#if>
<#list task_schedule.scheduleList as scheduleItem>
	<@printScheduledCode scheduleItem "	" />
</#list>
}
		</#list>
	</#list>
</#list>
// ##SCHEDULED_COMPOSITE_TASK_FUNCTION_IMPLEMENTATION::END

// ##SCHEDULED_COMPOSITE_TASKS_TEMPLATE::START
<#list schedule_info as task_name, scheduled_task>
	<#list scheduled_task.mappedProcessorList as compositeMappedProcessor>
SScheduleList g_astScheduleList_${scheduled_task.parentTaskName}_${compositeMappedProcessor.modeId}_${compositeMappedProcessor.processorId}_${compositeMappedProcessor.processorLocalId}[] = {
		<#list compositeMappedProcessor.compositeTaskScheduleList as task_schedule>
	{
		${task_schedule.scheduleId}, // Schedule ID
		${scheduled_task.parentTaskName}_${compositeMappedProcessor.modeId}_${compositeMappedProcessor.processorId}_${compositeMappedProcessor.processorLocalId}_${task_schedule.scheduleId}_Go, // Composite GO function
		${task_schedule.throughputConstraint}, // Throughput constraint
	},
		</#list>
};
	</#list>
</#list>
// ##SCHEDULED_COMPOSITE_TASKS_TEMPLATE::END



SScheduledTasks g_astScheduledTaskList[] = {
<#list schedule_info as task_name, mapped_schedule>
	<#list mapped_schedule.mappedProcessorList as compositeMappedProcessor>
	{	<#if mapped_schedule.parentTaskId == -1>NULL<#else>&g_astTasks_${flat_task[task_name].parentTaskGraphName}[${flat_task[task_name].inGraphIndex}]</#if>, // Parent Task ID
		${compositeMappedProcessor.modeId}, // Mode transition mode ID
		g_astScheduleList_${mapped_schedule.parentTaskName}_${compositeMappedProcessor.modeId}_${compositeMappedProcessor.processorId}_${compositeMappedProcessor.processorLocalId}, // schedule list per throught constraint
		${compositeMappedProcessor.compositeTaskScheduleList?size}, // The number of schedules in the schedule list
		0, // Schedule Index (Default to set 0)
		${compositeMappedProcessor.sequenceIdInMode}, // Mode Sequence ID 
	},
	</#list>
</#list>
};


// ##MAPPING_SCHEDULING_INFO_TEMPLATE::START
SMappingSchedulingInfo g_astMappingAndSchedulingInfo[] = {
<#list mapping_info as task_name, mapped_task>
	<#list mapped_task.mappedProcessorList as mappedProcessor>
	{	TASK_TYPE_${mapped_task.mappedTaskType}, // Task type
		{ .pstTask = &g_astTasks_${mapped_task.parentTaskGraphName}[${mapped_task.inGraphIndex}] }, // Task ID or composite task information
		${mappedProcessor.processorId}, // Processor ID
		${mappedProcessor.processorLocalId}, // Processor local ID
	},
	</#list>
</#list>
<#list schedule_info as task_name, scheduled_task>
	<#list scheduled_task.mappedProcessorList as scheduledProcessor>
	{	TASK_TYPE_${scheduled_task.mappedTaskType}, // Task type
		{ .pstScheduledTasks = &g_astScheduledTaskList[${scheduledProcessor.inArrayIndex}] }, // Task ID or composite task information
		${scheduledProcessor.processorId}, // Processor ID
		${scheduledProcessor.processorLocalId}, // Processor local ID
	},
	</#list>
</#list>
};
// ##MAPPING_SCHEDULING_INFO_TEMPLATE::END


int g_nChannelNum = ARRAYLEN(g_astChannels);
int g_nNumOfTasks_top = ARRAYLEN(g_astTasks_top);
int g_nTaskIdToTaskNum = ARRAYLEN(g_astTaskIdToTask);
int g_nProcessorInfoNum = ARRAYLEN(g_astProcessorInfo);
int g_nMappingAndSchedulingInfoNum = ARRAYLEN(g_astMappingAndSchedulingInfo);



