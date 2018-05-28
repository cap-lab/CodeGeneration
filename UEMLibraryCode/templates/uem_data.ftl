/* uem_data.c made by UEM Translator */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_data.h>
#include <UKTask.h>
#include <UKModeTransition.h>
#include <UKHostMemorySystem.h>
<#if gpu_used == true>
#include <UKGPUMemorySystem.h>
</#if>

SExecutionTime g_stExecutionTime = { ${execution_time.value?c}, TIME_METRIC_${execution_time.metric} } ;


<#assign timerSlotSize=10 />
#define MAX_TIMER_SLOT_SIZE (${timerSlotSize})



<#list flat_task as task_name, task>
	<#if task.type == "CONTROL">
STimer g_stTimer_${task.name}[MAX_TIMER_SLOT_SIZE] = {
		<#list 0..(timerSlotSize-1) as index>
	{
		${index}, // Slot ID 
		INVALID_TIME_VALUE, // Timer value
		FALSE, // Alarm checked
		0, // Alarm time
	},
		</#list>
};

	</#if>
</#list>

// ##TASK_CODE_TEMPLATE::START
<#list flat_task as task_name, task>
	<#if !task.childTaskGraphName??>
		<#if task.language=="C">
#ifdef __cplusplus
extern "C"
{
#endif 
		</#if>
		<#list 0..(task.taskFuncNum-1) as task_func_id>
void ${task.name}_Init${task_func_id}(int nTaskId);
void ${task.name}_Go${task_func_id}(int nTaskId);
void ${task.name}_Wrapup${task_func_id}();
		</#list>
		<#if task.language=="C">
#ifdef __cplusplus
}
#endif 
		</#if>
	</#if>

</#list>
// ##TASK_CODE_TEMPLATE::END


// ##LIBRARY_INIT_WRAPUP_TEMPLATE::START
<#list library_info as libraryName, library>
		<#if library.language=="C">
#ifdef __cplusplus
extern "C"
{
#endif 
		</#if>
void l_${libraryName}_init();
void l_${libraryName}_wrapup();
		<#if library.language=="C">
#ifdef __cplusplus
}
#endif 
		</#if>

</#list>
// ##LIBRARY_INIT_WRAPUP_TEMPLATE::END


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


// ##LOOP_STRUCTURE_TEMPLATE::START
<#list flat_task as task_name, task>
	<#if task.loopStruct??>
SLoopInfo g_stLoopStruct_${task.name} = {
	LOOP_TYPE_${task.loopStruct.loopType},
	${task.loopStruct.loopCount},
	${task.loopStruct.designatedTaskId},
};

	</#if>
</#list>
// ##LOOP_STRUCTURE_TEMPLATE::END

// ##TASK_LIST_DECLARATION_TEMPLATE::START
<#list task_graph as graph_name, task_graph>
extern STask g_astTasks_${task_graph.name}[];
</#list>
// ##TASK_LIST_DECLARATION_TEMPLATE::END


// ##TASK_GRAPH_DECLARATION_TEMPLATE::START
<#list task_graph as graph_name, task_graph>
extern STaskGraph g_stGraph_${task_graph.name};
</#list>
// ##TASK_GRAPH_DECLARATION_TEMPLATE::END


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
		g_pastRelatedChildTasks_${task.name}_${task_mode.name},
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
};

		<#if (task.modeTransition.modeMap?size > 1)>
static uem_bool transitMode_${task.name}(SModeTransitionMachine *pstModeTransition) 
{
	uem_bool bModeChanged = FALSE;
			<#list task.modeTransition.variableMap as var_name, var_type>
	int ${var_name};
			</#list>
	int nCurrentModeId = pstModeTransition->astModeMap[pstModeTransition->nCurModeIndex].nModeId;
	int nNextModeId = nCurrentModeId;
	int nVarIndex = 0;
	
			<#list task.modeTransition.variableMap as var_name, var_type>
	nVarIndex = UKModeTransition_GetVariableIndexByName(pstModeTransition, "${var_name}");
	${var_name} = pstModeTransition->astVarIntMap[nVarIndex].nValue;
			</#list>
		
			<#list task.modeTransition.transitionList as transition>
	if(nCurrentModeId == ${task.modeTransition.modeMap[transition.srcMode].id}
	  <#list transition.conditionList as condition>&& ${condition.leftOperand} ${condition.operator} ${condition.rightOperand}</#list> )
	{
		nNextModeId = ${task.modeTransition.modeMap[transition.dstMode].id};
		bModeChanged = TRUE;
	}
			</#list>
	if(bModeChanged == TRUE)
	{	// update only the mode is changed
		pstModeTransition->nNextModeIndex = UKModeTransition_GetModeIndexByModeId(pstModeTransition, nNextModeId);
		pstModeTransition->enModeState = MODE_STATE_TRANSITING;
	}
	
	return bModeChanged;
}
		</#if>

SModeTransitionMachine g_stModeTransition_${task.name} = {
	${task.id},
	g_astModeMap_${task.name}, // mode list
	${task.modeTransition.modeMap?size}, // number of modes
	g_astVariableIntMap_${task.name}, // Integer variable list
	${task.modeTransition.variableMap?size}, // number of integer variables
	<#if (task.modeTransition.modeMap?size > 1)>transitMode_${task.name}<#else>NULL</#if>, // mode transition function
	0, // Current mode index
	0, // Next mode index
	MODE_STATE_TRANSITING, // mode state (to decide source task execution)
};
	</#if>
</#list>
// ##MODE_TRANSITION_TEMPLATE::END


// ##AVAILABLE_CHUNK_LIST_TEMPLATE::START
<#list channel_list as channel>
SAvailableChunk g_astAvailableInputChunk_channel_${channel.index}[] = {
<#list 0..(channel.inputPort.maximumChunkNum-1) as chunk_id>
	{ ${chunk_id}, 0, (SAvailableChunk *) NULL, (SAvailableChunk *) NULL, },
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
		"${task_param.name}",
		{ <#if task_param.type == "INT">.nParam = ${task_param.value?c},<#else>.dbParam = ${task_param.value?c}</#if> },
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

SSharedMemoryChannel g_stSharedMemoryChannel_${channel.index} = {
	<#switch channel.communicationType>
		<#case "SHARED_MEMORY">		 
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
		<#case "TCP_CLIENT">
			<#break>
		<#case "TCP_SERVER">
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
		<#case "TCP_CLIENT">
			<#break>
		<#case "TCP_SERVER">
			<#break>
	</#switch>
};

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
		<#case "TCP_CLIENT">
			<#break>
		<#case "TCP_SERVER">
			<#break>
		</#switch>
	},
</#list>
};
// ##CHANNEL_LIST_TEMPLATE::END




// ##TASK_ITERATION_TEMPLATE::START
<#list flat_task as task_name, task>
STaskIteration g_astTaskIteration_${task_name}[] = {
	<#list task.iterationCountList as mode_id, count_value>
	{
		${mode_id}, // Mode ID
		<#if count_value == 0>1<#else>${count_value}</#if>, // iteration count
	},
	</#list>	
};

</#list>
// ##TASK_ITERATION_TEMPLATE::END


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
		${task.period?c}, // Period
		TIME_METRIC_${task.periodMetric}, // Period metric
		<#if task.childTaskGraphName??>&g_stGraph_${task.childTaskGraphName}<#else>(STaskGraph *) NULL</#if>, // Subgraph
		&g_stGraph_${task.parentTaskGraphName}, // Parent task graph
		<#if task.modeTransition??>&g_stModeTransition_${task.name}<#else>(SModeTransitionMachine *) NULL</#if>, // MTM information
		<#if task.loopStruct??>&g_stLoopStruct_${task.name}<#else>(SLoopInfo *) NULL</#if>, // Loop information
		<#if (task.taskParamList?size > 0)>g_astTaskParameter_${task.name}<#else>(STaskParameter *) NULL</#if>, // Task parameter information
		${task.taskParamList?size}, // Task parameter number
		<#if task.staticScheduled == true>TRUE<#else>FALSE</#if>, // Statically scheduled or not
		0,	  // Throughput constraint
		(HThreadMutex) NULL, // Mutex
		(HThreadEvent) NULL, // Conditional variable
		g_astTaskIteration_${task.name}, // Task iteration count (only used when the parent task graph is data flow)
		0, // current run count in iteration
		0, // current iteration
		0, // target iteration (this variable is used for calling delayed stop task)
		<#if task.type == "CONTROL">g_stTimer_${task.name}<#else>(STimer *) NULL</#if>, // Timer slot (used by control task)
	},
	</#list>
};

</#list>

// ##TASK_LIST_TEMPLATE::END

// ##TASK_GRAPH_TEMPLATE::START
<#list task_graph as graph_name, task_graph_element>
STaskGraph g_stGraph_${task_graph_element.name} = {
		GRAPH_TYPE_${task_graph_element.taskGraphType}, // Task graph type
		g_astTasks_${task_graph_element.name}, // current task graph's task list
		${task_graph_element.taskList?size}, // number of tasks
		<#if task_graph_element.parentTask??>&g_astTasks_${task_graph_element.parentTask.parentTaskGraphName}[${task_graph_element.parentTask.inGraphIndex}]<#else>(STask *) NULL</#if>, // parent task
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


<#macro printScheduledCode scheduleItem space compositeMappedProcessor parentTaskName>
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
			<@printScheduledCode loop_schedule_item newspace compositeMappedProcessor parentTaskName />
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
		<#if compositeMappedProcessor.srcTaskMap[scheduleItem.taskName]?? && (flat_task[parentTaskName].modeTransition.modeMap?size > 1)>
${innerspace}{
${innerspace}	EModeState enModeState = MODE_STATE_TRANSITING;
${innerspace}	uem_result result;
${innerspace}	STask *pstTask = NULL;
${innerspace}	
${innerspace}	enModeState = UKModeTransition_GetModeState(nTaskId);
${innerspace}
${innerspace}	if(enModeState == MODE_STATE_TRANSITING)
${innerspace}	{
${innerspace}		${scheduleItem.taskName}_Go${scheduleItem.taskFuncId}(${flat_task[scheduleItem.taskName].id});//printf("${scheduleItem.taskName}_Go${scheduleItem.taskFuncId} called-- (Line: %d)\n", __LINE__);
${innerspace}		result = UKTask_GetTaskFromTaskId(nTaskId, &pstTask);
${innerspace}		if(result == ERR_UEM_NOERROR)
${innerspace}		{
${innerspace}			result = UCThreadMutex_Lock(pstTask->hMutex);
${innerspace}			if(result == ERR_UEM_NOERROR){
${innerspace}				transitMode_${parentTaskName}(g_astTasks_${flat_task[parentTaskName].parentTaskGraphName}[${flat_task[parentTaskName].inGraphIndex}].pstMTMInfo);
${innerspace}				UCThreadMutex_Unlock(pstTask->hMutex);
${innerspace}			}	
${innerspace}			return; // exit when the mode is MODE_STATE_TRANSITING
${innerspace}		}
${innerspace}	}
${innerspace}}
		<#else>
${innerspace}${scheduleItem.taskName}_Go${scheduleItem.taskFuncId}(${flat_task[scheduleItem.taskName].id});//printf("${scheduleItem.taskName}_Go${scheduleItem.taskFuncId} called (Line: %d)\n", __LINE__);
		</#if>
		<#if compositeMappedProcessor.srcTaskMap[scheduleItem.taskName]??>
${innerspace}{
${innerspace}	EInternalTaskState enState = INTERNAL_STATE_STOP;
${innerspace}	UKTask_GetTaskState(nTaskId, "${parentTaskName}", &enState);
${innerspace}	if(enState == INTERNAL_STATE_STOP || enState == INTERNAL_STATE_END) return; 
${innerspace}}
		</#if>			
		<#if (scheduleItem.repetition > 1) >
${space}}

		</#if>
	</#if>
</#macro>

// ##SCHEDULED_COMPOSITE_TASK_FUNCTION_IMPLEMENTATION::START
<#list schedule_info as task_name, mapped_schedule>
	<#list mapped_schedule.mappedProcessorList as compositeMappedProcessor>
		<#list compositeMappedProcessor.compositeTaskScheduleList as task_schedule>
void ${mapped_schedule.parentTaskName}_${compositeMappedProcessor.modeId}_${compositeMappedProcessor.processorId}_${compositeMappedProcessor.processorLocalId}_${task_schedule.throughputConstraint?c}_Go(int nTaskId) 
{
			<#if (task_schedule.maxLoopVariableNum > 0) >
				<#list 0..(task_schedule.maxLoopVariableNum-1) as variable_id>
	int depth${variable_id};
				</#list>

			</#if>
<#list task_schedule.scheduleList as scheduleItem>
	<@printScheduledCode scheduleItem "	" compositeMappedProcessor mapped_schedule.parentTaskName />
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
		${scheduled_task.parentTaskName}_${compositeMappedProcessor.modeId}_${compositeMappedProcessor.processorId}_${compositeMappedProcessor.processorLocalId}_${task_schedule.throughputConstraint?c}_Go, // Composite GO function
		${task_schedule.throughputConstraint?c}, // Throughput constraint
		<#if task_schedule.hasSourceTask == true>TRUE<#else>FALSE</#if>,
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
		g_astScheduleList_${mapped_schedule.parentTaskName}_${compositeMappedProcessor.modeId}_${compositeMappedProcessor.processorId}_${compositeMappedProcessor.processorLocalId}, // schedule list per throughput constraint
		${compositeMappedProcessor.compositeTaskScheduleList?size}, // The number of schedules in the schedule list
		0, // Schedule Index (Default to set 0)
		${compositeMappedProcessor.sequenceIdInMode}, // Mode Sequence ID 
	},
	</#list>
</#list>
};


// ##MAPPING_SCHEDULING_INFO_TEMPLATE::START

SMappedGeneralTaskInfo g_astGeneralTaskMappingInfo[] = {
<#list mapping_info as task_name, mapped_task>
	<#list mapped_task.mappedProcessorList as mappedProcessor>
	{	TASK_TYPE_${mapped_task.mappedTaskType}, // Task type
		&g_astTasks_${mapped_task.parentTaskGraphName}[${mapped_task.inGraphIndex}], // Task ID or composite task information
		${mappedProcessor.processorId}, // Processor ID
		${mappedProcessor.processorLocalId}, // Processor local ID
	},
	</#list>
</#list>
};


SMappedCompositeTaskInfo g_astCompositeTaskMappingInfo[] = {
<#list schedule_info as task_name, scheduled_task>
	<#list scheduled_task.mappedProcessorList as scheduledProcessor>
	{
		&g_astScheduledTaskList[${scheduledProcessor.inArrayIndex}],
		${scheduledProcessor.processorId}, // Processor ID
		${scheduledProcessor.processorLocalId}, // Processor local ID		
	},
	</#list>
</#list>
};


SMappedTaskInfo g_stMappingInfo = {
	<#if (mapping_info?size > 0)>g_astGeneralTaskMappingInfo<#else>NULL</#if>, // general task array
	<#if (mapping_info?size > 0)>ARRAYLEN(g_astGeneralTaskMappingInfo)<#else>0</#if>, // size of general task array
	<#if (schedule_info?size > 0)>g_astCompositeTaskMappingInfo<#else>(SMappedCompositeTaskInfo *) NULL</#if>, // composite task array
	<#if (schedule_info?size > 0)>ARRAYLEN(g_astCompositeTaskMappingInfo)<#else>0</#if>, // size of composite task array
};

// ##MAPPING_SCHEDULING_INFO_TEMPLATE::END


// ##LIBRARY_INFO_TEMPLATE::START
SLibrary g_stLibraryInfo[] = {
<#list library_info as libraryName, library>
	{
		"${libraryName}",
		l_${libraryName}_init,
		l_${libraryName}_wrapup,
	},
</#list>
};

// ##LIBRARY_INFO_TEMPLATE::END


int g_nChannelNum = ARRAYLEN(g_astChannels);
int g_nNumOfTasks_top = ARRAYLEN(g_astTasks_top);
int g_nTaskIdToTaskNum = ARRAYLEN(g_astTaskIdToTask);
int g_nProcessorInfoNum = ARRAYLEN(g_astProcessorInfo);
int g_nLibraryInfoNum = <#if (library_info?size > 0)>ARRAYLEN(g_stLibraryInfo)<#else>0</#if>;
int g_nTimerSlotNum = MAX_TIMER_SLOT_SIZE;
