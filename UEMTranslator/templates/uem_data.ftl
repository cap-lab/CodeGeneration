/*
 * uem_data.c
 *
 *  Created on: 2017. 9. 7.
 *      Author: jej
 */


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
// ##CHANNEL_LOOP::START
#define CHANNEL_/*[CHANNEL_ID]*/_SIZE (/*[CHANNEL_SIZE]*/)
// ##CHANNEL_LOOP::END
// ##CHANNEL_SIZE_DEFINITION_TEMPLATE::END

// ##CHANNEL_BUFFER_DEFINITION_TEMPLATE::START
// ##CHANNEL_LOOP::START
char s_pChannel_/*[CHANNEL_ID]*/_buffer[CHANNEL_/*[CHANNEL_ID]*/_SIZE];
// ##CHANNEL_LOOP::END
// ##CHANNEL_BUFFER_DEFINITION_TEMPLATE::END

// ##CHUNK_DEFINITION_TEMPLATE::START
// ##CHANNEL_LOOP::START
SChunk g_astChunk_channel_/*[CHANNEL_ID]*/_out[] = {
// ##CHUNK_LOOP::START
	{
		s_pChannel_/*[CHANNEL_ID]*/_buffer/*[CHUNK_LOOP_ADDRESS]*/, // Chunk start pointer
		s_pChannel_/*[CHANNEL_ID]*/_buffer/*[CHUNK_LOOP_ADDRESS]*/, // Data start pointer
		s_pChannel_/*[CHANNEL_ID]*/_buffer/*[CHUNK_LOOP_ADDRESS]*/, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
// ##CHUNK_LOOP::END
};

SChunk g_astChunk_channel_/*[CHANNEL_ID]*/_in[] = {
// ##CHUNK_LOOP::START
	{
		s_pChannel_/*[CHANNEL_ID]*/_buffer/*[CHUNK_LOOP_ADDRESS]*/, // Chunk start pointer
		s_pChannel_/*[CHANNEL_ID]*/_buffer/*[CHUNK_LOOP_ADDRESS]*/, // Data start pointer
		s_pChannel_/*[CHANNEL_ID]*/_buffer/*[CHUNK_LOOP_ADDRESS]*/, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
// ##CHUNK_LOOP::END
};

// ##CHANNEL_LOOP::END
// ##CHUNK_DEFINITION_TEMPLATE::END


// ##PORT_SAMPLE_RATE_TEMPLATE::START
// ##CHANNEL_LOOP::START
SPortSampleRate g_astPortSampleRate_/*[INPUT_PORT_NESTED_TASK_NAME]*/_/*[INPUT_PORT_NAME]*/[] = {
// ##INPUT_PORT_MODE_LOOP::START
	{ 	/*[MODE_NAME]*/, // Mode name
		/*[MODE_SAMPLE_RATE]*/, // Sample rate
		/*[BROADCAST_LOOP_COUNT]*/, // Available number of data
	},
// ##INPUT_PORT_MODE_LOOP::END	
};

SPortSampleRate g_astPortSampleRate_/*[OUTPUT_PORT_NESTED_TASK_NAME]*/_/*[OUTPUT_PORT_NAME]*/[] = {
// ##INPUT_PORT_MODE_LOOP::START
	{ 	/*[MODE_NAME]*/, // Mode name
		/*[MODE_SAMPLE_RATE]*/, // Sample rate
		/*[BROADCAST_LOOP_COUNT]*/, // Available number of data
	},
// ##INPUT_PORT_MODE_LOOP::END	
};
// ##CHANNEL_LOOP::END
// ##PORT_SAMPLE_RATE_TEMPLATE::END


// ##AVAILABLE_CHUNK_LIST_TEMPLATE::START
// ##CHANNEL_LOOP::START
SAvailableChunk g_astAvailableInputChunk_channel_/*[CHANNEL_ID]*/[] = {
// ##CHUNK_LOOP::START
	{ /*[CHUNK_LOOP_INDEX]*/, 0, NULL, NULL, },
// ##CHUNK_LOOP::END
};
// ##CHANNEL_LOOP::END
// ##AVAILABLE_CHUNK_LIST_TEMPLATE::END


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
// ##CHANNEL_LOOP::START
	{
		/*[CHANNEL_ID]*/, // Channel ID
		/*[CHANNEL_COMMUNICATION_TYPE]*/, // Channel communication type
		/*[CHANNEL_TYPE]*/, // Channel type
		s_pChannel_/*[CHANNEL_ID]*/_buffer, // Channel buffer pointer
		CHANNEL_/*[CHANNEL_ID]*/_SIZE, // Channel size
		s_pChannel_/*[CHANNEL_ID]*/_buffer, // Channel data start
		s_pChannel_/*[CHANNEL_ID]*/_buffer, // Channel data end
		0, // Channel data length
		0, // reference count
		NULL, // Mutex
		NULL, // Event
		{
			/*[INPUT_PORT_TASK_ID]*/, // Task ID
			/*[INPUT_PORT_NAME]*/, // Port name
			/*[INPUT_PORT_SAMPLE_RATE_TYPE]*/, // Port sample rate type
			g_astPortSampleRate_/*[INPUT_PORT_NESTED_TASK_NAME]*/_/*[INPUT_PORT_NAME]*/, // Array of sample rate list
			/*[INPUT_PORT_MODE_NUM]*/, // Array element number of sample rate list
			0, //Selected sample rate index
			/*[INPUT_PORT_SAMPLE_SIZE]*/, // Sample size
			/*[INPUT_PORT_TYPE]*/, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			/*[OUTPUT_PORT_TYPE]*/, // Task ID
			/*[OUTPUT_PORT_NAME]*/, // Port name
			/*[OUTPUT_PORT_SAMPLE_RATE_TYPE]*/, // Port sample rate type
			g_astPortSampleRate_/*[OUTPUT_PORT_NESTED_TASK_NAME]*/_/*[OUTPUT_PORT_NAME]*/, // Array of sample rate list
			/*[OUTPUT_PORT_MODE_NUM]*/, // Array element number of sample rate list
			0, //Selected sample rate index
			/*[OUTPUT_PORT_SAMPLE_SIZE]*/, // Sample size
			/*[OUTPUT_PORT_TYPE]*/, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
		{
			g_astChunk_channel_/*[CHANNEL_ID]*/_in, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Input chunk information
		{
			g_astChunk_channel_/*[CHANNEL_ID]*/out, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Output chunk information
		CHUNK_NUM_NOT_INITIALIZED, // Written output chunk number
		g_astAvailableInputChunk_channel_/*[CHANNEL_ID]*/, // Available chunk list
		/*[MAXIMUM_CHUNK_NUM]*/,
		NULL, // Chunk list head
		NULL, // Chunk list tail
	},
// ##CHANNEL_LOOP::END
};
// ##CHANNEL_LIST_TEMPLATE::END

// ##TASK_LIST_TEMPLATE::START
<#list task_graph as graph_name, task_graph>
STask g_astTasks_${task_graph.name}[] = {
<#list task_graph.taskList as task>
	{ 	${task.id}, // Task ID
		${task.name}, // Task name
		TASK_TYPE_${task.type}, // Task Type
		g_ast_${task.name}_functions, // Task function array
		${task.taskFuncNum}, // Task function array number
		RUN_CONDITION_${task.runCondition}, // Run condition
		1, // Run rate
		${task.period}, // Period
		TIME_METRIC_${task.periodMetric}, // Period metric
		NULL, // Subgraph
		&g_stGraph_${task.parentTaskGraphName}, // Parent task graph
		NULL, // MTM information
		NULL, // Loop information
		NULL, // Task parameter information
		${task.staticScheduled?c}, // Statically scheduled or not
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
		/*[TASK_GRAPH_TYPE]*/, // Task graph type
		g_astTasks_${task_graph.name}, // current task graph's task list
		NULL, // parent task
};
</#list>
// ##TASK_GRAPH_TEMPLATE::END

// ##TASK_ID_TO_TASK_MAP_TEMPLATE::START
STaskIdToTaskMap g_astTaskIdToTask[] = {
<#list flat_task as task_name, task>
	{ 	${task.id}, // Task ID
		${task.name}, // Task name
		&g_astTasks_${task.parentTaskGraphName}[${task.inGraphIndex}], // Task structure pointer
	},
</#list>
};
// ##TASK_ID_TO_TASK_MAP_TEMPLATE::END


// ##PROCESSOR_INFO_TEMPLATE::START
SProcessor g_astProcessorInfo[] = {
// ##PROCESSOR_LOOP::START
	{ 	/*[PROCESSOR_ID]*/, // Processor ID
		/*[PROCESSOR_IS_CPU]*/, // Processor is CPU?			
		/*[PROCESSOR_NAME]*/, // Processor name
		/*[PROCESSOR_POOL_SIZE]*/, // Processor pool size
	},
// ##PROCESSOR_LOOP::END
};
// ##PROCESSOR_INFO_TEMPLATE::END


// ##SCHEDULED_COMPOSITE_TASKS_TEMPLATE::START

// ##SCHEDULED_COMPOSITE_TASKS_TEMPLATE::END
SScheduledTasks g_astScheduledTaskList[] = {
// ##SCHEDULED_TASKS_LOOP::START
	{	/*[PARENT_TASK_ID]*/, // Parent Task ID
		/*[MODE_ID]*/, // Mode transition mode ID
		/*[SCHEDULE_LIST]*/, // schedule list per throught constraint
		/*[NUM_OF_SCHEDULES]*/, // The number of schedules in the schedule list
		0, // Schedule Index (Default to set 0)
		/*[SEQUENCE_IN_MODE]*/, // Mode Sequence ID 
	},
// ##SCHEDULED_TASKS_LOOP::END
};

// ##MAPPING_SCHEDULING_INFO_TEMPLATE::START
SMappingSchedulingInfo g_astMappingAndSchedulingInfo[] = {
// ##TASK_MAPPING_LOOP::START
	{	/*[MAPPED_TASK_TYPE]*/, // Task type
		{ /*[MAPPED_TASK_INFO]*/ }, // Task ID or composite task information
		/*[MAPPED_PROCESSOR_ID]*/, // Processor ID
		/*[MAPPED_PROCESSOR_ID_LOCAL_ID]*/, // Processor local ID
	},
// ##TASK_MAPPING_LOOP::END
};
// ##MAPPING_SCHEDULING_INFO_TEMPLATE::END




