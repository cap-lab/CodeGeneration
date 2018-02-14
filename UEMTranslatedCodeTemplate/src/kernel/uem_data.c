/* uem_data.c made by UEM Translator */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_data.h>
#include <UKTask.h>
#include <UKModeTransition.h>

SExecutionTime g_stExecutionTime = { 20, TIME_METRIC_SEC } ;

// ##TASK_CODE_TEMPLATE::START
void H263_Dec_Init0(int nTaskId);
void H263_Dec_Go0(int nTaskId);
void H263_Dec_Wrapup0();

void H263_Enc_Init0(int nTaskId);
void H263_Enc_Go0(int nTaskId);
void H263_Enc_Wrapup0();

void Display_Init0(int nTaskId);
void Display_Go0(int nTaskId);
void Display_Wrapup0();

// ##TASK_CODE_TEMPLATE::END


// ##LIBRARY_INIT_WRAPUP_TEMPLATE::START
void l_dct_init();
void l_dct_wrapup();

void l_idct_init();
void l_idct_wrapup();

// ##LIBRARY_INIT_WRAPUP_TEMPLATE::END


// ##CHANNEL_SIZE_DEFINITION_TEMPLATE::START
#define CHANNEL_0_SIZE (30000)
#define CHANNEL_1_SIZE (4)
#define CHANNEL_2_SIZE (25344)
#define CHANNEL_3_SIZE (6336)
#define CHANNEL_4_SIZE (6336)
// ##CHANNEL_SIZE_DEFINITION_TEMPLATE::END

// ##CHANNEL_BUFFER_DEFINITION_TEMPLATE::START
char s_pChannel_0_buffer[CHANNEL_0_SIZE];
char s_pChannel_1_buffer[CHANNEL_1_SIZE];
char s_pChannel_2_buffer[CHANNEL_2_SIZE];
char s_pChannel_3_buffer[CHANNEL_3_SIZE];
char s_pChannel_4_buffer[CHANNEL_4_SIZE];
// ##CHANNEL_BUFFER_DEFINITION_TEMPLATE::END

// ##CHUNK_DEFINITION_TEMPLATE::START
SChunk g_astChunk_channel_0_out[] = {
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_0_in[] = {
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_1_out[] = {
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_1_in[] = {
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_2_out[] = {
	{
		s_pChannel_2_buffer, // Chunk start pointer
		s_pChannel_2_buffer, // Data start pointer
		s_pChannel_2_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_2_in[] = {
	{
		s_pChannel_2_buffer, // Chunk start pointer
		s_pChannel_2_buffer, // Data start pointer
		s_pChannel_2_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_3_out[] = {
	{
		s_pChannel_3_buffer, // Chunk start pointer
		s_pChannel_3_buffer, // Data start pointer
		s_pChannel_3_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_3_in[] = {
	{
		s_pChannel_3_buffer, // Chunk start pointer
		s_pChannel_3_buffer, // Data start pointer
		s_pChannel_3_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_4_out[] = {
	{
		s_pChannel_4_buffer, // Chunk start pointer
		s_pChannel_4_buffer, // Data start pointer
		s_pChannel_4_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_4_in[] = {
	{
		s_pChannel_4_buffer, // Chunk start pointer
		s_pChannel_4_buffer, // Data start pointer
		s_pChannel_4_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

// ##CHUNK_DEFINITION_TEMPLATE::END
//portSampleRateList


// ##PORT_SAMPLE_RATE_TEMPLATE::START
SPortSampleRate g_astPortSampleRate_H263_Dec_stream_in[] = {
};

SPortSampleRate g_astPortSampleRate_H263_Enc_stream_out[] = {
};

SPortSampleRate g_astPortSampleRate_H263_Dec_size_in[] = {
};

SPortSampleRate g_astPortSampleRate_H263_Enc_size_out[] = {
};

SPortSampleRate g_astPortSampleRate_Display_Y_in[] = {
};

SPortSampleRate g_astPortSampleRate_H263_Dec_Y_out[] = {
};

SPortSampleRate g_astPortSampleRate_Display_U_in[] = {
};

SPortSampleRate g_astPortSampleRate_H263_Dec_U_out[] = {
};

SPortSampleRate g_astPortSampleRate_Display_V_in[] = {
};

SPortSampleRate g_astPortSampleRate_H263_Dec_V_out[] = {
};

// ##PORT_SAMPLE_RATE_TEMPLATE::END


// ##PORT_ARRAY_TEMPLATE::START
SPort g_astPortInfo[] = {
	{
		1, // Task ID
		"stream_in", // Port name
		PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
		g_astPortSampleRate_H263_Dec_stream_in, // Array of sample rate list
		0, // Array element number of sample rate list
		0, //Selected sample rate index
		30000, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
		
	{
		0, // Task ID
		"stream_out", // Port name
		PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
		g_astPortSampleRate_H263_Enc_stream_out, // Array of sample rate list
		0, // Array element number of sample rate list
		0, //Selected sample rate index
		30000, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
		
	{
		1, // Task ID
		"size_in", // Port name
		PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
		g_astPortSampleRate_H263_Dec_size_in, // Array of sample rate list
		0, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
		
	{
		0, // Task ID
		"size_out", // Port name
		PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
		g_astPortSampleRate_H263_Enc_size_out, // Array of sample rate list
		0, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
		
	{
		2, // Task ID
		"Y_in", // Port name
		PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
		g_astPortSampleRate_Display_Y_in, // Array of sample rate list
		0, // Array element number of sample rate list
		0, //Selected sample rate index
		25344, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
		
	{
		1, // Task ID
		"Y_out", // Port name
		PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
		g_astPortSampleRate_H263_Dec_Y_out, // Array of sample rate list
		0, // Array element number of sample rate list
		0, //Selected sample rate index
		25344, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
		
	{
		2, // Task ID
		"U_in", // Port name
		PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
		g_astPortSampleRate_Display_U_in, // Array of sample rate list
		0, // Array element number of sample rate list
		0, //Selected sample rate index
		6336, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
		
	{
		1, // Task ID
		"U_out", // Port name
		PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
		g_astPortSampleRate_H263_Dec_U_out, // Array of sample rate list
		0, // Array element number of sample rate list
		0, //Selected sample rate index
		6336, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
		
	{
		2, // Task ID
		"V_in", // Port name
		PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
		g_astPortSampleRate_Display_V_in, // Array of sample rate list
		0, // Array element number of sample rate list
		0, //Selected sample rate index
		6336, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
		
	{
		1, // Task ID
		"V_out", // Port name
		PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
		g_astPortSampleRate_H263_Dec_V_out, // Array of sample rate list
		0, // Array element number of sample rate list
		0, //Selected sample rate index
		6336, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
		
};
// ##PORT_ARRAY_TEMPLATE::END


// ##LOOP_STRUCTURE_TEMPLATE::START
// ##LOOP_STRUCTURE_TEMPLATE::END

// ##TASK_LIST_DECLARATION_TEMPLATE::START
STask g_astTasks_top[];
// ##TASK_LIST_DECLARATION_TEMPLATE::END


// ##TASK_GRAPH_DECLARATION_TEMPLATE::START
STaskGraph g_stGraph_top;
// ##TASK_GRAPH_DECLARATION_TEMPLATE::END


// ##MODE_TRANSITION_TEMPLATE::START
// ##MODE_TRANSITION_TEMPLATE::END


// ##AVAILABLE_CHUNK_LIST_TEMPLATE::START
SAvailableChunk g_astAvailableInputChunk_channel_0[] = {
	{ 0, 0, NULL, NULL, },
};
SAvailableChunk g_astAvailableInputChunk_channel_1[] = {
	{ 0, 0, NULL, NULL, },
};
SAvailableChunk g_astAvailableInputChunk_channel_2[] = {
	{ 0, 0, NULL, NULL, },
};
SAvailableChunk g_astAvailableInputChunk_channel_3[] = {
	{ 0, 0, NULL, NULL, },
};
SAvailableChunk g_astAvailableInputChunk_channel_4[] = {
	{ 0, 0, NULL, NULL, },
};
// ##AVAILABLE_CHUNK_LIST_TEMPLATE::END

// ##TASK_PARAMETER_TEMPLATE::START
// ##TASK_PARAMETER_TEMPLATE::END

// ##TASK_FUNCTION_LIST::START
STaskFunctions g_ast_H263_Dec_functions[] = {
	{
		H263_Dec_Init0, // Task init function
		H263_Dec_Go0, // Task go function
		H263_Dec_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_H263_Enc_functions[] = {
	{
		H263_Enc_Init0, // Task init function
		H263_Enc_Go0, // Task go function
		H263_Enc_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_Display_functions[] = {
	{
		Display_Init0, // Task init function
		Display_Go0, // Task go function
		Display_Wrapup0, // Task wrapup function
	},
};

// ##TASK_FUNCTION_LIST::END


// ##CHANNEL_LIST_TEMPLATE::START
SChannel g_astChannels[] = {
	{
		0, // Channel ID
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_GENERAL, // Channel type
		s_pChannel_0_buffer, // Channel buffer pointer
		CHANNEL_0_SIZE, // Channel size
		s_pChannel_0_buffer, // Channel data start
		s_pChannel_0_buffer, // Channel data end
		0, // Channel data length
		0, // Read reference count
		0, // Write reference count
		FALSE, // Read exit setting
		FALSE, // Write exit setting
		NULL, // Mutex
		NULL, // Read available notice event
		NULL, // Write available notice event
		{
			1, // Task ID
			"stream_in", // Port name
			PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
			g_astPortSampleRate_H263_Dec_stream_in, // Array of sample rate list
			0, // Array element number of sample rate list
			0, //Selected sample rate index
			30000, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			0, // Task ID
			"stream_out", // Port name
			PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
			g_astPortSampleRate_H263_Enc_stream_out, // Array of sample rate list
			0, // Array element number of sample rate list
			0, //Selected sample rate index
			30000, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
		{
			g_astChunk_channel_0_in, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Input chunk information
		{
			g_astChunk_channel_0_out, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Output chunk information
		CHUNK_NUM_NOT_INITIALIZED, // Written output chunk number
		g_astAvailableInputChunk_channel_0, // Available chunk list
		1,
		NULL, // Chunk list head
		NULL, // Chunk list tail
		0, // Initial data length 
	},
	{
		1, // Channel ID
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_GENERAL, // Channel type
		s_pChannel_1_buffer, // Channel buffer pointer
		CHANNEL_1_SIZE, // Channel size
		s_pChannel_1_buffer, // Channel data start
		s_pChannel_1_buffer, // Channel data end
		0, // Channel data length
		0, // Read reference count
		0, // Write reference count
		FALSE, // Read exit setting
		FALSE, // Write exit setting
		NULL, // Mutex
		NULL, // Read available notice event
		NULL, // Write available notice event
		{
			1, // Task ID
			"size_in", // Port name
			PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
			g_astPortSampleRate_H263_Dec_size_in, // Array of sample rate list
			0, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			0, // Task ID
			"size_out", // Port name
			PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
			g_astPortSampleRate_H263_Enc_size_out, // Array of sample rate list
			0, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
		{
			g_astChunk_channel_1_in, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Input chunk information
		{
			g_astChunk_channel_1_out, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Output chunk information
		CHUNK_NUM_NOT_INITIALIZED, // Written output chunk number
		g_astAvailableInputChunk_channel_1, // Available chunk list
		1,
		NULL, // Chunk list head
		NULL, // Chunk list tail
		0, // Initial data length 
	},
	{
		2, // Channel ID
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_GENERAL, // Channel type
		s_pChannel_2_buffer, // Channel buffer pointer
		CHANNEL_2_SIZE, // Channel size
		s_pChannel_2_buffer, // Channel data start
		s_pChannel_2_buffer, // Channel data end
		0, // Channel data length
		0, // Read reference count
		0, // Write reference count
		FALSE, // Read exit setting
		FALSE, // Write exit setting
		NULL, // Mutex
		NULL, // Read available notice event
		NULL, // Write available notice event
		{
			2, // Task ID
			"Y_in", // Port name
			PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
			g_astPortSampleRate_Display_Y_in, // Array of sample rate list
			0, // Array element number of sample rate list
			0, //Selected sample rate index
			25344, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			1, // Task ID
			"Y_out", // Port name
			PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
			g_astPortSampleRate_H263_Dec_Y_out, // Array of sample rate list
			0, // Array element number of sample rate list
			0, //Selected sample rate index
			25344, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
		{
			g_astChunk_channel_2_in, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Input chunk information
		{
			g_astChunk_channel_2_out, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Output chunk information
		CHUNK_NUM_NOT_INITIALIZED, // Written output chunk number
		g_astAvailableInputChunk_channel_2, // Available chunk list
		1,
		NULL, // Chunk list head
		NULL, // Chunk list tail
		0, // Initial data length 
	},
	{
		3, // Channel ID
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_GENERAL, // Channel type
		s_pChannel_3_buffer, // Channel buffer pointer
		CHANNEL_3_SIZE, // Channel size
		s_pChannel_3_buffer, // Channel data start
		s_pChannel_3_buffer, // Channel data end
		0, // Channel data length
		0, // Read reference count
		0, // Write reference count
		FALSE, // Read exit setting
		FALSE, // Write exit setting
		NULL, // Mutex
		NULL, // Read available notice event
		NULL, // Write available notice event
		{
			2, // Task ID
			"U_in", // Port name
			PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
			g_astPortSampleRate_Display_U_in, // Array of sample rate list
			0, // Array element number of sample rate list
			0, //Selected sample rate index
			6336, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			1, // Task ID
			"U_out", // Port name
			PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
			g_astPortSampleRate_H263_Dec_U_out, // Array of sample rate list
			0, // Array element number of sample rate list
			0, //Selected sample rate index
			6336, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
		{
			g_astChunk_channel_3_in, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Input chunk information
		{
			g_astChunk_channel_3_out, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Output chunk information
		CHUNK_NUM_NOT_INITIALIZED, // Written output chunk number
		g_astAvailableInputChunk_channel_3, // Available chunk list
		1,
		NULL, // Chunk list head
		NULL, // Chunk list tail
		0, // Initial data length 
	},
	{
		4, // Channel ID
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_GENERAL, // Channel type
		s_pChannel_4_buffer, // Channel buffer pointer
		CHANNEL_4_SIZE, // Channel size
		s_pChannel_4_buffer, // Channel data start
		s_pChannel_4_buffer, // Channel data end
		0, // Channel data length
		0, // Read reference count
		0, // Write reference count
		FALSE, // Read exit setting
		FALSE, // Write exit setting
		NULL, // Mutex
		NULL, // Read available notice event
		NULL, // Write available notice event
		{
			2, // Task ID
			"V_in", // Port name
			PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
			g_astPortSampleRate_Display_V_in, // Array of sample rate list
			0, // Array element number of sample rate list
			0, //Selected sample rate index
			6336, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			1, // Task ID
			"V_out", // Port name
			PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
			g_astPortSampleRate_H263_Dec_V_out, // Array of sample rate list
			0, // Array element number of sample rate list
			0, //Selected sample rate index
			6336, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
		{
			g_astChunk_channel_4_in, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Input chunk information
		{
			g_astChunk_channel_4_out, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Output chunk information
		CHUNK_NUM_NOT_INITIALIZED, // Written output chunk number
		g_astAvailableInputChunk_channel_4, // Available chunk list
		1,
		NULL, // Chunk list head
		NULL, // Chunk list tail
		0, // Initial data length 
	},
};
// ##CHANNEL_LIST_TEMPLATE::END


// ##TASK_LIST_TEMPLATE::START
STask g_astTasks_top[] = {
	{ 	0, // Task ID
		"H263_Enc", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_H263_Enc_functions, // Task function array
		1, // Task function array number
		RUN_CONDITION_TIME_DRIVEN, // Run condition
		1, // Run rate
		1, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // Subgraph
		&g_stGraph_top, // Parent task graph
		NULL, // MTM information
		NULL, // Loop information
		NULL, // Task parameter information
		0, // Task parameter number
		FALSE, // Statically scheduled or not
		0,	  // Throughput constraint
		NULL, // Mutex
		NULL, // Conditional variable
	},
	{ 	1, // Task ID
		"H263_Dec", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_H263_Dec_functions, // Task function array
		1, // Task function array number
		RUN_CONDITION_DATA_DRIVEN, // Run condition
		1, // Run rate
		1, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // Subgraph
		&g_stGraph_top, // Parent task graph
		NULL, // MTM information
		NULL, // Loop information
		NULL, // Task parameter information
		0, // Task parameter number
		FALSE, // Statically scheduled or not
		0,	  // Throughput constraint
		NULL, // Mutex
		NULL, // Conditional variable
	},
	{ 	2, // Task ID
		"Display", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_Display_functions, // Task function array
		1, // Task function array number
		RUN_CONDITION_DATA_DRIVEN, // Run condition
		1, // Run rate
		1, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // Subgraph
		&g_stGraph_top, // Parent task graph
		NULL, // MTM information
		NULL, // Loop information
		NULL, // Task parameter information
		0, // Task parameter number
		FALSE, // Statically scheduled or not
		0,	  // Throughput constraint
		NULL, // Mutex
		NULL, // Conditional variable
	},
};


// ##TASK_LIST_TEMPLATE::END

// ##TASK_GRAPH_TEMPLATE::START
STaskGraph g_stGraph_top = {
		GRAPH_TYPE_PROCESS_NETWORK, // TODO: Task graph type (not used now)
		g_astTasks_top, // current task graph's task list
		3, // number of tasks
		NULL, // parent task
};

// ##TASK_GRAPH_TEMPLATE::END

// ##TASK_ID_TO_TASK_MAP_TEMPLATE::START
STaskIdToTaskMap g_astTaskIdToTask[] = {
	{ 	1, // Task ID
		"H263_Dec", // Task name
		&g_astTasks_top[1], // Task structure pointer
	},
	{ 	0, // Task ID
		"H263_Enc", // Task name
		&g_astTasks_top[0], // Task structure pointer
	},
	{ 	2, // Task ID
		"Display", // Task name
		&g_astTasks_top[2], // Task structure pointer
	},
};
// ##TASK_ID_TO_TASK_MAP_TEMPLATE::END


// ##PROCESSOR_INFO_TEMPLATE::START
SProcessor g_astProcessorInfo[] = {

	{ 	0, // Processor ID
		TRUE, // Processor is CPU?			
		"i7_0", // Processor name
		4, // Processor pool size
	},
};
// ##PROCESSOR_INFO_TEMPLATE::END



// ##SCHEDULED_COMPOSITE_TASK_FUNCTION_IMPLEMENTATION::START
// ##SCHEDULED_COMPOSITE_TASK_FUNCTION_IMPLEMENTATION::END

// ##SCHEDULED_COMPOSITE_TASKS_TEMPLATE::START
// ##SCHEDULED_COMPOSITE_TASKS_TEMPLATE::END



SScheduledTasks g_astScheduledTaskList[] = {
};


// ##MAPPING_SCHEDULING_INFO_TEMPLATE::START
SMappingSchedulingInfo g_astMappingAndSchedulingInfo[] = {
	{	TASK_TYPE_COMPUTATIONAL, // Task type
		{ .pstTask = &g_astTasks_top[1] }, // Task ID or composite task information
		0, // Processor ID
		1, // Processor local ID
	},
	{	TASK_TYPE_COMPUTATIONAL, // Task type
		{ .pstTask = &g_astTasks_top[0] }, // Task ID or composite task information
		0, // Processor ID
		0, // Processor local ID
	},
	{	TASK_TYPE_COMPUTATIONAL, // Task type
		{ .pstTask = &g_astTasks_top[2] }, // Task ID or composite task information
		0, // Processor ID
		2, // Processor local ID
	},
};

SMappedGeneralTaskInfo g_astGeneralTaskMappingInfo[] = {
	{	TASK_TYPE_COMPUTATIONAL, // Task type
		&g_astTasks_top[1], // Task ID or composite task information
		0, // Processor ID
		1, // Processor local ID
	},
	{	TASK_TYPE_COMPUTATIONAL, // Task type
		&g_astTasks_top[0], // Task ID or composite task information
		0, // Processor ID
		0, // Processor local ID
	},
	{	TASK_TYPE_COMPUTATIONAL, // Task type
		&g_astTasks_top[2], // Task ID or composite task information
		0, // Processor ID
		2, // Processor local ID
	},
};


SMappedCompositeTaskInfo g_astCompositeTaskMappingInfo[] = {
};


SMappedTaskInfo g_stMappingInfo = {
	g_astGeneralTaskMappingInfo, // general task array
	ARRAYLEN(g_astGeneralTaskMappingInfo), // size of general task array
	NULL, // composite task array
	0, // size of composite task array
};

// ##MAPPING_SCHEDULING_INFO_TEMPLATE::END


// ##LIBRARY_INFO_TEMPLATE::START
SLibrary g_stLibraryInfo[] = {
	{
		"dct",
		l_dct_init,
		l_dct_wrapup,
	},
	{
		"idct",
		l_idct_init,
		l_idct_wrapup,
	},
};

// ##LIBRARY_INFO_TEMPLATE::END


int g_nChannelNum = ARRAYLEN(g_astChannels);
int g_nNumOfTasks_top = ARRAYLEN(g_astTasks_top);
int g_nTaskIdToTaskNum = ARRAYLEN(g_astTaskIdToTask);
int g_nProcessorInfoNum = ARRAYLEN(g_astProcessorInfo);
int g_nMappingAndSchedulingInfoNum = ARRAYLEN(g_astMappingAndSchedulingInfo);
int g_nLibraryInfoNum = ARRAYLEN(g_stLibraryInfo);

