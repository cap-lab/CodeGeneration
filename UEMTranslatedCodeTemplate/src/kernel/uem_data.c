/* uem_data.c made by UEM Translator */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_data.h>

SExecutionTime g_stExecutionTime = { 90, TIME_METRIC_MICROSEC } ;

// ##TASK_CODE_TEMPLATE::START
void MatA_Init0(int nTaskId);
void MatA_Go0();
void MatA_Wrapup0();

void MatB_Init0(int nTaskId);
void MatB_Go0();
void MatB_Wrapup0();

void VecMul_Init0(int nTaskId);
void VecMul_Go0();
void VecMul_Wrapup0();

void Display_Init0(int nTaskId);
void Display_Go0();
void Display_Wrapup0();

// ##TASK_CODE_TEMPLATE::END

// ##CHANNEL_SIZE_DEFINITION_TEMPLATE::START
#define CHANNEL_0_SIZE (12)
#define CHANNEL_1_SIZE (12)
#define CHANNEL_2_SIZE (36)
// ##CHANNEL_SIZE_DEFINITION_TEMPLATE::END

// ##CHANNEL_BUFFER_DEFINITION_TEMPLATE::START
char s_pChannel_0_buffer[CHANNEL_0_SIZE];
char s_pChannel_1_buffer[CHANNEL_1_SIZE];
char s_pChannel_2_buffer[CHANNEL_2_SIZE];
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

// ##CHUNK_DEFINITION_TEMPLATE::END
//portSampleRateList


// ##PORT_SAMPLE_RATE_TEMPLATE::START
SPortSampleRate g_astPortSampleRate_VecMul_in1[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_MatA_out[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_VecMul_in2[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_MatB_out[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Display_in[] = {
	{ 	"Default", // Mode name
		9, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_VecMul_out[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

// ##PORT_SAMPLE_RATE_TEMPLATE::END


// ##PORT_ARRAY_TEMPLATE::START
SPort g_astPortInfo[] = {
	{
		2, // Task ID
		"in1", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_VecMul_in1, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		12, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
		
	{
		0, // Task ID
		"out", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_MatA_out, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
		
	{
		2, // Task ID
		"in2", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_VecMul_in2, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		12, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
		
	{
		1, // Task ID
		"out", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_MatB_out, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
		
	{
		3, // Task ID
		"in", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Display_in, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
		
	{
		2, // Task ID
		"out", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_VecMul_out, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
		
};
// ##PORT_ARRAY_TEMPLATE::END


// ##LOOP_STRUCTURE_TEMPLATE::START
// ##LOOP_STRUCTURE_TEMPLATE::END

// ##VARIABLE_INT_MAP_TEMPLATE::START


// ##VARIABLE_INT_MAP_TEMPLATE::END


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
// ##AVAILABLE_CHUNK_LIST_TEMPLATE::END

// ##TASK_PARAMETER_TEMPLATE::START
// ##TASK_PARAMETER_TEMPLATE::END

// ##TASK_FUNCTION_LIST::START
STaskFunctions g_ast_MatA_functions[] = {
	{
		MatA_Init0, // Task init function
		MatA_Go0, // Task go function
		MatA_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_MatB_functions[] = {
	{
		MatB_Init0, // Task init function
		MatB_Go0, // Task go function
		MatB_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_VecMul_functions[] = {
	{
		VecMul_Init0, // Task init function
		VecMul_Go0, // Task go function
		VecMul_Wrapup0, // Task wrapup function
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
		FALSE, // Exit setting
		NULL, // Mutex
		NULL, // Read available notice event
		NULL, // Write available notice event
		{
			2, // Task ID
			"in1", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_VecMul_in1, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			12, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			0, // Task ID
			"out", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_MatA_out, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
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
		FALSE, // Exit setting
		NULL, // Mutex
		NULL, // Read available notice event
		NULL, // Write available notice event
		{
			2, // Task ID
			"in2", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_VecMul_in2, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			12, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			1, // Task ID
			"out", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_MatB_out, // Array of sample rate list
			1, // Array element number of sample rate list
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
		FALSE, // Exit setting
		NULL, // Mutex
		NULL, // Read available notice event
		NULL, // Write available notice event
		{
			3, // Task ID
			"in", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Display_in, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			2, // Task ID
			"out", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_VecMul_out, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
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
	},
};
// ##CHANNEL_LIST_TEMPLATE::END

// ##TASK_LIST_TEMPLATE::START
STask g_astTasks_top[] = {
	{ 	0, // Task ID
		"MatA", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_MatA_functions, // Task function array
		1, // Task function array number
		RUN_CONDITION_TIME_DRIVEN, // Run condition
		1, // Run rate
		10, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // Subgraph
		&g_stGraph_top, // Parent task graph
		NULL, // MTM information
		NULL, // Loop information
		NULL, // Task parameter information
		FALSE, // Statically scheduled or not
		NULL, // Mutex
		NULL, // Conditional variable
	},
	{ 	1, // Task ID
		"MatB", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_MatB_functions, // Task function array
		1, // Task function array number
		RUN_CONDITION_TIME_DRIVEN, // Run condition
		1, // Run rate
		10, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // Subgraph
		&g_stGraph_top, // Parent task graph
		NULL, // MTM information
		NULL, // Loop information
		NULL, // Task parameter information
		FALSE, // Statically scheduled or not
		NULL, // Mutex
		NULL, // Conditional variable
	},
	{ 	2, // Task ID
		"VecMul", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_VecMul_functions, // Task function array
		1, // Task function array number
		RUN_CONDITION_DATA_DRIVEN, // Run condition
		1, // Run rate
		10, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // Subgraph
		&g_stGraph_top, // Parent task graph
		NULL, // MTM information
		NULL, // Loop information
		NULL, // Task parameter information
		FALSE, // Statically scheduled or not
		NULL, // Mutex
		NULL, // Conditional variable
	},
	{ 	3, // Task ID
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
		FALSE, // Statically scheduled or not
		NULL, // Mutex
		NULL, // Conditional variable
	},
};

// ##TASK_LIST_TEMPLATE::END

// ##TASK_GRAPH_TEMPLATE::START
STaskGraph g_stGraph_top = {
		GRAPH_TYPE_PROCESS_NETWORK, // TODO: Task graph type (not used now)
		g_astTasks_top, // current task graph's task list
		NULL, // parent task
};
// ##TASK_GRAPH_TEMPLATE::END

// ##TASK_ID_TO_TASK_MAP_TEMPLATE::START
STaskIdToTaskMap g_astTaskIdToTask[] = {
	{ 	0, // Task ID
		"MatA", // Task name
		&g_astTasks_top[0], // Task structure pointer
	},
	{ 	1, // Task ID
		"MatB", // Task name
		&g_astTasks_top[1], // Task structure pointer
	},
	{ 	2, // Task ID
		"VecMul", // Task name
		&g_astTasks_top[2], // Task structure pointer
	},
	{ 	3, // Task ID
		"Display", // Task name
		&g_astTasks_top[3], // Task structure pointer
	},
};
// ##TASK_ID_TO_TASK_MAP_TEMPLATE::END


// ##PROCESSOR_INFO_TEMPLATE::START
SProcessor g_astProcessorInfo[] = {

	{ 	0, // Processor ID
		TRUE, // Processor is CPU?			
		"i7_0", // Processor name
		5, // Processor pool size
	},
};
// ##PROCESSOR_INFO_TEMPLATE::END



// ##SCHEDULED_COMPOSITE_TASK_FUNCTION_IMPLEMENTATION::START
void top_0_0_0_1_Go() 
{
	MatA_Go0();
	MatB_Go0();
	VecMul_Go0();
	MatA_Go0();
	MatB_Go0();
	VecMul_Go0();
	MatA_Go0();
	MatB_Go0();
	VecMul_Go0();
	MatA_Go0();
	MatB_Go0();
	VecMul_Go0();
	MatA_Go0();
	MatB_Go0();
	VecMul_Go0();
	MatA_Go0();
	MatB_Go0();
	VecMul_Go0();
	MatA_Go0();
	MatB_Go0();
	VecMul_Go0();
	MatA_Go0();
	MatB_Go0();
	VecMul_Go0();
	MatA_Go0();
	MatB_Go0();
	VecMul_Go0();
	Display_Go0();
}
// ##SCHEDULED_COMPOSITE_TASK_FUNCTION_IMPLEMENTATION::END

// ##SCHEDULED_COMPOSITE_TASKS_TEMPLATE::START
SScheduleList g_astScheduleList_top_0_0_0[] = {
	{
		1, // Schedule ID
		top_0_0_0_1_Go, // Composite GO function
		0, // Throughput constraint
	},
};
// ##SCHEDULED_COMPOSITE_TASKS_TEMPLATE::END



SScheduledTasks g_astScheduledTaskList[] = {
	{	NULL, // Parent Task ID
		0, // Mode transition mode ID
		g_astScheduleList_top_0_0_0, // schedule list per throught constraint
		1, // The number of schedules in the schedule list
		0, // Schedule Index (Default to set 0)
		0, // Mode Sequence ID 
	},
};


// ##MAPPING_SCHEDULING_INFO_TEMPLATE::START
SMappingSchedulingInfo g_astMappingAndSchedulingInfo[] = {
	{	TASK_TYPE_COMPOSITE, // Task type
		{ .pstScheduledTasks = &g_astScheduledTaskList[0] }, // Task ID or composite task information
		0, // Processor ID
		0, // Processor local ID
	},
};
// ##MAPPING_SCHEDULING_INFO_TEMPLATE::END


int g_nChannelNum = ARRAYLEN(g_astChannels);
int g_nNumOfTasks_top = ARRAYLEN(g_astTasks_top);
int g_nTaskIdToTaskNum = ARRAYLEN(g_astTaskIdToTask);
int g_nProcessorInfoNum = ARRAYLEN(g_astProcessorInfo);
int g_nMappingAndSchedulingInfoNum = ARRAYLEN(g_astMappingAndSchedulingInfo);



