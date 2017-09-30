/*
 * uem_data.c
 *
 *  Created on: 2017. 9. 7.
 *      Author: jej
 */

#include <uem_data.h>

// ##TASK_CODE_DECLARATION_SECTION:START
void MatA_Init(int nTaskId);
void MatA_Go();
void MatA_Wrapup();

void MatB_Init(int nTaskId);
void MatB_Go();
void MatB_Wrapup();

void VecMul_Init(int nTaskId);
void VecMul_Go();
void VecMul_Wrapup();

void Display_Init(int nTaskId);
void Display_Go();
void Display_Wrapup();
// ##TASK_CODE_DECLARATION_SECTION::END

// ##UEM_DATA_GENERATION_SECTION::START
#define CHANNEL_0_SIZE (3)
#define CHANNEL_1_SIZE (3)
#define CHANNEL_2_SIZE (36)

char s_pChannel_0_buffer[CHANNEL_0_SIZE];
char s_pChannel_1_buffer[CHANNEL_1_SIZE];
char s_pChannel_2_buffer[CHANNEL_2_SIZE];

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
		0, // Available data number (for broadcast loop)
	},
};

SChunk g_astChunk_channel_1_out[] = {
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // wWitten data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_1_in[] = {
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number (for broadcast loop)
	},
};

SChunk g_astChunk_channel_2_out[] = {
	{
		s_pChannel_2_buffer, // Chunk start pointer
		s_pChannel_2_buffer, // Data start pointer
		s_pChannel_2_buffer, // Data end pointer
		0, // wWitten data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_2_in[] = {
	{
		s_pChannel_2_buffer, // Chunk start pointer
		s_pChannel_2_buffer, // Data start pointer
		s_pChannel_2_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number (for broadcast loop)
	},
};

SPortSampleRate g_astPortSampleRate_MatA_out[] = {
	{ "Default", // Mode name
	   1, // Sample rate
	   1, // Available number of data
	},
};


SPortSampleRate g_astPortSampleRate_VecMul_in1[] = {
	{ "Default", // Mode name
	   1, // Sample rate
	   1, // Available number of data
	},
};


SPortSampleRate g_astPortSampleRate_MatB_out[] = {
	{ "Default", // Mode name
	   1, // Sample rate
	   1, // Available number of data
	},
};


SPortSampleRate g_astPortSampleRate_VecMul_in2[] = {
	{ "Default", // Mode name
	   1, // Sample rate
	   1, // Available number of data
	},
};


SPortSampleRate g_astPortSampleRate_VecMul_out[] = {
	{ "Default", // Mode name
	   1, // Sample rate
	   1, // Available number of data
	},
};


SPortSampleRate g_astPortSampleRate_Display_in[] = {
	{ "Default", // Mode name
	   9, // Sample rate
	   1, // Available number of data
	},
};

SAvailableChunk g_astAvailableInputChunk_channel_0[] = {
	{ 0, 0, NULL, NULL, },
};

SAvailableChunk g_astAvailableInputChunk_channel_1[] = {
	{ 0, 0, NULL, NULL, },
};

SAvailableChunk g_astAvailableInputChunk_channel_2[] = {
	{ 0, 0, NULL, NULL, },
};

// ports which located inside a subgraph and port-mapped to outer graph
//SPort g_astPortMapList[] = {
//	// all port information which are port-mapped by other ports
//};


SChannel g_astChannels[] = {
	{
		0, // Channel ID
		COMMUNICATION_TYPE_SHARED_MEMORY, // Communication type
		CHANNEL_TYPE_GENERAL, // Channel type
		s_pChannel_0_buffer, // Channel buffer pointer
		CHANNEL_0_SIZE, // Channel size
		s_pChannel_0_buffer, // Channel data start
		s_pChannel_0_buffer, // Channel data end
		0, // Channel data length
		0, // reference count
		NULL, // Mutex
		NULL, // Event
		{
			2, // Task ID
			"in1", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_VecMul_in1, // Array of sample rate list
			1, // Array element number of sample rate list
			0, // Selected sample rate index
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
			0, // Selected sample rate index
			12, // Sample size
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
		1, // Maximum chunk size
		NULL, // Chunk list head
		NULL, // Chunk list tail
	},
	{
		1, // Channel ID
		COMMUNICATION_TYPE_SHARED_MEMORY, // Communication type
		CHANNEL_TYPE_GENERAL, // Channel type
		s_pChannel_1_buffer, // Channel buffer pointer
		CHANNEL_1_SIZE, // Channel size
		s_pChannel_1_buffer, // Channel data start
		s_pChannel_1_buffer, // Channel data end
		0, // Channel data length
		0, // reference count
		NULL, // Mutex
		NULL, // Event
		{
			2, // Task ID
			"in2", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_VecMul_in2, // Array of sample rate list
			1, // Array element number of sample rate list
			0, // Selected sample rate index
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
			0, // Selected sample rate index
			12, // Sample size
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
		1, // Maximum chunk size
		NULL, // Chunk list head
		NULL, // Chunk list tail
	},
	{
		2, // Channel ID
		COMMUNICATION_TYPE_SHARED_MEMORY, // Communication type
		CHANNEL_TYPE_GENERAL, // Channel type
		s_pChannel_2_buffer, // Channel buffer pointer
		CHANNEL_2_SIZE, // Channel size
		s_pChannel_2_buffer, // Channel data start
		s_pChannel_2_buffer, // Channel data end
		0, // Channel data length
		0, // reference count
		NULL, // Mutex
		NULL, // Event
		{
			3, // Task ID
			"in", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Display_in, // Array of sample rate list
			1, // Array element number of sample rate list
			0, // Selected sample rate index
			36, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			2, // Task ID
			"out", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_VecMul_out, // Array of sample rate list
			1, // Array element number of sample rate list
			0, // Selected sample rate index
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
		1, // Maximum chunk size
		NULL, // Available Chunk list head
		NULL, // Available Chunk list tail
	},
};

STask g_astTasks_top[] = {
	{ 	0, // Task ID
		"MatA", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		MatA_Init, // Task init function
		MatA_Go, // Task go function
		MatA_Wrapup, // Task wrapup function
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
		MatB_Init, // Task init function
		MatB_Go, // Task go function
		MatB_Wrapup, // Task wrapup function
		RUN_CONDITION_TIME_DRIVEN, // Run condition
		1, // Run rate
		10, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // subgraph
		&g_stGraph_top, //parent task graph
		NULL, // MTM information
		NULL, // Loop information
		NULL, // Task parameter information
		FALSE, // statically scheduled (run by different tasks)
		NULL, // mutex
		NULL, // conditional variable
	},
	{ 	2, // Task ID
		"VecMul", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		VecMul_Init, // Task init function
		VecMul_Go, // Task go function
		VecMul_Wrapup, // Task wrapup function
		RUN_CONDITION_DATA_DRIVEN, // Run condition
		1, // Run rate
		10, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // subgraph
		&g_stGraph_top, //parent task graph
		NULL, // MTM information
		NULL, // Loop information
		NULL, // Task parameter information
		FALSE, // statically scheduled (run by different tasks)
		NULL, // mutex
		NULL, // conditional variable
	},
	{ 	3, // Task ID
		"Display", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		Display_Init, // Task init function
		Display_Go, // Task go function
		Display_Wrapup, // Task wrapup function
		RUN_CONDITION_DATA_DRIVEN, // Run condition
		1, // Run rate
		10, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // subgraph
		&g_stGraph_top, //parent task graph
		NULL, // MTM information
		NULL, // Loop information
		NULL, // Task parameter information
		FALSE, // statically scheduled (run by different tasks)
		NULL, // mutex
		NULL, // conditional variable
	},
};

STaskGraph g_stGraph_top = {
		GRAPH_TYPE_DATAFLOW, // Task graph type
		g_astTasks_top, // Current task graph's task list
		NULL, // Parent task
};

STaskIdToTaskMap g_astTaskIdToTask[] = {
	{ 	0, // Task ID
		&g_astTasks_top[0], // Task structure pointer
	},
	{ 	1, // Task ID
		&g_astTasks_top[1], // Task structure pointer
	},
	{ 	2, // Task ID
		&g_astTasks_top[2], // Task structure pointer
	},
};

SProcessor g_astProcessorInfo[] = {
	{ 	0, // Processor ID
		TRUE, // Processor is CPU?
		"i7_0", // Processor name
		4, // Processor pool size
	},
};

SMappingSchedulingInfo g_astMappingAndSchedulingInfo[] = {
	{	TASK_TYPE_COMPUTATIONAL, // Task type
		{ .nTaskId = 0 }, // Task ID or composite task information
		0, // Processor ID
		0, // Processor local ID
	},
	{	TASK_TYPE_COMPUTATIONAL, // Task type
		{ .nTaskId = 1 }, // Task ID or composite task information
		0, // Processor ID
		1, // Processor local ID
	},
	{	TASK_TYPE_COMPUTATIONAL, // Task type
		{ .nTaskId = 2 }, // Task ID or composite task information
		0, // Processor ID
		2, // Processor local ID
	},
	{	TASK_TYPE_COMPUTATIONAL, // Task type
		{ .nTaskId = 3 }, // Task ID or composite task information
		0, // Processor ID
		3, // Processor local ID
	},
};
// ##UEM_DATA_GENERATION_SECTION::END

int g_nChannelNum = ARRAYLEN(g_astChannels);
int g_nNumOfTasks_top = ARRAYLEN(g_astTasks_top);
int g_nTaskIdToTaskNum = ARRAYLEN(g_astTaskIdToTask);
int g_nProcessorInfoNum = ARRAYLEN(g_astProcessorInfo);
int g_nMappingAndSchedulingInfoNum = ARRAYLEN(g_astMappingAndSchedulingInfo);



