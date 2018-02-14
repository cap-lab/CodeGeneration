/* uem_data.c made by UEM Translator */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_data.h>
#include <UKTask.h>
#include <UKModeTransition.h>

SExecutionTime g_stExecutionTime = { 20, TIME_METRIC_SEC } ;

// ##TASK_CODE_TEMPLATE::START
void Counter_Receiver_Init0(int nTaskId);
void Counter_Receiver_Go0(int nTaskId);
void Counter_Receiver_Wrapup0();

void PingPong_Ping_Init0(int nTaskId);
void PingPong_Ping_Go0(int nTaskId);
void PingPong_Ping_Wrapup0();

void Control_Init0(int nTaskId);
void Control_Go0(int nTaskId);
void Control_Wrapup0();



void UserInput_Init0(int nTaskId);
void UserInput_Go0(int nTaskId);
void UserInput_Wrapup0();

void Counter_Sender_Init0(int nTaskId);
void Counter_Sender_Go0(int nTaskId);
void Counter_Sender_Wrapup0();

void PingPong_Pong_Init0(int nTaskId);
void PingPong_Pong_Go0(int nTaskId);
void PingPong_Pong_Wrapup0();

// ##TASK_CODE_TEMPLATE::END

// ##CHANNEL_SIZE_DEFINITION_TEMPLATE::START
#define CHANNEL_0_SIZE (16)
#define CHANNEL_1_SIZE (16)
#define CHANNEL_2_SIZE (40)
#define CHANNEL_3_SIZE (4)
// ##CHANNEL_SIZE_DEFINITION_TEMPLATE::END

// ##CHANNEL_BUFFER_DEFINITION_TEMPLATE::START
char s_pChannel_0_buffer[CHANNEL_0_SIZE];
char s_pChannel_1_buffer[CHANNEL_1_SIZE];
char s_pChannel_2_buffer[CHANNEL_2_SIZE];
char s_pChannel_3_buffer[CHANNEL_3_SIZE];
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

// ##CHUNK_DEFINITION_TEMPLATE::END
//portSampleRateList


// ##PORT_SAMPLE_RATE_TEMPLATE::START
SPortSampleRate g_astPortSampleRate_PingPong_Pong_input[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_PingPong_Ping_output[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_PingPong_Ping_input[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_PingPong_Pong_output[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Counter_Receiver_input[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Counter_Sender_output[] = {
	{ 	"Inc_Rate_1", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
	{ 	"Dec_Rate_10", // Mode name
		10, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Control_input[] = {
};

SPortSampleRate g_astPortSampleRate_UserInput_output[] = {
};

// ##PORT_SAMPLE_RATE_TEMPLATE::END


// ##PORT_ARRAY_TEMPLATE::START
SPort g_astPortInfo[] = {
	{
		3, // Task ID
		"input", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_PingPong_Pong_input, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
		
	{
		2, // Task ID
		"output", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_PingPong_Ping_output, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
		
	{
		2, // Task ID
		"input", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_PingPong_Ping_input, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
		
	{
		3, // Task ID
		"output", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_PingPong_Pong_output, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
		
	{
		6, // Task ID
		"input", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Counter_Receiver_input, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
		
	{
		5, // Task ID
		"output", // Port name
		PORT_SAMPLE_RATE_MULTIPLE, // Port sample rate type
		g_astPortSampleRate_Counter_Sender_output, // Array of sample rate list
		2, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
		
	{
		7, // Task ID
		"input", // Port name
		PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
		g_astPortSampleRate_Control_input, // Array of sample rate list
		0, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
		
	{
		0, // Task ID
		"output", // Port name
		PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
		g_astPortSampleRate_UserInput_output, // Array of sample rate list
		0, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
		
};
// ##PORT_ARRAY_TEMPLATE::END


// ##LOOP_STRUCTURE_TEMPLATE::START
// ##LOOP_STRUCTURE_TEMPLATE::END

// ##TASK_LIST_DECLARATION_TEMPLATE::START
STask g_astTasks_Counter[];
STask g_astTasks_top[];
STask g_astTasks_PingPong[];
// ##TASK_LIST_DECLARATION_TEMPLATE::END


// ##TASK_GRAPH_DECLARATION_TEMPLATE::START
STaskGraph g_stGraph_Counter;
STaskGraph g_stGraph_top;
STaskGraph g_stGraph_PingPong;
// ##TASK_GRAPH_DECLARATION_TEMPLATE::END


// ##MODE_TRANSITION_TEMPLATE::START
STask *g_pastRelatedChildTasks_Counter_Inc_Rate_1[] = {
	&g_astTasks_Counter[1],
	&g_astTasks_Counter[0],
};
STask *g_pastRelatedChildTasks_Counter_Dec_Rate_10[] = {
	&g_astTasks_Counter[1],
	&g_astTasks_Counter[0],
};
		
SModeMap g_astModeMap_Counter[] = {
	{
		0,
		"Inc_Rate_1",
		g_pastRelatedChildTasks_Counter_Inc_Rate_1,
		2,
	},
	{
		1,
		"Dec_Rate_10",
		g_pastRelatedChildTasks_Counter_Dec_Rate_10,
		2,
	},
};

SVariableIntMap g_astVariableIntMap_Counter[] = {
	{
		0,
		"Var",
		0, 
	},
};

static uem_bool transitMode_Counter(SModeTransitionMachine *pstModeTransition) 
{
	uem_bool bModeChanged = FALSE;
	int Var;
	int nCurrentModeId = pstModeTransition->astModeMap[pstModeTransition->nCurModeIndex].nModeId;
	int nNextModeId = nCurrentModeId;
	int nVarIndex = 0;
	
	nVarIndex = UKModeTransition_GetVariableIndexByName(pstModeTransition, "Var");
	Var = pstModeTransition->astVarIntMap[nVarIndex].nValue;
		
	if(nCurrentModeId == 0
	  && Var == 1 )
	{
		nNextModeId = 1;
		bModeChanged = TRUE;
	}
	if(nCurrentModeId == 1
	  && Var == 0 )
	{
		nNextModeId = 0;
		bModeChanged = TRUE;
	}
		
	pstModeTransition->nNextModeIndex = UKModeTransition_GetModeIndexByModeId(pstModeTransition, nNextModeId);
	
	return bModeChanged;
}

SModeTransitionMachine g_stModeTransition_Counter = {
	4,
	g_astModeMap_Counter, // mode list
	2, // number of modes
	g_astVariableIntMap_Counter, // Integer variable list
	1, // number of integer variables
	transitMode_Counter, // mode transition function
	0, // Current mode index
	0, // Next mode index
};
STask *g_pastRelatedChildTasks_PingPong_Default[] = {
	&g_astTasks_PingPong[1],
	&g_astTasks_PingPong[0],
};
		
SModeMap g_astModeMap_PingPong[] = {
	{
		0,
		"Default",
		g_pastRelatedChildTasks_PingPong_Default,
		2,
	},
};

SVariableIntMap g_astVariableIntMap_PingPong[] = {
};


SModeTransitionMachine g_stModeTransition_PingPong = {
	1,
	g_astModeMap_PingPong, // mode list
	1, // number of modes
	g_astVariableIntMap_PingPong, // Integer variable list
	0, // number of integer variables
	NULL, // mode transition function
	0, // Current mode index
	0, // Next mode index
};
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
// ##AVAILABLE_CHUNK_LIST_TEMPLATE::END

// ##TASK_PARAMETER_TEMPLATE::START
STaskParameter g_astTaskParameter_PingPong[] = {
	{
		0,
		PARAMETER_TYPE_INT,
		"Speed",
		{ .nParam = 1, },
	},
};
// ##TASK_PARAMETER_TEMPLATE::END

// ##TASK_FUNCTION_LIST::START
STaskFunctions g_ast_Counter_Receiver_functions[] = {
	{
		Counter_Receiver_Init0, // Task init function
		Counter_Receiver_Go0, // Task go function
		Counter_Receiver_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_PingPong_Ping_functions[] = {
	{
		PingPong_Ping_Init0, // Task init function
		PingPong_Ping_Go0, // Task go function
		PingPong_Ping_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_Control_functions[] = {
	{
		Control_Init0, // Task init function
		Control_Go0, // Task go function
		Control_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_Counter_functions[] = {
};

STaskFunctions g_ast_PingPong_functions[] = {
};

STaskFunctions g_ast_UserInput_functions[] = {
	{
		UserInput_Init0, // Task init function
		UserInput_Go0, // Task go function
		UserInput_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_Counter_Sender_functions[] = {
	{
		Counter_Sender_Init0, // Task init function
		Counter_Sender_Go0, // Task go function
		Counter_Sender_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_PingPong_Pong_functions[] = {
	{
		PingPong_Pong_Init0, // Task init function
		PingPong_Pong_Go0, // Task go function
		PingPong_Pong_Wrapup0, // Task wrapup function
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
			3, // Task ID
			"input", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_PingPong_Pong_input, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			2, // Task ID
			"output", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_PingPong_Ping_output, // Array of sample rate list
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
			2, // Task ID
			"input", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_PingPong_Ping_input, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			3, // Task ID
			"output", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_PingPong_Pong_output, // Array of sample rate list
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
		4, // Initial data length 
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
			6, // Task ID
			"input", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Counter_Receiver_input, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			5, // Task ID
			"output", // Port name
			PORT_SAMPLE_RATE_MULTIPLE, // Port sample rate type
			g_astPortSampleRate_Counter_Sender_output, // Array of sample rate list
			2, // Array element number of sample rate list
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
			7, // Task ID
			"input", // Port name
			PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
			g_astPortSampleRate_Control_input, // Array of sample rate list
			0, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			0, // Task ID
			"output", // Port name
			PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
			g_astPortSampleRate_UserInput_output, // Array of sample rate list
			0, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
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
};
// ##CHANNEL_LIST_TEMPLATE::END


// ##TASK_LIST_TEMPLATE::START
STask g_astTasks_Counter[] = {
	{ 	6, // Task ID
		"Counter_Receiver", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_Counter_Receiver_functions, // Task function array
		1, // Task function array number
		RUN_CONDITION_DATA_DRIVEN, // Run condition
		1, // Run rate
		1, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // Subgraph
		&g_stGraph_Counter, // Parent task graph
		NULL, // MTM information
		NULL, // Loop information
		NULL, // Task parameter information
		0, // Task parameter number
		TRUE, // Statically scheduled or not
		0,	  // Throughput constraint
		NULL, // Mutex
		NULL, // Conditional variable
	},
	{ 	5, // Task ID
		"Counter_Sender", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_Counter_Sender_functions, // Task function array
		1, // Task function array number
		RUN_CONDITION_TIME_DRIVEN, // Run condition
		1, // Run rate
		1, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // Subgraph
		&g_stGraph_Counter, // Parent task graph
		NULL, // MTM information
		NULL, // Loop information
		NULL, // Task parameter information
		0, // Task parameter number
		TRUE, // Statically scheduled or not
		0,	  // Throughput constraint
		NULL, // Mutex
		NULL, // Conditional variable
	},
};

STask g_astTasks_top[] = {
	{ 	4, // Task ID
		"Counter", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_Counter_functions, // Task function array
		0, // Task function array number
		RUN_CONDITION_TIME_DRIVEN, // Run condition
		1, // Run rate
		1, // Period
		TIME_METRIC_MICROSEC, // Period metric
		&g_stGraph_Counter, // Subgraph
		&g_stGraph_top, // Parent task graph
		&g_stModeTransition_Counter, // MTM information
		NULL, // Loop information
		NULL, // Task parameter information
		0, // Task parameter number
		TRUE, // Statically scheduled or not
		0,	  // Throughput constraint
		NULL, // Mutex
		NULL, // Conditional variable
	},
	{ 	1, // Task ID
		"PingPong", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_PingPong_functions, // Task function array
		0, // Task function array number
		RUN_CONDITION_TIME_DRIVEN, // Run condition
		1, // Run rate
		1, // Period
		TIME_METRIC_MICROSEC, // Period metric
		&g_stGraph_PingPong, // Subgraph
		&g_stGraph_top, // Parent task graph
		&g_stModeTransition_PingPong, // MTM information
		NULL, // Loop information
		g_astTaskParameter_PingPong, // Task parameter information
		1, // Task parameter number
		TRUE, // Statically scheduled or not
		0,	  // Throughput constraint
		NULL, // Mutex
		NULL, // Conditional variable
	},
	{ 	0, // Task ID
		"UserInput", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_UserInput_functions, // Task function array
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
	{ 	7, // Task ID
		"Control", // Task name
		TASK_TYPE_CONTROL, // Task Type
		g_ast_Control_functions, // Task function array
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

STask g_astTasks_PingPong[] = {
	{ 	2, // Task ID
		"PingPong_Ping", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_PingPong_Ping_functions, // Task function array
		1, // Task function array number
		RUN_CONDITION_TIME_DRIVEN, // Run condition
		1, // Run rate
		1, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // Subgraph
		&g_stGraph_PingPong, // Parent task graph
		NULL, // MTM information
		NULL, // Loop information
		NULL, // Task parameter information
		0, // Task parameter number
		TRUE, // Statically scheduled or not
		0,	  // Throughput constraint
		NULL, // Mutex
		NULL, // Conditional variable
	},
	{ 	3, // Task ID
		"PingPong_Pong", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_PingPong_Pong_functions, // Task function array
		1, // Task function array number
		RUN_CONDITION_DATA_DRIVEN, // Run condition
		1, // Run rate
		1, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // Subgraph
		&g_stGraph_PingPong, // Parent task graph
		NULL, // MTM information
		NULL, // Loop information
		NULL, // Task parameter information
		0, // Task parameter number
		TRUE, // Statically scheduled or not
		0,	  // Throughput constraint
		NULL, // Mutex
		NULL, // Conditional variable
	},
};


// ##TASK_LIST_TEMPLATE::END

// ##TASK_GRAPH_TEMPLATE::START
STaskGraph g_stGraph_Counter = {
		GRAPH_TYPE_PROCESS_NETWORK, // TODO: Task graph type (not used now)
		g_astTasks_Counter, // current task graph's task list
		2, // number of tasks
		&g_astTasks_top[0], // parent task
};

STaskGraph g_stGraph_top = {
		GRAPH_TYPE_PROCESS_NETWORK, // TODO: Task graph type (not used now)
		g_astTasks_top, // current task graph's task list
		4, // number of tasks
		NULL, // parent task
};

STaskGraph g_stGraph_PingPong = {
		GRAPH_TYPE_PROCESS_NETWORK, // TODO: Task graph type (not used now)
		g_astTasks_PingPong, // current task graph's task list
		2, // number of tasks
		&g_astTasks_top[1], // parent task
};

// ##TASK_GRAPH_TEMPLATE::END

// ##TASK_ID_TO_TASK_MAP_TEMPLATE::START
STaskIdToTaskMap g_astTaskIdToTask[] = {
	{ 	6, // Task ID
		"Counter_Receiver", // Task name
		&g_astTasks_Counter[0], // Task structure pointer
	},
	{ 	2, // Task ID
		"PingPong_Ping", // Task name
		&g_astTasks_PingPong[0], // Task structure pointer
	},
	{ 	7, // Task ID
		"Control", // Task name
		&g_astTasks_top[3], // Task structure pointer
	},
	{ 	4, // Task ID
		"Counter", // Task name
		&g_astTasks_top[0], // Task structure pointer
	},
	{ 	1, // Task ID
		"PingPong", // Task name
		&g_astTasks_top[1], // Task structure pointer
	},
	{ 	0, // Task ID
		"UserInput", // Task name
		&g_astTasks_top[2], // Task structure pointer
	},
	{ 	5, // Task ID
		"Counter_Sender", // Task name
		&g_astTasks_Counter[1], // Task structure pointer
	},
	{ 	3, // Task ID
		"PingPong_Pong", // Task name
		&g_astTasks_PingPong[1], // Task structure pointer
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
void Counter_1_0_0_0_Go(int nTaskId) 
{
	Counter_Receiver_Go0(6);
	Counter_Receiver_Go0(6);
	Counter_Receiver_Go0(6);
	Counter_Receiver_Go0(6);
	Counter_Receiver_Go0(6);
	Counter_Receiver_Go0(6);
	Counter_Receiver_Go0(6);
	Counter_Receiver_Go0(6);
	Counter_Receiver_Go0(6);
	Counter_Receiver_Go0(6);
}

void Counter_1_0_2_0_Go(int nTaskId) 
{
	Counter_Sender_Go0(5);
	{
		uem_bool bTransition = FALSE;
		uem_result result;
		STask *pstTask = NULL;
		result = UKTask_GetTaskFromTaskId(nTaskId, &pstTask);
		if(result == ERR_UEM_NOERROR)
		{
			result = UCThreadMutex_Lock(pstTask->hMutex);
			if(result == ERR_UEM_NOERROR){
				bTransition = transitMode_Counter(g_astTasks_top[0].pstMTMInfo);
				UCThreadMutex_Unlock(pstTask->hMutex);
			}
			
			if(bTransition == TRUE) return; // exit when the transition is changed.
		}
	}
	{
		EInternalTaskState enState = INTERNAL_STATE_STOP;
		UKTask_GetTaskState("Counter", &enState);
		if(enState == INTERNAL_STATE_STOP || enState == INTERNAL_STATE_END) return; 
	}
}

void Counter_0_0_0_0_Go(int nTaskId) 
{
	Counter_Receiver_Go0(6);
}

void Counter_0_0_2_0_Go(int nTaskId) 
{
	Counter_Sender_Go0(5);
	{
		uem_bool bTransition = FALSE;
		uem_result result;
		STask *pstTask = NULL;
		result = UKTask_GetTaskFromTaskId(nTaskId, &pstTask);
		if(result == ERR_UEM_NOERROR)
		{
			result = UCThreadMutex_Lock(pstTask->hMutex);
			if(result == ERR_UEM_NOERROR){
				bTransition = transitMode_Counter(g_astTasks_top[0].pstMTMInfo);
				UCThreadMutex_Unlock(pstTask->hMutex);
			}
			
			if(bTransition == TRUE) return; // exit when the transition is changed.
		}
	}
	{
		EInternalTaskState enState = INTERNAL_STATE_STOP;
		UKTask_GetTaskState("Counter", &enState);
		if(enState == INTERNAL_STATE_STOP || enState == INTERNAL_STATE_END) return; 
	}
}

void PingPong_0_0_0_0_Go(int nTaskId) 
{
	PingPong_Ping_Go0(2);
	PingPong_Pong_Go0(3);
}

// ##SCHEDULED_COMPOSITE_TASK_FUNCTION_IMPLEMENTATION::END

// ##SCHEDULED_COMPOSITE_TASKS_TEMPLATE::START
SScheduleList g_astScheduleList_Counter_1_0_0[] = {
	{
		Counter_1_0_0_0_Go, // Composite GO function
		0, // Throughput constraint
	},
};
SScheduleList g_astScheduleList_Counter_1_0_2[] = {
	{
		Counter_1_0_2_0_Go, // Composite GO function
		0, // Throughput constraint
	},
};
SScheduleList g_astScheduleList_Counter_0_0_0[] = {
	{
		Counter_0_0_0_0_Go, // Composite GO function
		0, // Throughput constraint
	},
};
SScheduleList g_astScheduleList_Counter_0_0_2[] = {
	{
		Counter_0_0_2_0_Go, // Composite GO function
		0, // Throughput constraint
	},
};
SScheduleList g_astScheduleList_PingPong_0_0_0[] = {
	{
		PingPong_0_0_0_0_Go, // Composite GO function
		0, // Throughput constraint
	},
};
// ##SCHEDULED_COMPOSITE_TASKS_TEMPLATE::END



SScheduledTasks g_astScheduledTaskList[] = {
	{	&g_astTasks_top[0], // Parent Task ID
		1, // Mode transition mode ID
		g_astScheduleList_Counter_1_0_0, // schedule list per throughput constraint
		1, // The number of schedules in the schedule list
		0, // Schedule Index (Default to set 0)
		0, // Mode Sequence ID 
	},
	{	&g_astTasks_top[0], // Parent Task ID
		1, // Mode transition mode ID
		g_astScheduleList_Counter_1_0_2, // schedule list per throughput constraint
		1, // The number of schedules in the schedule list
		0, // Schedule Index (Default to set 0)
		1, // Mode Sequence ID 
	},
	{	&g_astTasks_top[0], // Parent Task ID
		0, // Mode transition mode ID
		g_astScheduleList_Counter_0_0_0, // schedule list per throughput constraint
		1, // The number of schedules in the schedule list
		0, // Schedule Index (Default to set 0)
		0, // Mode Sequence ID 
	},
	{	&g_astTasks_top[0], // Parent Task ID
		0, // Mode transition mode ID
		g_astScheduleList_Counter_0_0_2, // schedule list per throughput constraint
		1, // The number of schedules in the schedule list
		0, // Schedule Index (Default to set 0)
		1, // Mode Sequence ID 
	},
	{	&g_astTasks_top[1], // Parent Task ID
		0, // Mode transition mode ID
		g_astScheduleList_PingPong_0_0_0, // schedule list per throughput constraint
		1, // The number of schedules in the schedule list
		0, // Schedule Index (Default to set 0)
		0, // Mode Sequence ID 
	},
};


// ##MAPPING_SCHEDULING_INFO_TEMPLATE::START
SMappingSchedulingInfo g_astMappingAndSchedulingInfo[] = {
	{	TASK_TYPE_CONTROL, // Task type
		{ .pstTask = &g_astTasks_top[3] }, // Task ID or composite task information
		0, // Processor ID
		1, // Processor local ID
	},
	{	TASK_TYPE_COMPUTATIONAL, // Task type
		{ .pstTask = &g_astTasks_top[2] }, // Task ID or composite task information
		0, // Processor ID
		0, // Processor local ID
	},
	{	TASK_TYPE_COMPOSITE, // Task type
		{ .pstScheduledTasks = &g_astScheduledTaskList[0] }, // Task ID or composite task information
		0, // Processor ID
		0, // Processor local ID
	},
	{	TASK_TYPE_COMPOSITE, // Task type
		{ .pstScheduledTasks = &g_astScheduledTaskList[1] }, // Task ID or composite task information
		0, // Processor ID
		2, // Processor local ID
	},
	{	TASK_TYPE_COMPOSITE, // Task type
		{ .pstScheduledTasks = &g_astScheduledTaskList[2] }, // Task ID or composite task information
		0, // Processor ID
		0, // Processor local ID
	},
	{	TASK_TYPE_COMPOSITE, // Task type
		{ .pstScheduledTasks = &g_astScheduledTaskList[3] }, // Task ID or composite task information
		0, // Processor ID
		2, // Processor local ID
	},
	{	TASK_TYPE_COMPOSITE, // Task type
		{ .pstScheduledTasks = &g_astScheduledTaskList[4] }, // Task ID or composite task information
		0, // Processor ID
		0, // Processor local ID
	},
};

SMappedGeneralTaskInfo g_astGeneralTaskMappingInfo[] = {
	{	TASK_TYPE_CONTROL, // Task type
		&g_astTasks_top[3], // Task ID or composite task information
		0, // Processor ID
		1, // Processor local ID
	},
	{	TASK_TYPE_COMPUTATIONAL, // Task type
		&g_astTasks_top[2], // Task ID or composite task information
		0, // Processor ID
		0, // Processor local ID
	},
};


SMappedCompositeTaskInfo g_astCompositeTaskMappingInfo[] = {
	{
		&g_astScheduledTaskList[0],
		0, // Processor ID
		0, // Processor local ID		
	},
	{
		&g_astScheduledTaskList[1],
		0, // Processor ID
		2, // Processor local ID		
	},
	{
		&g_astScheduledTaskList[2],
		0, // Processor ID
		0, // Processor local ID		
	},
	{
		&g_astScheduledTaskList[3],
		0, // Processor ID
		2, // Processor local ID		
	},
	{
		&g_astScheduledTaskList[4],
		0, // Processor ID
		0, // Processor local ID		
	},
};


SMappedTaskInfo g_stMappingInfo = {
	g_astGeneralTaskMappingInfo, // general task array
	ARRAYLEN(g_astGeneralTaskMappingInfo), // size of general task array
	g_astCompositeTaskMappingInfo, // composite task array
	ARRAYLEN(g_astCompositeTaskMappingInfo), // size of composite task array
};

// ##MAPPING_SCHEDULING_INFO_TEMPLATE::END


int g_nChannelNum = ARRAYLEN(g_astChannels);
int g_nNumOfTasks_top = ARRAYLEN(g_astTasks_top);
int g_nTaskIdToTaskNum = ARRAYLEN(g_astTaskIdToTask);
int g_nProcessorInfoNum = ARRAYLEN(g_astProcessorInfo);
int g_nMappingAndSchedulingInfoNum = ARRAYLEN(g_astMappingAndSchedulingInfo);

