/* uem_data.c made by UEM Translator */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_data.h>
#include <UKTask.h>
#include <UKModeTransition.h>

SExecutionTime g_stExecutionTime = { 10, TIME_METRIC_SEC } ;

// ##TASK_CODE_TEMPLATE::START
void Receiver_Decryption_Init0(int nTaskId);
void Receiver_Decryption_Go0(int nTaskId);
void Receiver_Decryption_Wrapup0();


void Control_Init0(int nTaskId);
void Control_Go0(int nTaskId);
void Control_Wrapup0();

void Receiver_Unpacking_Init0(int nTaskId);
void Receiver_Unpacking_Go0(int nTaskId);
void Receiver_Unpacking_Wrapup0();

void Sender_Encryption_Init0(int nTaskId);
void Sender_Encryption_Go0(int nTaskId);
void Sender_Encryption_Wrapup0();


void Sender_Transfer_Init0(int nTaskId);
void Sender_Transfer_Go0(int nTaskId);
void Sender_Transfer_Wrapup0();

void UserInput_Init0(int nTaskId);
void UserInput_Go0(int nTaskId);
void UserInput_Wrapup0();

void Receiver_Display_Init0(int nTaskId);
void Receiver_Display_Go0(int nTaskId);
void Receiver_Display_Wrapup0();

void Sender_Packing_Init0(int nTaskId);
void Sender_Packing_Go0(int nTaskId);
void Sender_Packing_Wrapup0();

void IncomingMsg_Init0(int nTaskId);
void IncomingMsg_Go0(int nTaskId);
void IncomingMsg_Wrapup0();

// ##TASK_CODE_TEMPLATE::END


// ##LIBRARY_INIT_WRAPUP_TEMPLATE::START
void l_CryptographyLibrary_init();
void l_CryptographyLibrary_wrapup();

// ##LIBRARY_INIT_WRAPUP_TEMPLATE::END


// ##CHANNEL_SIZE_DEFINITION_TEMPLATE::START
#define CHANNEL_0_SIZE (1024)
#define CHANNEL_1_SIZE (1024)
#define CHANNEL_2_SIZE (1024)
#define CHANNEL_3_SIZE (1024)
#define CHANNEL_4_SIZE (4)
#define CHANNEL_5_SIZE (4)
#define CHANNEL_6_SIZE (1024)
#define CHANNEL_7_SIZE (1024)
// ##CHANNEL_SIZE_DEFINITION_TEMPLATE::END

// ##CHANNEL_BUFFER_DEFINITION_TEMPLATE::START
char s_pChannel_0_buffer[CHANNEL_0_SIZE];
char s_pChannel_1_buffer[CHANNEL_1_SIZE];
char s_pChannel_2_buffer[CHANNEL_2_SIZE];
char s_pChannel_3_buffer[CHANNEL_3_SIZE];
char s_pChannel_4_buffer[CHANNEL_4_SIZE];
char s_pChannel_5_buffer[CHANNEL_5_SIZE];
char s_pChannel_6_buffer[CHANNEL_6_SIZE];
char s_pChannel_7_buffer[CHANNEL_7_SIZE];
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

SChunk g_astChunk_channel_5_out[] = {
	{
		s_pChannel_5_buffer, // Chunk start pointer
		s_pChannel_5_buffer, // Data start pointer
		s_pChannel_5_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_5_in[] = {
	{
		s_pChannel_5_buffer, // Chunk start pointer
		s_pChannel_5_buffer, // Data start pointer
		s_pChannel_5_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_6_out[] = {
	{
		s_pChannel_6_buffer, // Chunk start pointer
		s_pChannel_6_buffer, // Data start pointer
		s_pChannel_6_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_6_in[] = {
	{
		s_pChannel_6_buffer, // Chunk start pointer
		s_pChannel_6_buffer, // Data start pointer
		s_pChannel_6_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_7_out[] = {
	{
		s_pChannel_7_buffer, // Chunk start pointer
		s_pChannel_7_buffer, // Data start pointer
		s_pChannel_7_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_7_in[] = {
	{
		s_pChannel_7_buffer, // Chunk start pointer
		s_pChannel_7_buffer, // Data start pointer
		s_pChannel_7_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

// ##CHUNK_DEFINITION_TEMPLATE::END
//portSampleRateList


// ##PORT_SAMPLE_RATE_TEMPLATE::START
SPortSampleRate g_astPortSampleRate_Sender_Encryption_in[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Sender_Packing_out[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Sender_Transfer_in[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Sender_Encryption_out[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Receiver_Unpacking_in[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Receiver_Decryption_out[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Receiver_Display_in[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Receiver_Unpacking_out[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Control_inUser[] = {
};

SPortSampleRate g_astPortSampleRate_UserInput_outAck[] = {
};

SPortSampleRate g_astPortSampleRate_Control_inIncoming[] = {
};

SPortSampleRate g_astPortSampleRate_IncomingMsg_outAck[] = {
};

SPortSampleRate g_astPortSampleRate_Sender_Packing_in[] = {
};

SPortSampleRate g_astPortSampleRate_UserInput_outMsg[] = {
};

SPortSampleRate g_astPortSampleRate_Receiver_Decryption_in[] = {
};

SPortSampleRate g_astPortSampleRate_IncomingMsg_outMsg[] = {
};

// ##PORT_SAMPLE_RATE_TEMPLATE::END


// ##PORT_ARRAY_TEMPLATE::START
SPort g_astPortInfo[] = {
	{
		5, // Task ID
		"in", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Sender_Encryption_in, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		4, // Task ID
		"out", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Sender_Packing_out, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		6, // Task ID
		"in", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Sender_Transfer_in, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		5, // Task ID
		"out", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Sender_Encryption_out, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		9, // Task ID
		"in", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Receiver_Unpacking_in, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		8, // Task ID
		"out", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Receiver_Decryption_out, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		10, // Task ID
		"in", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Receiver_Display_in, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		9, // Task ID
		"out", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Receiver_Unpacking_out, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		2, // Task ID
		"inUser", // Port name
		PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
		g_astPortSampleRate_Control_inUser, // Array of sample rate list
		0, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		0, // Task ID
		"outAck", // Port name
		PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
		g_astPortSampleRate_UserInput_outAck, // Array of sample rate list
		0, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		2, // Task ID
		"inIncoming", // Port name
		PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
		g_astPortSampleRate_Control_inIncoming, // Array of sample rate list
		0, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		1, // Task ID
		"outAck", // Port name
		PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
		g_astPortSampleRate_IncomingMsg_outAck, // Array of sample rate list
		0, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		4, // Task ID
		"in", // Port name
		PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
		g_astPortSampleRate_Sender_Packing_in, // Array of sample rate list
		0, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		0, // Task ID
		"outMsg", // Port name
		PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
		g_astPortSampleRate_UserInput_outMsg, // Array of sample rate list
		0, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		8, // Task ID
		"in", // Port name
		PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
		g_astPortSampleRate_Receiver_Decryption_in, // Array of sample rate list
		0, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		1, // Task ID
		"outMsg", // Port name
		PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
		g_astPortSampleRate_IncomingMsg_outMsg, // Array of sample rate list
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
STask g_astTasks_Sender[];
STask g_astTasks_top[];
STask g_astTasks_Receiver[];
// ##TASK_LIST_DECLARATION_TEMPLATE::END


// ##TASK_GRAPH_DECLARATION_TEMPLATE::START
STaskGraph g_stGraph_Sender;
STaskGraph g_stGraph_top;
STaskGraph g_stGraph_Receiver;
// ##TASK_GRAPH_DECLARATION_TEMPLATE::END


// ##MODE_TRANSITION_TEMPLATE::START
STask *g_pastRelatedChildTasks_Sender_default[] = {
	&g_astTasks_Sender[0],
	&g_astTasks_Sender[2],
	&g_astTasks_Sender[1],
};
		
SModeMap g_astModeMap_Sender[] = {
	{
		0,
		"default",
		g_pastRelatedChildTasks_Sender_default,
		3,
	},
};

SVariableIntMap g_astVariableIntMap_Sender[] = {
	{
		0,
		"var",
		0, 
	},
};


SModeTransitionMachine g_stModeTransition_Sender = {
	3,
	g_astModeMap_Sender, // mode list
	1, // number of modes
	g_astVariableIntMap_Sender, // Integer variable list
	1, // number of integer variables
	NULL, // mode transition function
	0, // Current mode index
	0, // Next mode index
	MODE_STATE_TRANSITING, // mode state (to decide source task execution)
};
STask *g_pastRelatedChildTasks_Receiver_default[] = {
	&g_astTasks_Receiver[0],
	&g_astTasks_Receiver[2],
	&g_astTasks_Receiver[1],
};
		
SModeMap g_astModeMap_Receiver[] = {
	{
		0,
		"default",
		g_pastRelatedChildTasks_Receiver_default,
		3,
	},
};

SVariableIntMap g_astVariableIntMap_Receiver[] = {
	{
		0,
		"var",
		0, 
	},
};


SModeTransitionMachine g_stModeTransition_Receiver = {
	7,
	g_astModeMap_Receiver, // mode list
	1, // number of modes
	g_astVariableIntMap_Receiver, // Integer variable list
	1, // number of integer variables
	NULL, // mode transition function
	0, // Current mode index
	0, // Next mode index
	MODE_STATE_TRANSITING, // mode state (to decide source task execution)
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
SAvailableChunk g_astAvailableInputChunk_channel_4[] = {
	{ 0, 0, NULL, NULL, },
};
SAvailableChunk g_astAvailableInputChunk_channel_5[] = {
	{ 0, 0, NULL, NULL, },
};
SAvailableChunk g_astAvailableInputChunk_channel_6[] = {
	{ 0, 0, NULL, NULL, },
};
SAvailableChunk g_astAvailableInputChunk_channel_7[] = {
	{ 0, 0, NULL, NULL, },
};
// ##AVAILABLE_CHUNK_LIST_TEMPLATE::END

// ##TASK_PARAMETER_TEMPLATE::START
// ##TASK_PARAMETER_TEMPLATE::END

// ##TASK_FUNCTION_LIST::START
STaskFunctions g_ast_Receiver_Decryption_functions[] = {
	{
		Receiver_Decryption_Init0, // Task init function
		Receiver_Decryption_Go0, // Task go function
		Receiver_Decryption_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_Sender_functions[] = {
};

STaskFunctions g_ast_Control_functions[] = {
	{
		Control_Init0, // Task init function
		Control_Go0, // Task go function
		Control_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_Receiver_Unpacking_functions[] = {
	{
		Receiver_Unpacking_Init0, // Task init function
		Receiver_Unpacking_Go0, // Task go function
		Receiver_Unpacking_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_Sender_Encryption_functions[] = {
	{
		Sender_Encryption_Init0, // Task init function
		Sender_Encryption_Go0, // Task go function
		Sender_Encryption_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_Receiver_functions[] = {
};

STaskFunctions g_ast_Sender_Transfer_functions[] = {
	{
		Sender_Transfer_Init0, // Task init function
		Sender_Transfer_Go0, // Task go function
		Sender_Transfer_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_UserInput_functions[] = {
	{
		UserInput_Init0, // Task init function
		UserInput_Go0, // Task go function
		UserInput_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_Receiver_Display_functions[] = {
	{
		Receiver_Display_Init0, // Task init function
		Receiver_Display_Go0, // Task go function
		Receiver_Display_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_Sender_Packing_functions[] = {
	{
		Sender_Packing_Init0, // Task init function
		Sender_Packing_Go0, // Task go function
		Sender_Packing_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_IncomingMsg_functions[] = {
	{
		IncomingMsg_Init0, // Task init function
		IncomingMsg_Go0, // Task go function
		IncomingMsg_Wrapup0, // Task wrapup function
	},
};

// ##TASK_FUNCTION_LIST::END


// ##CHANNEL_LIST_TEMPLATE::START
SChannel g_astChannels[] = {
	{
		0, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
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
			5, // Task ID
			"in", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Sender_Encryption_in, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			4, // Task ID
			"out", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Sender_Packing_out, // Array of sample rate list
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
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
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
			6, // Task ID
			"in", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Sender_Transfer_in, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			5, // Task ID
			"out", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Sender_Encryption_out, // Array of sample rate list
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
		0, // Initial data length 
	},
	{
		2, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
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
			9, // Task ID
			"in", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Receiver_Unpacking_in, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			8, // Task ID
			"out", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Receiver_Decryption_out, // Array of sample rate list
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
		0, // Initial data length 
	},
	{
		3, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
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
			10, // Task ID
			"in", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Receiver_Display_in, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			9, // Task ID
			"out", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Receiver_Unpacking_out, // Array of sample rate list
			1, // Array element number of sample rate list
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
	{
		4, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
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
			"inUser", // Port name
			PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
			g_astPortSampleRate_Control_inUser, // Array of sample rate list
			0, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			0, // Task ID
			"outAck", // Port name
			PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
			g_astPortSampleRate_UserInput_outAck, // Array of sample rate list
			0, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
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
	{
		5, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_GENERAL, // Channel type
		s_pChannel_5_buffer, // Channel buffer pointer
		CHANNEL_5_SIZE, // Channel size
		s_pChannel_5_buffer, // Channel data start
		s_pChannel_5_buffer, // Channel data end
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
			"inIncoming", // Port name
			PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
			g_astPortSampleRate_Control_inIncoming, // Array of sample rate list
			0, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			1, // Task ID
			"outAck", // Port name
			PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
			g_astPortSampleRate_IncomingMsg_outAck, // Array of sample rate list
			0, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
		{
			g_astChunk_channel_5_in, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Input chunk information
		{
			g_astChunk_channel_5_out, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Output chunk information
		CHUNK_NUM_NOT_INITIALIZED, // Written output chunk number
		g_astAvailableInputChunk_channel_5, // Available chunk list
		1,
		NULL, // Chunk list head
		NULL, // Chunk list tail
		0, // Initial data length 
	},
	{
		6, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_GENERAL, // Channel type
		s_pChannel_6_buffer, // Channel buffer pointer
		CHANNEL_6_SIZE, // Channel size
		s_pChannel_6_buffer, // Channel data start
		s_pChannel_6_buffer, // Channel data end
		0, // Channel data length
		0, // Read reference count
		0, // Write reference count
		FALSE, // Read exit setting
		FALSE, // Write exit setting
		NULL, // Mutex
		NULL, // Read available notice event
		NULL, // Write available notice event
		{
			4, // Task ID
			"in", // Port name
			PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
			g_astPortSampleRate_Sender_Packing_in, // Array of sample rate list
			0, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			0, // Task ID
			"outMsg", // Port name
			PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
			g_astPortSampleRate_UserInput_outMsg, // Array of sample rate list
			0, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
		{
			g_astChunk_channel_6_in, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Input chunk information
		{
			g_astChunk_channel_6_out, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Output chunk information
		CHUNK_NUM_NOT_INITIALIZED, // Written output chunk number
		g_astAvailableInputChunk_channel_6, // Available chunk list
		1,
		NULL, // Chunk list head
		NULL, // Chunk list tail
		0, // Initial data length 
	},
	{
		7, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_GENERAL, // Channel type
		s_pChannel_7_buffer, // Channel buffer pointer
		CHANNEL_7_SIZE, // Channel size
		s_pChannel_7_buffer, // Channel data start
		s_pChannel_7_buffer, // Channel data end
		0, // Channel data length
		0, // Read reference count
		0, // Write reference count
		FALSE, // Read exit setting
		FALSE, // Write exit setting
		NULL, // Mutex
		NULL, // Read available notice event
		NULL, // Write available notice event
		{
			8, // Task ID
			"in", // Port name
			PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
			g_astPortSampleRate_Receiver_Decryption_in, // Array of sample rate list
			0, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			1, // Task ID
			"outMsg", // Port name
			PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
			g_astPortSampleRate_IncomingMsg_outMsg, // Array of sample rate list
			0, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
		{
			g_astChunk_channel_7_in, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Input chunk information
		{
			g_astChunk_channel_7_out, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Output chunk information
		CHUNK_NUM_NOT_INITIALIZED, // Written output chunk number
		g_astAvailableInputChunk_channel_7, // Available chunk list
		1,
		NULL, // Chunk list head
		NULL, // Chunk list tail
		0, // Initial data length 
	},
};
// ##CHANNEL_LIST_TEMPLATE::END


// ##TASK_LIST_TEMPLATE::START
STask g_astTasks_Sender[] = {
	{ 	4, // Task ID
		"Sender_Packing", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_Sender_Packing_functions, // Task function array
		1, // Task function array number
		RUN_CONDITION_DATA_DRIVEN, // Run condition
		1, // Run rate
		1, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // Subgraph
		&g_stGraph_Sender, // Parent task graph
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
		"Sender_Encryption", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_Sender_Encryption_functions, // Task function array
		1, // Task function array number
		RUN_CONDITION_DATA_DRIVEN, // Run condition
		1, // Run rate
		1, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // Subgraph
		&g_stGraph_Sender, // Parent task graph
		NULL, // MTM information
		NULL, // Loop information
		NULL, // Task parameter information
		0, // Task parameter number
		TRUE, // Statically scheduled or not
		0,	  // Throughput constraint
		NULL, // Mutex
		NULL, // Conditional variable
	},
	{ 	6, // Task ID
		"Sender_Transfer", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_Sender_Transfer_functions, // Task function array
		1, // Task function array number
		RUN_CONDITION_DATA_DRIVEN, // Run condition
		1, // Run rate
		1, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // Subgraph
		&g_stGraph_Sender, // Parent task graph
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
	{ 	7, // Task ID
		"Receiver", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_Receiver_functions, // Task function array
		0, // Task function array number
		RUN_CONDITION_CONTROL_DRIVEN, // Run condition
		1, // Run rate
		1, // Period
		TIME_METRIC_MICROSEC, // Period metric
		&g_stGraph_Receiver, // Subgraph
		&g_stGraph_top, // Parent task graph
		&g_stModeTransition_Receiver, // MTM information
		NULL, // Loop information
		NULL, // Task parameter information
		0, // Task parameter number
		TRUE, // Statically scheduled or not
		0,	  // Throughput constraint
		NULL, // Mutex
		NULL, // Conditional variable
	},
	{ 	3, // Task ID
		"Sender", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_Sender_functions, // Task function array
		0, // Task function array number
		RUN_CONDITION_CONTROL_DRIVEN, // Run condition
		1, // Run rate
		1, // Period
		TIME_METRIC_MICROSEC, // Period metric
		&g_stGraph_Sender, // Subgraph
		&g_stGraph_top, // Parent task graph
		&g_stModeTransition_Sender, // MTM information
		NULL, // Loop information
		NULL, // Task parameter information
		0, // Task parameter number
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
	{ 	1, // Task ID
		"IncomingMsg", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_IncomingMsg_functions, // Task function array
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
	{ 	2, // Task ID
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

STask g_astTasks_Receiver[] = {
	{ 	8, // Task ID
		"Receiver_Decryption", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_Receiver_Decryption_functions, // Task function array
		1, // Task function array number
		RUN_CONDITION_DATA_DRIVEN, // Run condition
		1, // Run rate
		1, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // Subgraph
		&g_stGraph_Receiver, // Parent task graph
		NULL, // MTM information
		NULL, // Loop information
		NULL, // Task parameter information
		0, // Task parameter number
		TRUE, // Statically scheduled or not
		0,	  // Throughput constraint
		NULL, // Mutex
		NULL, // Conditional variable
	},
	{ 	9, // Task ID
		"Receiver_Unpacking", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_Receiver_Unpacking_functions, // Task function array
		1, // Task function array number
		RUN_CONDITION_DATA_DRIVEN, // Run condition
		1, // Run rate
		1, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // Subgraph
		&g_stGraph_Receiver, // Parent task graph
		NULL, // MTM information
		NULL, // Loop information
		NULL, // Task parameter information
		0, // Task parameter number
		TRUE, // Statically scheduled or not
		0,	  // Throughput constraint
		NULL, // Mutex
		NULL, // Conditional variable
	},
	{ 	10, // Task ID
		"Receiver_Display", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_Receiver_Display_functions, // Task function array
		1, // Task function array number
		RUN_CONDITION_DATA_DRIVEN, // Run condition
		1, // Run rate
		1, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // Subgraph
		&g_stGraph_Receiver, // Parent task graph
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
STaskGraph g_stGraph_Sender = {
		GRAPH_TYPE_PROCESS_NETWORK, // TODO: Task graph type (not used now)
		g_astTasks_Sender, // current task graph's task list
		3, // number of tasks
		&g_astTasks_top[1], // parent task
};

STaskGraph g_stGraph_top = {
		GRAPH_TYPE_PROCESS_NETWORK, // TODO: Task graph type (not used now)
		g_astTasks_top, // current task graph's task list
		5, // number of tasks
		NULL, // parent task
};

STaskGraph g_stGraph_Receiver = {
		GRAPH_TYPE_PROCESS_NETWORK, // TODO: Task graph type (not used now)
		g_astTasks_Receiver, // current task graph's task list
		3, // number of tasks
		&g_astTasks_top[0], // parent task
};

// ##TASK_GRAPH_TEMPLATE::END

// ##TASK_ID_TO_TASK_MAP_TEMPLATE::START
STaskIdToTaskMap g_astTaskIdToTask[] = {
	{ 	8, // Task ID
		"Receiver_Decryption", // Task name
		&g_astTasks_Receiver[0], // Task structure pointer
	},
	{ 	3, // Task ID
		"Sender", // Task name
		&g_astTasks_top[1], // Task structure pointer
	},
	{ 	2, // Task ID
		"Control", // Task name
		&g_astTasks_top[4], // Task structure pointer
	},
	{ 	9, // Task ID
		"Receiver_Unpacking", // Task name
		&g_astTasks_Receiver[1], // Task structure pointer
	},
	{ 	5, // Task ID
		"Sender_Encryption", // Task name
		&g_astTasks_Sender[1], // Task structure pointer
	},
	{ 	7, // Task ID
		"Receiver", // Task name
		&g_astTasks_top[0], // Task structure pointer
	},
	{ 	6, // Task ID
		"Sender_Transfer", // Task name
		&g_astTasks_Sender[2], // Task structure pointer
	},
	{ 	0, // Task ID
		"UserInput", // Task name
		&g_astTasks_top[2], // Task structure pointer
	},
	{ 	10, // Task ID
		"Receiver_Display", // Task name
		&g_astTasks_Receiver[2], // Task structure pointer
	},
	{ 	4, // Task ID
		"Sender_Packing", // Task name
		&g_astTasks_Sender[0], // Task structure pointer
	},
	{ 	1, // Task ID
		"IncomingMsg", // Task name
		&g_astTasks_top[3], // Task structure pointer
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
void Sender_0_0_0_50000_Go(int nTaskId) 
{
	Sender_Packing_Go0(4);//
	Sender_Encryption_Go0(5);//
}

void Sender_0_0_3_50000_Go(int nTaskId) 
{
	Sender_Transfer_Go0(6);//
}

void Receiver_0_0_1_50000_Go(int nTaskId) 
{
	Receiver_Decryption_Go0(8);//
	Receiver_Unpacking_Go0(9);//
}

void Receiver_0_0_2_50000_Go(int nTaskId) 
{
	Receiver_Display_Go0(10);//
}

// ##SCHEDULED_COMPOSITE_TASK_FUNCTION_IMPLEMENTATION::END

// ##SCHEDULED_COMPOSITE_TASKS_TEMPLATE::START
SScheduleList g_astScheduleList_Sender_0_0_0[] = {
	{
		Sender_0_0_0_50000_Go, // Composite GO function
		50000, // Throughput constraint
		FALSE,
	},
};
SScheduleList g_astScheduleList_Sender_0_0_3[] = {
	{
		Sender_0_0_3_50000_Go, // Composite GO function
		50000, // Throughput constraint
		FALSE,
	},
};
SScheduleList g_astScheduleList_Receiver_0_0_1[] = {
	{
		Receiver_0_0_1_50000_Go, // Composite GO function
		50000, // Throughput constraint
		FALSE,
	},
};
SScheduleList g_astScheduleList_Receiver_0_0_2[] = {
	{
		Receiver_0_0_2_50000_Go, // Composite GO function
		50000, // Throughput constraint
		FALSE,
	},
};
// ##SCHEDULED_COMPOSITE_TASKS_TEMPLATE::END



SScheduledTasks g_astScheduledTaskList[] = {
	{	&g_astTasks_top[1], // Parent Task ID
		0, // Mode transition mode ID
		g_astScheduleList_Sender_0_0_0, // schedule list per throughput constraint
		1, // The number of schedules in the schedule list
		0, // Schedule Index (Default to set 0)
		0, // Mode Sequence ID 
	},
	{	&g_astTasks_top[1], // Parent Task ID
		0, // Mode transition mode ID
		g_astScheduleList_Sender_0_0_3, // schedule list per throughput constraint
		1, // The number of schedules in the schedule list
		0, // Schedule Index (Default to set 0)
		1, // Mode Sequence ID 
	},
	{	&g_astTasks_top[0], // Parent Task ID
		0, // Mode transition mode ID
		g_astScheduleList_Receiver_0_0_1, // schedule list per throughput constraint
		1, // The number of schedules in the schedule list
		0, // Schedule Index (Default to set 0)
		0, // Mode Sequence ID 
	},
	{	&g_astTasks_top[0], // Parent Task ID
		0, // Mode transition mode ID
		g_astScheduleList_Receiver_0_0_2, // schedule list per throughput constraint
		1, // The number of schedules in the schedule list
		0, // Schedule Index (Default to set 0)
		1, // Mode Sequence ID 
	},
};


// ##MAPPING_SCHEDULING_INFO_TEMPLATE::START

SMappedGeneralTaskInfo g_astGeneralTaskMappingInfo[] = {
	{	TASK_TYPE_CONTROL, // Task type
		&g_astTasks_top[4], // Task ID or composite task information
		0, // Processor ID
		0, // Processor local ID
	},
	{	TASK_TYPE_COMPUTATIONAL, // Task type
		&g_astTasks_top[2], // Task ID or composite task information
		0, // Processor ID
		0, // Processor local ID
	},
	{	TASK_TYPE_COMPUTATIONAL, // Task type
		&g_astTasks_top[3], // Task ID or composite task information
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
		3, // Processor local ID		
	},
	{
		&g_astScheduledTaskList[2],
		0, // Processor ID
		1, // Processor local ID		
	},
	{
		&g_astScheduledTaskList[3],
		0, // Processor ID
		2, // Processor local ID		
	},
};


SMappedTaskInfo g_stMappingInfo = {
	g_astGeneralTaskMappingInfo, // general task array
	ARRAYLEN(g_astGeneralTaskMappingInfo), // size of general task array
	g_astCompositeTaskMappingInfo, // composite task array
	ARRAYLEN(g_astCompositeTaskMappingInfo), // size of composite task array
};

// ##MAPPING_SCHEDULING_INFO_TEMPLATE::END


// ##LIBRARY_INFO_TEMPLATE::START
SLibrary g_stLibraryInfo[] = {
	{
		"CryptographyLibrary",
		l_CryptographyLibrary_init,
		l_CryptographyLibrary_wrapup,
	},
};

// ##LIBRARY_INFO_TEMPLATE::END


int g_nChannelNum = ARRAYLEN(g_astChannels);
int g_nNumOfTasks_top = ARRAYLEN(g_astTasks_top);
int g_nTaskIdToTaskNum = ARRAYLEN(g_astTaskIdToTask);
int g_nProcessorInfoNum = ARRAYLEN(g_astProcessorInfo);
int g_nLibraryInfoNum = ARRAYLEN(g_stLibraryInfo);

