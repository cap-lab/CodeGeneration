/* uem_data.c made by UEM Translator */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_data.h>
#include <UKTask.h>
#include <UKModeTransition.h>

SExecutionTime g_stExecutionTime = { 20, TIME_METRIC_SEC } ;

// ##TASK_CODE_TEMPLATE::START
void Image_Smoother_Init0(int nTaskId);
void Image_Smoother_Go0(int nTaskId);
void Image_Smoother_Wrapup0();

void Graph_Counter_Init0(int nTaskId);
void Graph_Counter_Go0(int nTaskId);
void Graph_Counter_Wrapup0();

void Image_Display_Image_S_Init0(int nTaskId);
void Image_Display_Image_S_Go0(int nTaskId);
void Image_Display_Image_S_Wrapup0();

void Image_Noising_C_Init0(int nTaskId);
void Image_Noising_C_Go0(int nTaskId);
void Image_Noising_C_Wrapup0();


void Input_Init0(int nTaskId);
void Input_Go0(int nTaskId);
void Input_Wrapup0();

void Control_Init0(int nTaskId);
void Control_Go0(int nTaskId);
void Control_Wrapup0();


void Image_ReadImage_Init0(int nTaskId);
void Image_ReadImage_Go0(int nTaskId);
void Image_ReadImage_Wrapup0();

void Graph_Calculator_Init0(int nTaskId);
void Graph_Calculator_Go0(int nTaskId);
void Graph_Calculator_Wrapup0();

void Image_Display_Image_C_Init0(int nTaskId);
void Image_Display_Image_C_Go0(int nTaskId);
void Image_Display_Image_C_Wrapup0();

void Image_Noising_S_Init0(int nTaskId);
void Image_Noising_S_Go0(int nTaskId);
void Image_Noising_S_Wrapup0();

void Graph_Display_Graph_Init0(int nTaskId);
void Graph_Display_Graph_Go0(int nTaskId);
void Graph_Display_Graph_Wrapup0();

void Image_Sharpener_Init0(int nTaskId);
void Image_Sharpener_Go0(int nTaskId);
void Image_Sharpener_Wrapup0();

// ##TASK_CODE_TEMPLATE::END


// ##LIBRARY_INIT_WRAPUP_TEMPLATE::START
void l_Image_RandomGen_init();
void l_Image_RandomGen_wrapup();

void l_Graph_Math_init();
void l_Graph_Math_wrapup();

// ##LIBRARY_INIT_WRAPUP_TEMPLATE::END


// ##CHANNEL_SIZE_DEFINITION_TEMPLATE::START
#define CHANNEL_0_SIZE (40)
#define CHANNEL_1_SIZE (4)
#define CHANNEL_2_SIZE (2880000)
#define CHANNEL_3_SIZE (2880000)
#define CHANNEL_4_SIZE (2880000)
#define CHANNEL_5_SIZE (2880000)
#define CHANNEL_6_SIZE (2880000)
#define CHANNEL_7_SIZE (2880000)
#define CHANNEL_8_SIZE (4)
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
char s_pChannel_8_buffer[CHANNEL_8_SIZE];
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

SChunk g_astChunk_channel_8_out[] = {
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_8_in[] = {
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

// ##CHUNK_DEFINITION_TEMPLATE::END
//portSampleRateList


// ##PORT_SAMPLE_RATE_TEMPLATE::START
SPortSampleRate g_astPortSampleRate_Graph_Calculator_in[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Graph_Counter_out[] = {
	{ 	"SIN", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
	{ 	"COS", // Mode name
		10, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Graph_Display_Graph_in[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Graph_Calculator_out[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Image_Sharpener_in[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Image_ReadImage_out[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Image_Smoother_in[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Image_Noising_C_in[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Image_Sharpener_out[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Image_Display_Image_C_in[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Image_Noising_C_out[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Image_Noising_S_in[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Image_Smoother_out[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Image_Display_Image_S_in[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Image_Noising_S_out[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Control_in[] = {
};

SPortSampleRate g_astPortSampleRate_Input_out[] = {
};

// ##PORT_SAMPLE_RATE_TEMPLATE::END


// ##PORT_ARRAY_TEMPLATE::START
SPort g_astPortInfo[] = {
	{
		4, // Task ID
		"in", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Graph_Calculator_in, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		3, // Task ID
		"out", // Port name
		PORT_SAMPLE_RATE_MULTIPLE, // Port sample rate type
		g_astPortSampleRate_Graph_Counter_out, // Array of sample rate list
		2, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		5, // Task ID
		"in", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Graph_Display_Graph_in, // Array of sample rate list
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
		g_astPortSampleRate_Graph_Calculator_out, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		8, // Task ID
		"in", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Image_Sharpener_in, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		7, // Task ID
		"out", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Image_ReadImage_out, // Array of sample rate list
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
		g_astPortSampleRate_Image_Smoother_in, // Array of sample rate list
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
		g_astPortSampleRate_Image_Noising_C_in, // Array of sample rate list
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
		g_astPortSampleRate_Image_Sharpener_out, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		12, // Task ID
		"in", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Image_Display_Image_C_in, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		10, // Task ID
		"out", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Image_Noising_C_out, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		11, // Task ID
		"in", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Image_Noising_S_in, // Array of sample rate list
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
		g_astPortSampleRate_Image_Smoother_out, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		13, // Task ID
		"in", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Image_Display_Image_S_in, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		11, // Task ID
		"out", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Image_Noising_S_out, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		1, // Task ID
		"in", // Port name
		PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
		g_astPortSampleRate_Control_in, // Array of sample rate list
		0, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		0, // Task ID
		"out", // Port name
		PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
		g_astPortSampleRate_Input_out, // Array of sample rate list
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
STask g_astTasks_Graph[];
STask g_astTasks_top[];
STask g_astTasks_Image[];
// ##TASK_LIST_DECLARATION_TEMPLATE::END


// ##TASK_GRAPH_DECLARATION_TEMPLATE::START
STaskGraph g_stGraph_Graph;
STaskGraph g_stGraph_top;
STaskGraph g_stGraph_Image;
// ##TASK_GRAPH_DECLARATION_TEMPLATE::END


// ##MODE_TRANSITION_TEMPLATE::START
STask *g_pastRelatedChildTasks_Image_Default[] = {
	&g_astTasks_Image[5],
	&g_astTasks_Image[6],
	&g_astTasks_Image[1],
	&g_astTasks_Image[3],
	&g_astTasks_Image[2],
	&g_astTasks_Image[0],
	&g_astTasks_Image[4],
};
		
SModeMap g_astModeMap_Image[] = {
	{
		0,
		"Default",
		g_pastRelatedChildTasks_Image_Default,
		7,
	},
};

SVariableIntMap g_astVariableIntMap_Image[] = {
};


SModeTransitionMachine g_stModeTransition_Image = {
	6,
	g_astModeMap_Image, // mode list
	1, // number of modes
	g_astVariableIntMap_Image, // Integer variable list
	0, // number of integer variables
	NULL, // mode transition function
	0, // Current mode index
	0, // Next mode index
};
STask *g_pastRelatedChildTasks_Graph_COS[] = {
	&g_astTasks_Graph[1],
	&g_astTasks_Graph[2],
	&g_astTasks_Graph[0],
};
STask *g_pastRelatedChildTasks_Graph_SIN[] = {
	&g_astTasks_Graph[1],
	&g_astTasks_Graph[2],
	&g_astTasks_Graph[0],
};
		
SModeMap g_astModeMap_Graph[] = {
	{
		1,
		"COS",
		g_pastRelatedChildTasks_Graph_COS,
		3,
	},
	{
		0,
		"SIN",
		g_pastRelatedChildTasks_Graph_SIN,
		3,
	},
};

SVariableIntMap g_astVariableIntMap_Graph[] = {
	{
		0,
		"Var",
		0, 
	},
};

static uem_bool transitMode_Graph(SModeTransitionMachine *pstModeTransition) 
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
	  && Var == 2 )
	{
		nNextModeId = 0;
		bModeChanged = TRUE;
	}
		
	pstModeTransition->nNextModeIndex = UKModeTransition_GetModeIndexByModeId(pstModeTransition, nNextModeId);
	
	return bModeChanged;
}

SModeTransitionMachine g_stModeTransition_Graph = {
	2,
	g_astModeMap_Graph, // mode list
	2, // number of modes
	g_astVariableIntMap_Graph, // Integer variable list
	1, // number of integer variables
	transitMode_Graph, // mode transition function
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
SAvailableChunk g_astAvailableInputChunk_channel_8[] = {
	{ 0, 0, NULL, NULL, },
};
// ##AVAILABLE_CHUNK_LIST_TEMPLATE::END

// ##TASK_PARAMETER_TEMPLATE::START
STaskParameter g_astTaskParameter_Image[] = {
	{
		0,
		PARAMETER_TYPE_INT,
		"Gain",
		{ .nParam = 0, },
	},
};
// ##TASK_PARAMETER_TEMPLATE::END

// ##TASK_FUNCTION_LIST::START
STaskFunctions g_ast_Image_Smoother_functions[] = {
	{
		Image_Smoother_Init0, // Task init function
		Image_Smoother_Go0, // Task go function
		Image_Smoother_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_Graph_Counter_functions[] = {
	{
		Graph_Counter_Init0, // Task init function
		Graph_Counter_Go0, // Task go function
		Graph_Counter_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_Image_Display_Image_S_functions[] = {
	{
		Image_Display_Image_S_Init0, // Task init function
		Image_Display_Image_S_Go0, // Task go function
		Image_Display_Image_S_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_Image_Noising_C_functions[] = {
	{
		Image_Noising_C_Init0, // Task init function
		Image_Noising_C_Go0, // Task go function
		Image_Noising_C_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_Image_functions[] = {
};

STaskFunctions g_ast_Input_functions[] = {
	{
		Input_Init0, // Task init function
		Input_Go0, // Task go function
		Input_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_Control_functions[] = {
	{
		Control_Init0, // Task init function
		Control_Go0, // Task go function
		Control_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_Graph_functions[] = {
};

STaskFunctions g_ast_Image_ReadImage_functions[] = {
	{
		Image_ReadImage_Init0, // Task init function
		Image_ReadImage_Go0, // Task go function
		Image_ReadImage_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_Graph_Calculator_functions[] = {
	{
		Graph_Calculator_Init0, // Task init function
		Graph_Calculator_Go0, // Task go function
		Graph_Calculator_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_Image_Display_Image_C_functions[] = {
	{
		Image_Display_Image_C_Init0, // Task init function
		Image_Display_Image_C_Go0, // Task go function
		Image_Display_Image_C_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_Image_Noising_S_functions[] = {
	{
		Image_Noising_S_Init0, // Task init function
		Image_Noising_S_Go0, // Task go function
		Image_Noising_S_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_Graph_Display_Graph_functions[] = {
	{
		Graph_Display_Graph_Init0, // Task init function
		Graph_Display_Graph_Go0, // Task go function
		Graph_Display_Graph_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_Image_Sharpener_functions[] = {
	{
		Image_Sharpener_Init0, // Task init function
		Image_Sharpener_Go0, // Task go function
		Image_Sharpener_Wrapup0, // Task wrapup function
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
			4, // Task ID
			"in", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Graph_Calculator_in, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			3, // Task ID
			"out", // Port name
			PORT_SAMPLE_RATE_MULTIPLE, // Port sample rate type
			g_astPortSampleRate_Graph_Counter_out, // Array of sample rate list
			2, // Array element number of sample rate list
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
			5, // Task ID
			"in", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Graph_Display_Graph_in, // Array of sample rate list
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
			g_astPortSampleRate_Graph_Calculator_out, // Array of sample rate list
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
		3, // Next channel index (which is used for single port is connecting to multiple channels)
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
			8, // Task ID
			"in", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Image_Sharpener_in, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			7, // Task ID
			"out", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Image_ReadImage_out, // Array of sample rate list
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
			9, // Task ID
			"in", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Image_Smoother_in, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			7, // Task ID
			"out", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Image_ReadImage_out, // Array of sample rate list
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
			10, // Task ID
			"in", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Image_Noising_C_in, // Array of sample rate list
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
			g_astPortSampleRate_Image_Sharpener_out, // Array of sample rate list
			1, // Array element number of sample rate list
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
			12, // Task ID
			"in", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Image_Display_Image_C_in, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			10, // Task ID
			"out", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Image_Noising_C_out, // Array of sample rate list
			1, // Array element number of sample rate list
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
			11, // Task ID
			"in", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Image_Noising_S_in, // Array of sample rate list
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
			g_astPortSampleRate_Image_Smoother_out, // Array of sample rate list
			1, // Array element number of sample rate list
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
			13, // Task ID
			"in", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Image_Display_Image_S_in, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			11, // Task ID
			"out", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Image_Noising_S_out, // Array of sample rate list
			1, // Array element number of sample rate list
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
	{
		8, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_GENERAL, // Channel type
		s_pChannel_8_buffer, // Channel buffer pointer
		CHANNEL_8_SIZE, // Channel size
		s_pChannel_8_buffer, // Channel data start
		s_pChannel_8_buffer, // Channel data end
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
			"in", // Port name
			PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
			g_astPortSampleRate_Control_in, // Array of sample rate list
			0, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			0, // Task ID
			"out", // Port name
			PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
			g_astPortSampleRate_Input_out, // Array of sample rate list
			0, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
		{
			g_astChunk_channel_8_in, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Input chunk information
		{
			g_astChunk_channel_8_out, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Output chunk information
		CHUNK_NUM_NOT_INITIALIZED, // Written output chunk number
		g_astAvailableInputChunk_channel_8, // Available chunk list
		1,
		NULL, // Chunk list head
		NULL, // Chunk list tail
		0, // Initial data length 
	},
};
// ##CHANNEL_LIST_TEMPLATE::END


// ##TASK_LIST_TEMPLATE::START
STask g_astTasks_Graph[] = {
	{ 	5, // Task ID
		"Graph_Display_Graph", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_Graph_Display_Graph_functions, // Task function array
		1, // Task function array number
		RUN_CONDITION_DATA_DRIVEN, // Run condition
		1, // Run rate
		1, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // Subgraph
		&g_stGraph_Graph, // Parent task graph
		NULL, // MTM information
		NULL, // Loop information
		NULL, // Task parameter information
		0, // Task parameter number
		TRUE, // Statically scheduled or not
		0,	  // Throughput constraint
		NULL, // Mutex
		NULL, // Conditional variable
	},
	{ 	4, // Task ID
		"Graph_Calculator", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_Graph_Calculator_functions, // Task function array
		1, // Task function array number
		RUN_CONDITION_DATA_DRIVEN, // Run condition
		1, // Run rate
		1, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // Subgraph
		&g_stGraph_Graph, // Parent task graph
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
		"Graph_Counter", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_Graph_Counter_functions, // Task function array
		1, // Task function array number
		RUN_CONDITION_TIME_DRIVEN, // Run condition
		1, // Run rate
		1, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // Subgraph
		&g_stGraph_Graph, // Parent task graph
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
	{ 	2, // Task ID
		"Graph", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_Graph_functions, // Task function array
		0, // Task function array number
		RUN_CONDITION_DATA_DRIVEN, // Run condition
		1, // Run rate
		1, // Period
		TIME_METRIC_MICROSEC, // Period metric
		&g_stGraph_Graph, // Subgraph
		&g_stGraph_top, // Parent task graph
		&g_stModeTransition_Graph, // MTM information
		NULL, // Loop information
		NULL, // Task parameter information
		0, // Task parameter number
		TRUE, // Statically scheduled or not
		0,	  // Throughput constraint
		NULL, // Mutex
		NULL, // Conditional variable
	},
	{ 	6, // Task ID
		"Image", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_Image_functions, // Task function array
		0, // Task function array number
		RUN_CONDITION_DATA_DRIVEN, // Run condition
		1, // Run rate
		1000, // Period
		TIME_METRIC_MICROSEC, // Period metric
		&g_stGraph_Image, // Subgraph
		&g_stGraph_top, // Parent task graph
		&g_stModeTransition_Image, // MTM information
		NULL, // Loop information
		g_astTaskParameter_Image, // Task parameter information
		1, // Task parameter number
		TRUE, // Statically scheduled or not
		0,	  // Throughput constraint
		NULL, // Mutex
		NULL, // Conditional variable
	},
	{ 	0, // Task ID
		"Input", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_Input_functions, // Task function array
		1, // Task function array number
		RUN_CONDITION_TIME_DRIVEN, // Run condition
		1, // Run rate
		200, // Period
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

STask g_astTasks_Image[] = {
	{ 	8, // Task ID
		"Image_Sharpener", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_Image_Sharpener_functions, // Task function array
		1, // Task function array number
		RUN_CONDITION_DATA_DRIVEN, // Run condition
		1, // Run rate
		1, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // Subgraph
		&g_stGraph_Image, // Parent task graph
		NULL, // MTM information
		NULL, // Loop information
		NULL, // Task parameter information
		0, // Task parameter number
		TRUE, // Statically scheduled or not
		0,	  // Throughput constraint
		NULL, // Mutex
		NULL, // Conditional variable
	},
	{ 	11, // Task ID
		"Image_Noising_S", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_Image_Noising_S_functions, // Task function array
		1, // Task function array number
		RUN_CONDITION_DATA_DRIVEN, // Run condition
		1, // Run rate
		1, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // Subgraph
		&g_stGraph_Image, // Parent task graph
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
		"Image_Noising_C", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_Image_Noising_C_functions, // Task function array
		1, // Task function array number
		RUN_CONDITION_DATA_DRIVEN, // Run condition
		1, // Run rate
		1, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // Subgraph
		&g_stGraph_Image, // Parent task graph
		NULL, // MTM information
		NULL, // Loop information
		NULL, // Task parameter information
		0, // Task parameter number
		TRUE, // Statically scheduled or not
		0,	  // Throughput constraint
		NULL, // Mutex
		NULL, // Conditional variable
	},
	{ 	12, // Task ID
		"Image_Display_Image_C", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_Image_Display_Image_C_functions, // Task function array
		1, // Task function array number
		RUN_CONDITION_DATA_DRIVEN, // Run condition
		1, // Run rate
		1, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // Subgraph
		&g_stGraph_Image, // Parent task graph
		NULL, // MTM information
		NULL, // Loop information
		NULL, // Task parameter information
		0, // Task parameter number
		TRUE, // Statically scheduled or not
		0,	  // Throughput constraint
		NULL, // Mutex
		NULL, // Conditional variable
	},
	{ 	7, // Task ID
		"Image_ReadImage", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_Image_ReadImage_functions, // Task function array
		1, // Task function array number
		RUN_CONDITION_TIME_DRIVEN, // Run condition
		1, // Run rate
		1, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // Subgraph
		&g_stGraph_Image, // Parent task graph
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
		"Image_Smoother", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_Image_Smoother_functions, // Task function array
		1, // Task function array number
		RUN_CONDITION_DATA_DRIVEN, // Run condition
		1, // Run rate
		1, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // Subgraph
		&g_stGraph_Image, // Parent task graph
		NULL, // MTM information
		NULL, // Loop information
		NULL, // Task parameter information
		0, // Task parameter number
		TRUE, // Statically scheduled or not
		0,	  // Throughput constraint
		NULL, // Mutex
		NULL, // Conditional variable
	},
	{ 	13, // Task ID
		"Image_Display_Image_S", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_Image_Display_Image_S_functions, // Task function array
		1, // Task function array number
		RUN_CONDITION_DATA_DRIVEN, // Run condition
		1, // Run rate
		1, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // Subgraph
		&g_stGraph_Image, // Parent task graph
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
STaskGraph g_stGraph_Graph = {
		GRAPH_TYPE_PROCESS_NETWORK, // TODO: Task graph type (not used now)
		g_astTasks_Graph, // current task graph's task list
		3, // number of tasks
		&g_astTasks_top[0], // parent task
};

STaskGraph g_stGraph_top = {
		GRAPH_TYPE_PROCESS_NETWORK, // TODO: Task graph type (not used now)
		g_astTasks_top, // current task graph's task list
		4, // number of tasks
		NULL, // parent task
};

STaskGraph g_stGraph_Image = {
		GRAPH_TYPE_PROCESS_NETWORK, // TODO: Task graph type (not used now)
		g_astTasks_Image, // current task graph's task list
		7, // number of tasks
		&g_astTasks_top[1], // parent task
};

// ##TASK_GRAPH_TEMPLATE::END

// ##TASK_ID_TO_TASK_MAP_TEMPLATE::START
STaskIdToTaskMap g_astTaskIdToTask[] = {
	{ 	9, // Task ID
		"Image_Smoother", // Task name
		&g_astTasks_Image[5], // Task structure pointer
	},
	{ 	3, // Task ID
		"Graph_Counter", // Task name
		&g_astTasks_Graph[2], // Task structure pointer
	},
	{ 	13, // Task ID
		"Image_Display_Image_S", // Task name
		&g_astTasks_Image[6], // Task structure pointer
	},
	{ 	10, // Task ID
		"Image_Noising_C", // Task name
		&g_astTasks_Image[2], // Task structure pointer
	},
	{ 	6, // Task ID
		"Image", // Task name
		&g_astTasks_top[1], // Task structure pointer
	},
	{ 	0, // Task ID
		"Input", // Task name
		&g_astTasks_top[2], // Task structure pointer
	},
	{ 	1, // Task ID
		"Control", // Task name
		&g_astTasks_top[3], // Task structure pointer
	},
	{ 	2, // Task ID
		"Graph", // Task name
		&g_astTasks_top[0], // Task structure pointer
	},
	{ 	7, // Task ID
		"Image_ReadImage", // Task name
		&g_astTasks_Image[4], // Task structure pointer
	},
	{ 	4, // Task ID
		"Graph_Calculator", // Task name
		&g_astTasks_Graph[1], // Task structure pointer
	},
	{ 	12, // Task ID
		"Image_Display_Image_C", // Task name
		&g_astTasks_Image[3], // Task structure pointer
	},
	{ 	11, // Task ID
		"Image_Noising_S", // Task name
		&g_astTasks_Image[1], // Task structure pointer
	},
	{ 	5, // Task ID
		"Graph_Display_Graph", // Task name
		&g_astTasks_Graph[0], // Task structure pointer
	},
	{ 	8, // Task ID
		"Image_Sharpener", // Task name
		&g_astTasks_Image[0], // Task structure pointer
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
void Graph_1_0_0_0_Go(int nTaskId) 
{
	Graph_Display_Graph_Go0(5);
	Graph_Display_Graph_Go0(5);
	Graph_Display_Graph_Go0(5);
	Graph_Display_Graph_Go0(5);
	Graph_Display_Graph_Go0(5);
	Graph_Display_Graph_Go0(5);
	Graph_Display_Graph_Go0(5);
	Graph_Display_Graph_Go0(5);
	Graph_Display_Graph_Go0(5);
	Graph_Display_Graph_Go0(5);
}

void Graph_1_0_1_0_Go(int nTaskId) 
{
	Graph_Calculator_Go0(4);
	Graph_Calculator_Go0(4);
	Graph_Calculator_Go0(4);
	Graph_Calculator_Go0(4);
	Graph_Calculator_Go0(4);
	Graph_Calculator_Go0(4);
	Graph_Calculator_Go0(4);
	Graph_Calculator_Go0(4);
	Graph_Calculator_Go0(4);
	Graph_Calculator_Go0(4);
}

void Graph_1_0_3_0_Go(int nTaskId) 
{
	Graph_Counter_Go0(3);
	{
		uem_bool bTransition = FALSE;
		uem_result result;
		STask *pstTask = NULL;
		result = UKTask_GetTaskFromTaskId(nTaskId, &pstTask);
		if(result == ERR_UEM_NOERROR)
		{
			result = UCThreadMutex_Lock(pstTask->hMutex);
			if(result == ERR_UEM_NOERROR){
				bTransition = transitMode_Graph(g_astTasks_top[0].pstMTMInfo);
				UCThreadMutex_Unlock(pstTask->hMutex);
			}
			
			if(bTransition == TRUE) return; // exit when the transition is changed.
		}
	}
	{
		EInternalTaskState enState = INTERNAL_STATE_STOP;
		UKTask_GetTaskState("Graph", &enState);
		if(enState == INTERNAL_STATE_STOP || enState == INTERNAL_STATE_END) return; 
	}
}

void Graph_0_0_0_0_Go(int nTaskId) 
{
	Graph_Display_Graph_Go0(5);
}

void Graph_0_0_3_0_Go(int nTaskId) 
{
	Graph_Counter_Go0(3);
	{
		uem_bool bTransition = FALSE;
		uem_result result;
		STask *pstTask = NULL;
		result = UKTask_GetTaskFromTaskId(nTaskId, &pstTask);
		if(result == ERR_UEM_NOERROR)
		{
			result = UCThreadMutex_Lock(pstTask->hMutex);
			if(result == ERR_UEM_NOERROR){
				bTransition = transitMode_Graph(g_astTasks_top[0].pstMTMInfo);
				UCThreadMutex_Unlock(pstTask->hMutex);
			}
			
			if(bTransition == TRUE) return; // exit when the transition is changed.
		}
	}
	{
		EInternalTaskState enState = INTERNAL_STATE_STOP;
		UKTask_GetTaskState("Graph", &enState);
		if(enState == INTERNAL_STATE_STOP || enState == INTERNAL_STATE_END) return; 
	}
	Graph_Calculator_Go0(4);
}

void Image_0_0_0_0_Go(int nTaskId) 
{
	Image_Sharpener_Go0(8);
	Image_Noising_S_Go0(11);
}

void Image_0_0_1_0_Go(int nTaskId) 
{
	Image_Noising_C_Go0(10);
	Image_Display_Image_C_Go0(12);
}

void Image_0_0_2_0_Go(int nTaskId) 
{
	Image_ReadImage_Go0(7);
	Image_Smoother_Go0(9);
}

void Image_0_0_3_0_Go(int nTaskId) 
{
	Image_Display_Image_S_Go0(13);
}

// ##SCHEDULED_COMPOSITE_TASK_FUNCTION_IMPLEMENTATION::END

// ##SCHEDULED_COMPOSITE_TASKS_TEMPLATE::START
SScheduleList g_astScheduleList_Graph_1_0_0[] = {
	{
		Graph_1_0_0_0_Go, // Composite GO function
		0, // Throughput constraint
	},
};
SScheduleList g_astScheduleList_Graph_1_0_1[] = {
	{
		Graph_1_0_1_0_Go, // Composite GO function
		0, // Throughput constraint
	},
};
SScheduleList g_astScheduleList_Graph_1_0_3[] = {
	{
		Graph_1_0_3_0_Go, // Composite GO function
		0, // Throughput constraint
	},
};
SScheduleList g_astScheduleList_Graph_0_0_0[] = {
	{
		Graph_0_0_0_0_Go, // Composite GO function
		0, // Throughput constraint
	},
};
SScheduleList g_astScheduleList_Graph_0_0_3[] = {
	{
		Graph_0_0_3_0_Go, // Composite GO function
		0, // Throughput constraint
	},
};
SScheduleList g_astScheduleList_Image_0_0_0[] = {
	{
		Image_0_0_0_0_Go, // Composite GO function
		0, // Throughput constraint
	},
};
SScheduleList g_astScheduleList_Image_0_0_1[] = {
	{
		Image_0_0_1_0_Go, // Composite GO function
		0, // Throughput constraint
	},
};
SScheduleList g_astScheduleList_Image_0_0_2[] = {
	{
		Image_0_0_2_0_Go, // Composite GO function
		0, // Throughput constraint
	},
};
SScheduleList g_astScheduleList_Image_0_0_3[] = {
	{
		Image_0_0_3_0_Go, // Composite GO function
		0, // Throughput constraint
	},
};
// ##SCHEDULED_COMPOSITE_TASKS_TEMPLATE::END



SScheduledTasks g_astScheduledTaskList[] = {
	{	&g_astTasks_top[0], // Parent Task ID
		1, // Mode transition mode ID
		g_astScheduleList_Graph_1_0_0, // schedule list per throughput constraint
		1, // The number of schedules in the schedule list
		0, // Schedule Index (Default to set 0)
		0, // Mode Sequence ID 
	},
	{	&g_astTasks_top[0], // Parent Task ID
		1, // Mode transition mode ID
		g_astScheduleList_Graph_1_0_1, // schedule list per throughput constraint
		1, // The number of schedules in the schedule list
		0, // Schedule Index (Default to set 0)
		1, // Mode Sequence ID 
	},
	{	&g_astTasks_top[0], // Parent Task ID
		1, // Mode transition mode ID
		g_astScheduleList_Graph_1_0_3, // schedule list per throughput constraint
		1, // The number of schedules in the schedule list
		0, // Schedule Index (Default to set 0)
		2, // Mode Sequence ID 
	},
	{	&g_astTasks_top[0], // Parent Task ID
		0, // Mode transition mode ID
		g_astScheduleList_Graph_0_0_0, // schedule list per throughput constraint
		1, // The number of schedules in the schedule list
		0, // Schedule Index (Default to set 0)
		0, // Mode Sequence ID 
	},
	{	&g_astTasks_top[0], // Parent Task ID
		0, // Mode transition mode ID
		g_astScheduleList_Graph_0_0_3, // schedule list per throughput constraint
		1, // The number of schedules in the schedule list
		0, // Schedule Index (Default to set 0)
		1, // Mode Sequence ID 
	},
	{	&g_astTasks_top[1], // Parent Task ID
		0, // Mode transition mode ID
		g_astScheduleList_Image_0_0_0, // schedule list per throughput constraint
		1, // The number of schedules in the schedule list
		0, // Schedule Index (Default to set 0)
		0, // Mode Sequence ID 
	},
	{	&g_astTasks_top[1], // Parent Task ID
		0, // Mode transition mode ID
		g_astScheduleList_Image_0_0_1, // schedule list per throughput constraint
		1, // The number of schedules in the schedule list
		0, // Schedule Index (Default to set 0)
		1, // Mode Sequence ID 
	},
	{	&g_astTasks_top[1], // Parent Task ID
		0, // Mode transition mode ID
		g_astScheduleList_Image_0_0_2, // schedule list per throughput constraint
		1, // The number of schedules in the schedule list
		0, // Schedule Index (Default to set 0)
		2, // Mode Sequence ID 
	},
	{	&g_astTasks_top[1], // Parent Task ID
		0, // Mode transition mode ID
		g_astScheduleList_Image_0_0_3, // schedule list per throughput constraint
		1, // The number of schedules in the schedule list
		0, // Schedule Index (Default to set 0)
		3, // Mode Sequence ID 
	},
};


// ##MAPPING_SCHEDULING_INFO_TEMPLATE::START
SMappingSchedulingInfo g_astMappingAndSchedulingInfo[] = {
	{	TASK_TYPE_COMPUTATIONAL, // Task type
		{ .pstTask = &g_astTasks_top[2] }, // Task ID or composite task information
		0, // Processor ID
		0, // Processor local ID
	},
	{	TASK_TYPE_CONTROL, // Task type
		{ .pstTask = &g_astTasks_top[3] }, // Task ID or composite task information
		0, // Processor ID
		1, // Processor local ID
	},
	{	TASK_TYPE_COMPOSITE, // Task type
		{ .pstScheduledTasks = &g_astScheduledTaskList[0] }, // Task ID or composite task information
		0, // Processor ID
		0, // Processor local ID
	},
	{	TASK_TYPE_COMPOSITE, // Task type
		{ .pstScheduledTasks = &g_astScheduledTaskList[1] }, // Task ID or composite task information
		0, // Processor ID
		1, // Processor local ID
	},
	{	TASK_TYPE_COMPOSITE, // Task type
		{ .pstScheduledTasks = &g_astScheduledTaskList[2] }, // Task ID or composite task information
		0, // Processor ID
		3, // Processor local ID
	},
	{	TASK_TYPE_COMPOSITE, // Task type
		{ .pstScheduledTasks = &g_astScheduledTaskList[3] }, // Task ID or composite task information
		0, // Processor ID
		0, // Processor local ID
	},
	{	TASK_TYPE_COMPOSITE, // Task type
		{ .pstScheduledTasks = &g_astScheduledTaskList[4] }, // Task ID or composite task information
		0, // Processor ID
		3, // Processor local ID
	},
	{	TASK_TYPE_COMPOSITE, // Task type
		{ .pstScheduledTasks = &g_astScheduledTaskList[5] }, // Task ID or composite task information
		0, // Processor ID
		0, // Processor local ID
	},
	{	TASK_TYPE_COMPOSITE, // Task type
		{ .pstScheduledTasks = &g_astScheduledTaskList[6] }, // Task ID or composite task information
		0, // Processor ID
		1, // Processor local ID
	},
	{	TASK_TYPE_COMPOSITE, // Task type
		{ .pstScheduledTasks = &g_astScheduledTaskList[7] }, // Task ID or composite task information
		0, // Processor ID
		2, // Processor local ID
	},
	{	TASK_TYPE_COMPOSITE, // Task type
		{ .pstScheduledTasks = &g_astScheduledTaskList[8] }, // Task ID or composite task information
		0, // Processor ID
		3, // Processor local ID
	},
};

SMappedGeneralTaskInfo g_astGeneralTaskMappingInfo[] = {
	{	TASK_TYPE_COMPUTATIONAL, // Task type
		&g_astTasks_top[2], // Task ID or composite task information
		0, // Processor ID
		0, // Processor local ID
	},
	{	TASK_TYPE_CONTROL, // Task type
		&g_astTasks_top[3], // Task ID or composite task information
		0, // Processor ID
		1, // Processor local ID
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
		1, // Processor local ID		
	},
	{
		&g_astScheduledTaskList[2],
		0, // Processor ID
		3, // Processor local ID		
	},
	{
		&g_astScheduledTaskList[3],
		0, // Processor ID
		0, // Processor local ID		
	},
	{
		&g_astScheduledTaskList[4],
		0, // Processor ID
		3, // Processor local ID		
	},
	{
		&g_astScheduledTaskList[5],
		0, // Processor ID
		0, // Processor local ID		
	},
	{
		&g_astScheduledTaskList[6],
		0, // Processor ID
		1, // Processor local ID		
	},
	{
		&g_astScheduledTaskList[7],
		0, // Processor ID
		2, // Processor local ID		
	},
	{
		&g_astScheduledTaskList[8],
		0, // Processor ID
		3, // Processor local ID		
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
		"Image_RandomGen",
		l_Image_RandomGen_init,
		l_Image_RandomGen_wrapup,
	},
	{
		"Graph_Math",
		l_Graph_Math_init,
		l_Graph_Math_wrapup,
	},
};

// ##LIBRARY_INFO_TEMPLATE::END


int g_nChannelNum = ARRAYLEN(g_astChannels);
int g_nNumOfTasks_top = ARRAYLEN(g_astTasks_top);
int g_nTaskIdToTaskNum = ARRAYLEN(g_astTaskIdToTask);
int g_nProcessorInfoNum = ARRAYLEN(g_astProcessorInfo);
int g_nMappingAndSchedulingInfoNum = ARRAYLEN(g_astMappingAndSchedulingInfo);
int g_nLibraryInfoNum = ARRAYLEN(g_stLibraryInfo);

