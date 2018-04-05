/* uem_data.c made by UEM Translator */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_data.h>
#include <UKTask.h>
#include <UKModeTransition.h>
#include <UKHostMemorySystem.h>
//#include <UKGPUMemorySystem.h>

SExecutionTime g_stExecutionTime = { 300, TIME_METRIC_SEC } ;

// ##TASK_CODE_TEMPLATE::START
void RGBtoYUV_Init0(int nTaskId);
void RGBtoYUV_Go0(int nTaskId);
void RGBtoYUV_Wrapup0();

void Hough_Init0(int nTaskId);
void Hough_Go0(int nTaskId);
void Hough_Wrapup0();

void Sobel_Init0(int nTaskId);
void Sobel_Go0(int nTaskId);
void Sobel_Wrapup0();

void YUVtoRGB_Init0(int nTaskId);
void YUVtoRGB_Go0(int nTaskId);
void YUVtoRGB_Wrapup0();

void KNN_Init0(int nTaskId);
void KNN_Go0(int nTaskId);
void KNN_Wrapup0();

void NLM_Init0(int nTaskId);
void NLM_Go0(int nTaskId);
void NLM_Wrapup0();

void Blending_Init0(int nTaskId);
void Blending_Go0(int nTaskId);
void Blending_Wrapup0();

void Merge_Init0(int nTaskId);
void Merge_Go0(int nTaskId);
void Merge_Wrapup0();

void StoreImage_Init0(int nTaskId);
void StoreImage_Go0(int nTaskId);
void StoreImage_Wrapup0();

void LoadImage_Init0(int nTaskId);
void LoadImage_Go0(int nTaskId);
void LoadImage_Wrapup0();

void Gaussian_Init0(int nTaskId);
void Gaussian_Go0(int nTaskId);
void Gaussian_Wrapup0();

void Sharpen_Init0(int nTaskId);
void Sharpen_Go0(int nTaskId);
void Sharpen_Wrapup0();

void DrawLane_Init0(int nTaskId);
void DrawLane_Go0(int nTaskId);
void DrawLane_Wrapup0();

void NonMax_Init0(int nTaskId);
void NonMax_Go0(int nTaskId);
void NonMax_Wrapup0();

// ##TASK_CODE_TEMPLATE::END


// ##LIBRARY_INIT_WRAPUP_TEMPLATE::START
// ##LIBRARY_INIT_WRAPUP_TEMPLATE::END


// ##CHANNEL_SIZE_DEFINITION_TEMPLATE::START
#define CHANNEL_0_SIZE (1382400)
#define CHANNEL_1_SIZE (2764800)
#define CHANNEL_2_SIZE (2764800)
#define CHANNEL_3_SIZE (2764800)
#define CHANNEL_4_SIZE (2764800)
#define CHANNEL_5_SIZE (2990880)
#define CHANNEL_6_SIZE (2764800)
#define CHANNEL_7_SIZE (2764800)
#define CHANNEL_8_SIZE (2764800)
#define CHANNEL_9_SIZE (2764800)
#define CHANNEL_10_SIZE (2764800)
#define CHANNEL_11_SIZE (2764800)
#define CHANNEL_12_SIZE (2764800)
#define CHANNEL_13_SIZE (2764800)
#define CHANNEL_14_SIZE (1382400)
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
char s_pChannel_9_buffer[CHANNEL_9_SIZE];
char s_pChannel_10_buffer[CHANNEL_10_SIZE];
char s_pChannel_11_buffer[CHANNEL_11_SIZE];
char s_pChannel_12_buffer[CHANNEL_12_SIZE];
char s_pChannel_13_buffer[CHANNEL_13_SIZE];
char s_pChannel_14_buffer[CHANNEL_14_SIZE];
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

SChunk g_astChunk_channel_9_out[] = {
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_9_in[] = {
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_10_out[] = {
	{
		s_pChannel_10_buffer, // Chunk start pointer
		s_pChannel_10_buffer, // Data start pointer
		s_pChannel_10_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_10_in[] = {
	{
		s_pChannel_10_buffer, // Chunk start pointer
		s_pChannel_10_buffer, // Data start pointer
		s_pChannel_10_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_11_out[] = {
	{
		s_pChannel_11_buffer, // Chunk start pointer
		s_pChannel_11_buffer, // Data start pointer
		s_pChannel_11_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_11_in[] = {
	{
		s_pChannel_11_buffer, // Chunk start pointer
		s_pChannel_11_buffer, // Data start pointer
		s_pChannel_11_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_12_out[] = {
	{
		s_pChannel_12_buffer, // Chunk start pointer
		s_pChannel_12_buffer, // Data start pointer
		s_pChannel_12_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_12_in[] = {
	{
		s_pChannel_12_buffer, // Chunk start pointer
		s_pChannel_12_buffer, // Data start pointer
		s_pChannel_12_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_13_out[] = {
	{
		s_pChannel_13_buffer, // Chunk start pointer
		s_pChannel_13_buffer, // Data start pointer
		s_pChannel_13_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_13_in[] = {
	{
		s_pChannel_13_buffer, // Chunk start pointer
		s_pChannel_13_buffer, // Data start pointer
		s_pChannel_13_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_14_out[] = {
	{
		s_pChannel_14_buffer, // Chunk start pointer
		s_pChannel_14_buffer, // Data start pointer
		s_pChannel_14_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_14_in[] = {
	{
		s_pChannel_14_buffer, // Chunk start pointer
		s_pChannel_14_buffer, // Data start pointer
		s_pChannel_14_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

// ##CHUNK_DEFINITION_TEMPLATE::END
//portSampleRateList


// ##PORT_SAMPLE_RATE_TEMPLATE::START
SPortSampleRate g_astPortSampleRate_YUVtoRGB_input[] = {
};

SPortSampleRate g_astPortSampleRate_LoadImage_output[] = {
};

SPortSampleRate g_astPortSampleRate_Gaussian_input[] = {
};

SPortSampleRate g_astPortSampleRate_YUVtoRGB_output[] = {
};

SPortSampleRate g_astPortSampleRate_Sobel_input[] = {
};

SPortSampleRate g_astPortSampleRate_Gaussian_output[] = {
};

SPortSampleRate g_astPortSampleRate_NonMax_input[] = {
};

SPortSampleRate g_astPortSampleRate_Sobel_output[] = {
};

SPortSampleRate g_astPortSampleRate_Hough_input[] = {
};

SPortSampleRate g_astPortSampleRate_NonMax_output[] = {
};

SPortSampleRate g_astPortSampleRate_DrawLane_input[] = {
};

SPortSampleRate g_astPortSampleRate_Hough_output[] = {
};

SPortSampleRate g_astPortSampleRate_KNN_input[] = {
};

SPortSampleRate g_astPortSampleRate_NLM_input[] = {
};

SPortSampleRate g_astPortSampleRate_Blending_input_knn[] = {
};

SPortSampleRate g_astPortSampleRate_KNN_output[] = {
};

SPortSampleRate g_astPortSampleRate_Blending_input_nlm[] = {
};

SPortSampleRate g_astPortSampleRate_NLM_output[] = {
};

SPortSampleRate g_astPortSampleRate_Sharpen_input[] = {
};

SPortSampleRate g_astPortSampleRate_Blending_output[] = {
};

SPortSampleRate g_astPortSampleRate_Merge_input_origin[] = {
};

SPortSampleRate g_astPortSampleRate_Sharpen_output[] = {
};

SPortSampleRate g_astPortSampleRate_Merge_input_hough[] = {
};

SPortSampleRate g_astPortSampleRate_DrawLane_output[] = {
};

SPortSampleRate g_astPortSampleRate_RGBtoYUV_input[] = {
};

SPortSampleRate g_astPortSampleRate_Merge_output[] = {
};

SPortSampleRate g_astPortSampleRate_StoreImage_input[] = {
};

SPortSampleRate g_astPortSampleRate_RGBtoYUV_output[] = {
};

// ##PORT_SAMPLE_RATE_TEMPLATE::END


// ##PORT_ARRAY_TEMPLATE::START
SPort g_astPortInfo[] = {
	{
		1, // Task ID
		"input", // Port name
		PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
		g_astPortSampleRate_YUVtoRGB_input, // Array of sample rate list
		0, // Array element number of sample rate list
		0, //Selected sample rate index
		1, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		0, // Task ID
		"output", // Port name
		PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
		g_astPortSampleRate_LoadImage_output, // Array of sample rate list
		0, // Array element number of sample rate list
		0, //Selected sample rate index
		1, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		4, // Task ID
		"input", // Port name
		PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
		g_astPortSampleRate_Gaussian_input, // Array of sample rate list
		0, // Array element number of sample rate list
		0, //Selected sample rate index
		1, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		1, // Task ID
		"output", // Port name
		PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
		g_astPortSampleRate_YUVtoRGB_output, // Array of sample rate list
		0, // Array element number of sample rate list
		0, //Selected sample rate index
		1, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		5, // Task ID
		"input", // Port name
		PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
		g_astPortSampleRate_Sobel_input, // Array of sample rate list
		0, // Array element number of sample rate list
		0, //Selected sample rate index
		1, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		4, // Task ID
		"output", // Port name
		PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
		g_astPortSampleRate_Gaussian_output, // Array of sample rate list
		0, // Array element number of sample rate list
		0, //Selected sample rate index
		1, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		6, // Task ID
		"input", // Port name
		PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
		g_astPortSampleRate_NonMax_input, // Array of sample rate list
		0, // Array element number of sample rate list
		0, //Selected sample rate index
		1, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		5, // Task ID
		"output", // Port name
		PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
		g_astPortSampleRate_Sobel_output, // Array of sample rate list
		0, // Array element number of sample rate list
		0, //Selected sample rate index
		1, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		7, // Task ID
		"input", // Port name
		PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
		g_astPortSampleRate_Hough_input, // Array of sample rate list
		0, // Array element number of sample rate list
		0, //Selected sample rate index
		1, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		6, // Task ID
		"output", // Port name
		PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
		g_astPortSampleRate_NonMax_output, // Array of sample rate list
		0, // Array element number of sample rate list
		0, //Selected sample rate index
		1, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		8, // Task ID
		"input", // Port name
		PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
		g_astPortSampleRate_DrawLane_input, // Array of sample rate list
		0, // Array element number of sample rate list
		0, //Selected sample rate index
		1, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		7, // Task ID
		"output", // Port name
		PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
		g_astPortSampleRate_Hough_output, // Array of sample rate list
		0, // Array element number of sample rate list
		0, //Selected sample rate index
		1, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		3, // Task ID
		"input", // Port name
		PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
		g_astPortSampleRate_KNN_input, // Array of sample rate list
		0, // Array element number of sample rate list
		0, //Selected sample rate index
		1, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		2, // Task ID
		"input", // Port name
		PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
		g_astPortSampleRate_NLM_input, // Array of sample rate list
		0, // Array element number of sample rate list
		0, //Selected sample rate index
		1, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		9, // Task ID
		"input_knn", // Port name
		PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
		g_astPortSampleRate_Blending_input_knn, // Array of sample rate list
		0, // Array element number of sample rate list
		0, //Selected sample rate index
		1, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		3, // Task ID
		"output", // Port name
		PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
		g_astPortSampleRate_KNN_output, // Array of sample rate list
		0, // Array element number of sample rate list
		0, //Selected sample rate index
		1, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		9, // Task ID
		"input_nlm", // Port name
		PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
		g_astPortSampleRate_Blending_input_nlm, // Array of sample rate list
		0, // Array element number of sample rate list
		0, //Selected sample rate index
		1, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		2, // Task ID
		"output", // Port name
		PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
		g_astPortSampleRate_NLM_output, // Array of sample rate list
		0, // Array element number of sample rate list
		0, //Selected sample rate index
		1, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		10, // Task ID
		"input", // Port name
		PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
		g_astPortSampleRate_Sharpen_input, // Array of sample rate list
		0, // Array element number of sample rate list
		0, //Selected sample rate index
		1, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		9, // Task ID
		"output", // Port name
		PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
		g_astPortSampleRate_Blending_output, // Array of sample rate list
		0, // Array element number of sample rate list
		0, //Selected sample rate index
		1, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		11, // Task ID
		"input_origin", // Port name
		PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
		g_astPortSampleRate_Merge_input_origin, // Array of sample rate list
		0, // Array element number of sample rate list
		0, //Selected sample rate index
		1, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		10, // Task ID
		"output", // Port name
		PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
		g_astPortSampleRate_Sharpen_output, // Array of sample rate list
		0, // Array element number of sample rate list
		0, //Selected sample rate index
		1, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		11, // Task ID
		"input_hough", // Port name
		PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
		g_astPortSampleRate_Merge_input_hough, // Array of sample rate list
		0, // Array element number of sample rate list
		0, //Selected sample rate index
		1, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		8, // Task ID
		"output", // Port name
		PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
		g_astPortSampleRate_DrawLane_output, // Array of sample rate list
		0, // Array element number of sample rate list
		0, //Selected sample rate index
		1, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		12, // Task ID
		"input", // Port name
		PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
		g_astPortSampleRate_RGBtoYUV_input, // Array of sample rate list
		0, // Array element number of sample rate list
		0, //Selected sample rate index
		1, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		11, // Task ID
		"output", // Port name
		PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
		g_astPortSampleRate_Merge_output, // Array of sample rate list
		0, // Array element number of sample rate list
		0, //Selected sample rate index
		1, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		13, // Task ID
		"input", // Port name
		PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
		g_astPortSampleRate_StoreImage_input, // Array of sample rate list
		0, // Array element number of sample rate list
		0, //Selected sample rate index
		1, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		12, // Task ID
		"output", // Port name
		PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
		g_astPortSampleRate_RGBtoYUV_output, // Array of sample rate list
		0, // Array element number of sample rate list
		0, //Selected sample rate index
		1, // Sample size
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
SAvailableChunk g_astAvailableInputChunk_channel_9[] = {
	{ 0, 0, NULL, NULL, },
};
SAvailableChunk g_astAvailableInputChunk_channel_10[] = {
	{ 0, 0, NULL, NULL, },
};
SAvailableChunk g_astAvailableInputChunk_channel_11[] = {
	{ 0, 0, NULL, NULL, },
};
SAvailableChunk g_astAvailableInputChunk_channel_12[] = {
	{ 0, 0, NULL, NULL, },
};
SAvailableChunk g_astAvailableInputChunk_channel_13[] = {
	{ 0, 0, NULL, NULL, },
};
SAvailableChunk g_astAvailableInputChunk_channel_14[] = {
	{ 0, 0, NULL, NULL, },
};
// ##AVAILABLE_CHUNK_LIST_TEMPLATE::END

// ##TASK_PARAMETER_TEMPLATE::START
// ##TASK_PARAMETER_TEMPLATE::END

// ##TASK_FUNCTION_LIST::START
STaskFunctions g_ast_RGBtoYUV_functions[] = {
	{
		RGBtoYUV_Init0, // Task init function
		RGBtoYUV_Go0, // Task go function
		RGBtoYUV_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_Hough_functions[] = {
	{
		Hough_Init0, // Task init function
		Hough_Go0, // Task go function
		Hough_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_Sobel_functions[] = {
	{
		Sobel_Init0, // Task init function
		Sobel_Go0, // Task go function
		Sobel_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_YUVtoRGB_functions[] = {
	{
		YUVtoRGB_Init0, // Task init function
		YUVtoRGB_Go0, // Task go function
		YUVtoRGB_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_KNN_functions[] = {
	{
		KNN_Init0, // Task init function
		KNN_Go0, // Task go function
		KNN_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_NLM_functions[] = {
	{
		NLM_Init0, // Task init function
		NLM_Go0, // Task go function
		NLM_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_Blending_functions[] = {
	{
		Blending_Init0, // Task init function
		Blending_Go0, // Task go function
		Blending_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_Merge_functions[] = {
	{
		Merge_Init0, // Task init function
		Merge_Go0, // Task go function
		Merge_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_StoreImage_functions[] = {
	{
		StoreImage_Init0, // Task init function
		StoreImage_Go0, // Task go function
		StoreImage_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_LoadImage_functions[] = {
	{
		LoadImage_Init0, // Task init function
		LoadImage_Go0, // Task go function
		LoadImage_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_Gaussian_functions[] = {
	{
		Gaussian_Init0, // Task init function
		Gaussian_Go0, // Task go function
		Gaussian_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_Sharpen_functions[] = {
	{
		Sharpen_Init0, // Task init function
		Sharpen_Go0, // Task go function
		Sharpen_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_DrawLane_functions[] = {
	{
		DrawLane_Init0, // Task init function
		DrawLane_Go0, // Task go function
		DrawLane_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_NonMax_functions[] = {
	{
		NonMax_Init0, // Task init function
		NonMax_Go0, // Task go function
		NonMax_Wrapup0, // Task wrapup function
	},
};

// ##TASK_FUNCTION_LIST::END


SGenericMemoryAccess g_stHostMemory = {
	UKHostMemorySystem_CreateMemory,
	UKHostMemorySystem_CopyToMemory,
	UKHostMemorySystem_CopyFromMemory,
	UKHostMemorySystem_DestroyMemory,
};

/*
SGenericMemoryAccess g_stHostToDeviceMemory = {
	UKHostMemorySystem_CreateMemory,
	UKGPUMemorySystem_CopyHostToDeviceMemory,
	UKGPUMemorySystem_CopyDeviceToHostMemory,
	UKHostMemorySystem_DestroyMemory,
};

SGenericMemoryAccess g_stDeviceToHostMemory = {
	UKHostMemorySystem_CreateMemory,
	UKGPUMemorySystem_CopyDeviceToHostMemory,
	UKGPUMemorySystem_CopyHostToDeviceMemory,
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
	UKGPUMemorySystem_CopyHostToDeviceMemory,
	UKGPUMemorySystem_CopyDeviceToHostMemory,
	UKGPUMemorySystem_DestroyHostAllocMemory,
};
*/

// ##SPECIFIC_CHANNEL_LIST_TEMPLATE::START
SSharedMemoryChannel g_stSharedMemoryChannel_0 = {
		s_pChannel_0_buffer, // Channel buffer pointer
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
		1, // maximum input port chunk size for all port sample rate cases
		NULL, // Chunk list head
		NULL, // Chunk list tail 
		&g_stHostMemory, // Host memory access API
		TRUE, // memory is statically allocated
};

SSharedMemoryChannel g_stSharedMemoryChannel_1 = {
		s_pChannel_1_buffer, // Channel buffer pointer
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
		1, // maximum input port chunk size for all port sample rate cases
		NULL, // Chunk list head
		NULL, // Chunk list tail 
		&g_stHostMemory, // Host memory access API
		TRUE, // memory is statically allocated
};

SSharedMemoryChannel g_stSharedMemoryChannel_2 = {
		s_pChannel_2_buffer, // Channel buffer pointer
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
		1, // maximum input port chunk size for all port sample rate cases
		NULL, // Chunk list head
		NULL, // Chunk list tail 
		&g_stHostMemory, // Host memory access API
		TRUE, // memory is statically allocated
};

SSharedMemoryChannel g_stSharedMemoryChannel_3 = {
		s_pChannel_3_buffer, // Channel buffer pointer
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
		1, // maximum input port chunk size for all port sample rate cases
		NULL, // Chunk list head
		NULL, // Chunk list tail 
		&g_stHostMemory, // Host memory access API
		TRUE, // memory is statically allocated
};

SSharedMemoryChannel g_stSharedMemoryChannel_4 = {
		s_pChannel_4_buffer, // Channel buffer pointer
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
		1, // maximum input port chunk size for all port sample rate cases
		NULL, // Chunk list head
		NULL, // Chunk list tail 
		&g_stHostMemory, // Host memory access API
		TRUE, // memory is statically allocated
};

SSharedMemoryChannel g_stSharedMemoryChannel_5 = {
		s_pChannel_5_buffer, // Channel buffer pointer
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
		1, // maximum input port chunk size for all port sample rate cases
		NULL, // Chunk list head
		NULL, // Chunk list tail 
		&g_stHostMemory, // Host memory access API
		TRUE, // memory is statically allocated
};

SSharedMemoryChannel g_stSharedMemoryChannel_6 = {
		s_pChannel_6_buffer, // Channel buffer pointer
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
		1, // maximum input port chunk size for all port sample rate cases
		NULL, // Chunk list head
		NULL, // Chunk list tail 
		&g_stHostMemory, // Host memory access API
		TRUE, // memory is statically allocated
};

SSharedMemoryChannel g_stSharedMemoryChannel_7 = {
		s_pChannel_7_buffer, // Channel buffer pointer
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
		1, // maximum input port chunk size for all port sample rate cases
		NULL, // Chunk list head
		NULL, // Chunk list tail 
		&g_stHostMemory, // Host memory access API
		TRUE, // memory is statically allocated
};

SSharedMemoryChannel g_stSharedMemoryChannel_8 = {
		s_pChannel_8_buffer, // Channel buffer pointer
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
		1, // maximum input port chunk size for all port sample rate cases
		NULL, // Chunk list head
		NULL, // Chunk list tail 
		&g_stHostMemory, // Host memory access API
		TRUE, // memory is statically allocated
};

SSharedMemoryChannel g_stSharedMemoryChannel_9 = {
		s_pChannel_9_buffer, // Channel buffer pointer
		s_pChannel_9_buffer, // Channel data start
		s_pChannel_9_buffer, // Channel data end
		0, // Channel data length
		0, // Read reference count
		0, // Write reference count
		FALSE, // Read exit setting
		FALSE, // Write exit setting
		NULL, // Mutex
		NULL, // Read available notice event
		NULL, // Write available notice event
		{
			g_astChunk_channel_9_in, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Input chunk information
		{
			g_astChunk_channel_9_out, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Output chunk information
		CHUNK_NUM_NOT_INITIALIZED, // Written output chunk number
		g_astAvailableInputChunk_channel_9, // Available chunk list
		1, // maximum input port chunk size for all port sample rate cases
		NULL, // Chunk list head
		NULL, // Chunk list tail 
		&g_stHostMemory, // Host memory access API
		TRUE, // memory is statically allocated
};

SSharedMemoryChannel g_stSharedMemoryChannel_10 = {
		s_pChannel_10_buffer, // Channel buffer pointer
		s_pChannel_10_buffer, // Channel data start
		s_pChannel_10_buffer, // Channel data end
		0, // Channel data length
		0, // Read reference count
		0, // Write reference count
		FALSE, // Read exit setting
		FALSE, // Write exit setting
		NULL, // Mutex
		NULL, // Read available notice event
		NULL, // Write available notice event
		{
			g_astChunk_channel_10_in, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Input chunk information
		{
			g_astChunk_channel_10_out, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Output chunk information
		CHUNK_NUM_NOT_INITIALIZED, // Written output chunk number
		g_astAvailableInputChunk_channel_10, // Available chunk list
		1, // maximum input port chunk size for all port sample rate cases
		NULL, // Chunk list head
		NULL, // Chunk list tail 
		&g_stHostMemory, // Host memory access API
		TRUE, // memory is statically allocated
};

SSharedMemoryChannel g_stSharedMemoryChannel_11 = {
		s_pChannel_11_buffer, // Channel buffer pointer
		s_pChannel_11_buffer, // Channel data start
		s_pChannel_11_buffer, // Channel data end
		0, // Channel data length
		0, // Read reference count
		0, // Write reference count
		FALSE, // Read exit setting
		FALSE, // Write exit setting
		NULL, // Mutex
		NULL, // Read available notice event
		NULL, // Write available notice event
		{
			g_astChunk_channel_11_in, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Input chunk information
		{
			g_astChunk_channel_11_out, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Output chunk information
		CHUNK_NUM_NOT_INITIALIZED, // Written output chunk number
		g_astAvailableInputChunk_channel_11, // Available chunk list
		1, // maximum input port chunk size for all port sample rate cases
		NULL, // Chunk list head
		NULL, // Chunk list tail 
		&g_stHostMemory, // Host memory access API
		TRUE, // memory is statically allocated
};

SSharedMemoryChannel g_stSharedMemoryChannel_12 = {
		s_pChannel_12_buffer, // Channel buffer pointer
		s_pChannel_12_buffer, // Channel data start
		s_pChannel_12_buffer, // Channel data end
		0, // Channel data length
		0, // Read reference count
		0, // Write reference count
		FALSE, // Read exit setting
		FALSE, // Write exit setting
		NULL, // Mutex
		NULL, // Read available notice event
		NULL, // Write available notice event
		{
			g_astChunk_channel_12_in, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Input chunk information
		{
			g_astChunk_channel_12_out, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Output chunk information
		CHUNK_NUM_NOT_INITIALIZED, // Written output chunk number
		g_astAvailableInputChunk_channel_12, // Available chunk list
		1, // maximum input port chunk size for all port sample rate cases
		NULL, // Chunk list head
		NULL, // Chunk list tail 
		&g_stHostMemory, // Host memory access API
		TRUE, // memory is statically allocated
};

SSharedMemoryChannel g_stSharedMemoryChannel_13 = {
		s_pChannel_13_buffer, // Channel buffer pointer
		s_pChannel_13_buffer, // Channel data start
		s_pChannel_13_buffer, // Channel data end
		0, // Channel data length
		0, // Read reference count
		0, // Write reference count
		FALSE, // Read exit setting
		FALSE, // Write exit setting
		NULL, // Mutex
		NULL, // Read available notice event
		NULL, // Write available notice event
		{
			g_astChunk_channel_13_in, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Input chunk information
		{
			g_astChunk_channel_13_out, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Output chunk information
		CHUNK_NUM_NOT_INITIALIZED, // Written output chunk number
		g_astAvailableInputChunk_channel_13, // Available chunk list
		1, // maximum input port chunk size for all port sample rate cases
		NULL, // Chunk list head
		NULL, // Chunk list tail 
		&g_stHostMemory, // Host memory access API
		TRUE, // memory is statically allocated
};

SSharedMemoryChannel g_stSharedMemoryChannel_14 = {
		s_pChannel_14_buffer, // Channel buffer pointer
		s_pChannel_14_buffer, // Channel data start
		s_pChannel_14_buffer, // Channel data end
		0, // Channel data length
		0, // Read reference count
		0, // Write reference count
		FALSE, // Read exit setting
		FALSE, // Write exit setting
		NULL, // Mutex
		NULL, // Read available notice event
		NULL, // Write available notice event
		{
			g_astChunk_channel_14_in, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Input chunk information
		{
			g_astChunk_channel_14_out, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Output chunk information
		CHUNK_NUM_NOT_INITIALIZED, // Written output chunk number
		g_astAvailableInputChunk_channel_14, // Available chunk list
		1, // maximum input port chunk size for all port sample rate cases
		NULL, // Chunk list head
		NULL, // Chunk list tail 
		&g_stHostMemory, // Host memory access API
		TRUE, // memory is statically allocated
};

// ##SPECIFIC_CHANNEL_LIST_TEMPLATE::END


// ##CHANNEL_LIST_TEMPLATE::START
SChannel g_astChannels[] = {
	{
		0, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_GENERAL, // Channel type
		CHANNEL_0_SIZE, // Channel size
		{
			1, // Task ID
			"input", // Port name
			PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
			g_astPortSampleRate_YUVtoRGB_input, // Array of sample rate list
			0, // Array element number of sample rate list
			0, //Selected sample rate index
			1, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			0, // Task ID
			"output", // Port name
			PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
			g_astPortSampleRate_LoadImage_output, // Array of sample rate list
			0, // Array element number of sample rate list
			0, //Selected sample rate index
			1, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
		0, // Initial data length
		&g_stSharedMemoryChannel_0, // specific shared memory channel structure pointer
	},
	{
		1, // Channel ID
		6, // Next channel index (which is used for single port is connecting to multiple channels)
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_GENERAL, // Channel type
		CHANNEL_1_SIZE, // Channel size
		{
			4, // Task ID
			"input", // Port name
			PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
			g_astPortSampleRate_Gaussian_input, // Array of sample rate list
			0, // Array element number of sample rate list
			0, //Selected sample rate index
			1, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			1, // Task ID
			"output", // Port name
			PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
			g_astPortSampleRate_YUVtoRGB_output, // Array of sample rate list
			0, // Array element number of sample rate list
			0, //Selected sample rate index
			1, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
		0, // Initial data length
		&g_stSharedMemoryChannel_1, // specific shared memory channel structure pointer
	},
	{
		2, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_GENERAL, // Channel type
		CHANNEL_2_SIZE, // Channel size
		{
			5, // Task ID
			"input", // Port name
			PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
			g_astPortSampleRate_Sobel_input, // Array of sample rate list
			0, // Array element number of sample rate list
			0, //Selected sample rate index
			1, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			4, // Task ID
			"output", // Port name
			PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
			g_astPortSampleRate_Gaussian_output, // Array of sample rate list
			0, // Array element number of sample rate list
			0, //Selected sample rate index
			1, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
		0, // Initial data length
		&g_stSharedMemoryChannel_2, // specific shared memory channel structure pointer
	},
	{
		3, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_GENERAL, // Channel type
		CHANNEL_3_SIZE, // Channel size
		{
			6, // Task ID
			"input", // Port name
			PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
			g_astPortSampleRate_NonMax_input, // Array of sample rate list
			0, // Array element number of sample rate list
			0, //Selected sample rate index
			1, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			5, // Task ID
			"output", // Port name
			PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
			g_astPortSampleRate_Sobel_output, // Array of sample rate list
			0, // Array element number of sample rate list
			0, //Selected sample rate index
			1, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
		0, // Initial data length
		&g_stSharedMemoryChannel_3, // specific shared memory channel structure pointer
	},
	{
		4, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_GENERAL, // Channel type
		CHANNEL_4_SIZE, // Channel size
		{
			7, // Task ID
			"input", // Port name
			PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
			g_astPortSampleRate_Hough_input, // Array of sample rate list
			0, // Array element number of sample rate list
			0, //Selected sample rate index
			1, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			6, // Task ID
			"output", // Port name
			PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
			g_astPortSampleRate_NonMax_output, // Array of sample rate list
			0, // Array element number of sample rate list
			0, //Selected sample rate index
			1, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
		0, // Initial data length
		&g_stSharedMemoryChannel_4, // specific shared memory channel structure pointer
	},
	{
		5, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_GENERAL, // Channel type
		CHANNEL_5_SIZE, // Channel size
		{
			8, // Task ID
			"input", // Port name
			PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
			g_astPortSampleRate_DrawLane_input, // Array of sample rate list
			0, // Array element number of sample rate list
			0, //Selected sample rate index
			1, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			7, // Task ID
			"output", // Port name
			PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
			g_astPortSampleRate_Hough_output, // Array of sample rate list
			0, // Array element number of sample rate list
			0, //Selected sample rate index
			1, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
		0, // Initial data length
		&g_stSharedMemoryChannel_5, // specific shared memory channel structure pointer
	},
	{
		6, // Channel ID
		7, // Next channel index (which is used for single port is connecting to multiple channels)
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_GENERAL, // Channel type
		CHANNEL_6_SIZE, // Channel size
		{
			3, // Task ID
			"input", // Port name
			PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
			g_astPortSampleRate_KNN_input, // Array of sample rate list
			0, // Array element number of sample rate list
			0, //Selected sample rate index
			1, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			1, // Task ID
			"output", // Port name
			PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
			g_astPortSampleRate_YUVtoRGB_output, // Array of sample rate list
			0, // Array element number of sample rate list
			0, //Selected sample rate index
			1, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
		0, // Initial data length
		&g_stSharedMemoryChannel_6, // specific shared memory channel structure pointer
	},
	{
		7, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_GENERAL, // Channel type
		CHANNEL_7_SIZE, // Channel size
		{
			2, // Task ID
			"input", // Port name
			PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
			g_astPortSampleRate_NLM_input, // Array of sample rate list
			0, // Array element number of sample rate list
			0, //Selected sample rate index
			1, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			1, // Task ID
			"output", // Port name
			PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
			g_astPortSampleRate_YUVtoRGB_output, // Array of sample rate list
			0, // Array element number of sample rate list
			0, //Selected sample rate index
			1, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
		0, // Initial data length
		&g_stSharedMemoryChannel_7, // specific shared memory channel structure pointer
	},
	{
		8, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_GENERAL, // Channel type
		CHANNEL_8_SIZE, // Channel size
		{
			9, // Task ID
			"input_knn", // Port name
			PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
			g_astPortSampleRate_Blending_input_knn, // Array of sample rate list
			0, // Array element number of sample rate list
			0, //Selected sample rate index
			1, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			3, // Task ID
			"output", // Port name
			PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
			g_astPortSampleRate_KNN_output, // Array of sample rate list
			0, // Array element number of sample rate list
			0, //Selected sample rate index
			1, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
		0, // Initial data length
		&g_stSharedMemoryChannel_8, // specific shared memory channel structure pointer
	},
	{
		9, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_GENERAL, // Channel type
		CHANNEL_9_SIZE, // Channel size
		{
			9, // Task ID
			"input_nlm", // Port name
			PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
			g_astPortSampleRate_Blending_input_nlm, // Array of sample rate list
			0, // Array element number of sample rate list
			0, //Selected sample rate index
			1, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			2, // Task ID
			"output", // Port name
			PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
			g_astPortSampleRate_NLM_output, // Array of sample rate list
			0, // Array element number of sample rate list
			0, //Selected sample rate index
			1, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
		0, // Initial data length
		&g_stSharedMemoryChannel_9, // specific shared memory channel structure pointer
	},
	{
		10, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_GENERAL, // Channel type
		CHANNEL_10_SIZE, // Channel size
		{
			10, // Task ID
			"input", // Port name
			PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
			g_astPortSampleRate_Sharpen_input, // Array of sample rate list
			0, // Array element number of sample rate list
			0, //Selected sample rate index
			1, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			9, // Task ID
			"output", // Port name
			PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
			g_astPortSampleRate_Blending_output, // Array of sample rate list
			0, // Array element number of sample rate list
			0, //Selected sample rate index
			1, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
		0, // Initial data length
		&g_stSharedMemoryChannel_10, // specific shared memory channel structure pointer
	},
	{
		11, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_GENERAL, // Channel type
		CHANNEL_11_SIZE, // Channel size
		{
			11, // Task ID
			"input_origin", // Port name
			PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
			g_astPortSampleRate_Merge_input_origin, // Array of sample rate list
			0, // Array element number of sample rate list
			0, //Selected sample rate index
			1, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			10, // Task ID
			"output", // Port name
			PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
			g_astPortSampleRate_Sharpen_output, // Array of sample rate list
			0, // Array element number of sample rate list
			0, //Selected sample rate index
			1, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
		0, // Initial data length
		&g_stSharedMemoryChannel_11, // specific shared memory channel structure pointer
	},
	{
		12, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_GENERAL, // Channel type
		CHANNEL_12_SIZE, // Channel size
		{
			11, // Task ID
			"input_hough", // Port name
			PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
			g_astPortSampleRate_Merge_input_hough, // Array of sample rate list
			0, // Array element number of sample rate list
			0, //Selected sample rate index
			1, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			8, // Task ID
			"output", // Port name
			PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
			g_astPortSampleRate_DrawLane_output, // Array of sample rate list
			0, // Array element number of sample rate list
			0, //Selected sample rate index
			1, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
		0, // Initial data length
		&g_stSharedMemoryChannel_12, // specific shared memory channel structure pointer
	},
	{
		13, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_GENERAL, // Channel type
		CHANNEL_13_SIZE, // Channel size
		{
			12, // Task ID
			"input", // Port name
			PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
			g_astPortSampleRate_RGBtoYUV_input, // Array of sample rate list
			0, // Array element number of sample rate list
			0, //Selected sample rate index
			1, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			11, // Task ID
			"output", // Port name
			PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
			g_astPortSampleRate_Merge_output, // Array of sample rate list
			0, // Array element number of sample rate list
			0, //Selected sample rate index
			1, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
		0, // Initial data length
		&g_stSharedMemoryChannel_13, // specific shared memory channel structure pointer
	},
	{
		14, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_GENERAL, // Channel type
		CHANNEL_14_SIZE, // Channel size
		{
			13, // Task ID
			"input", // Port name
			PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
			g_astPortSampleRate_StoreImage_input, // Array of sample rate list
			0, // Array element number of sample rate list
			0, //Selected sample rate index
			1, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			12, // Task ID
			"output", // Port name
			PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
			g_astPortSampleRate_RGBtoYUV_output, // Array of sample rate list
			0, // Array element number of sample rate list
			0, //Selected sample rate index
			1, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
		0, // Initial data length
		&g_stSharedMemoryChannel_14, // specific shared memory channel structure pointer
	},
};
// ##CHANNEL_LIST_TEMPLATE::END




// ##TASK_ITERATION_TEMPLATE::START
STaskIteration g_astTaskIteration_RGBtoYUV[] = {
	{
		0, // Mode ID
		1, // iteration count
	},
};

STaskIteration g_astTaskIteration_Hough[] = {
	{
		0, // Mode ID
		1, // iteration count
	},
};

STaskIteration g_astTaskIteration_Sobel[] = {
	{
		0, // Mode ID
		1, // iteration count
	},
};

STaskIteration g_astTaskIteration_YUVtoRGB[] = {
	{
		0, // Mode ID
		1, // iteration count
	},
};

STaskIteration g_astTaskIteration_KNN[] = {
	{
		0, // Mode ID
		1, // iteration count
	},
};

STaskIteration g_astTaskIteration_NLM[] = {
	{
		0, // Mode ID
		1, // iteration count
	},
};

STaskIteration g_astTaskIteration_Blending[] = {
	{
		0, // Mode ID
		1, // iteration count
	},
};

STaskIteration g_astTaskIteration_Merge[] = {
	{
		0, // Mode ID
		1, // iteration count
	},
};

STaskIteration g_astTaskIteration_StoreImage[] = {
	{
		0, // Mode ID
		1, // iteration count
	},
};

STaskIteration g_astTaskIteration_LoadImage[] = {
	{
		0, // Mode ID
		1, // iteration count
	},
};

STaskIteration g_astTaskIteration_Gaussian[] = {
	{
		0, // Mode ID
		1, // iteration count
	},
};

STaskIteration g_astTaskIteration_Sharpen[] = {
	{
		0, // Mode ID
		1, // iteration count
	},
};

STaskIteration g_astTaskIteration_DrawLane[] = {
	{
		0, // Mode ID
		1, // iteration count
	},
};

STaskIteration g_astTaskIteration_NonMax[] = {
	{
		0, // Mode ID
		1, // iteration count
	},
};

// ##TASK_ITERATION_TEMPLATE::END


// ##TASK_LIST_TEMPLATE::START
STask g_astTasks_top[] = {
	{ 	0, // Task ID
		"LoadImage", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_LoadImage_functions, // Task function array
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
		g_astTaskIteration_LoadImage, // Task iteration count (only used when the parent task graph is data flow)
		0, // current run count in iteration
		0, // current iteration
		0, // target iteration (this variable is used for calling delayed stop task)
	},
	{ 	1, // Task ID
		"YUVtoRGB", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_YUVtoRGB_functions, // Task function array
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
		g_astTaskIteration_YUVtoRGB, // Task iteration count (only used when the parent task graph is data flow)
		0, // current run count in iteration
		0, // current iteration
		0, // target iteration (this variable is used for calling delayed stop task)
	},
	{ 	2, // Task ID
		"NLM", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_NLM_functions, // Task function array
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
		g_astTaskIteration_NLM, // Task iteration count (only used when the parent task graph is data flow)
		0, // current run count in iteration
		0, // current iteration
		0, // target iteration (this variable is used for calling delayed stop task)
	},
	{ 	3, // Task ID
		"KNN", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_KNN_functions, // Task function array
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
		g_astTaskIteration_KNN, // Task iteration count (only used when the parent task graph is data flow)
		0, // current run count in iteration
		0, // current iteration
		0, // target iteration (this variable is used for calling delayed stop task)
	},
	{ 	4, // Task ID
		"Gaussian", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_Gaussian_functions, // Task function array
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
		g_astTaskIteration_Gaussian, // Task iteration count (only used when the parent task graph is data flow)
		0, // current run count in iteration
		0, // current iteration
		0, // target iteration (this variable is used for calling delayed stop task)
	},
	{ 	5, // Task ID
		"Sobel", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_Sobel_functions, // Task function array
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
		g_astTaskIteration_Sobel, // Task iteration count (only used when the parent task graph is data flow)
		0, // current run count in iteration
		0, // current iteration
		0, // target iteration (this variable is used for calling delayed stop task)
	},
	{ 	6, // Task ID
		"NonMax", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_NonMax_functions, // Task function array
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
		g_astTaskIteration_NonMax, // Task iteration count (only used when the parent task graph is data flow)
		0, // current run count in iteration
		0, // current iteration
		0, // target iteration (this variable is used for calling delayed stop task)
	},
	{ 	7, // Task ID
		"Hough", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_Hough_functions, // Task function array
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
		g_astTaskIteration_Hough, // Task iteration count (only used when the parent task graph is data flow)
		0, // current run count in iteration
		0, // current iteration
		0, // target iteration (this variable is used for calling delayed stop task)
	},
	{ 	8, // Task ID
		"DrawLane", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_DrawLane_functions, // Task function array
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
		g_astTaskIteration_DrawLane, // Task iteration count (only used when the parent task graph is data flow)
		0, // current run count in iteration
		0, // current iteration
		0, // target iteration (this variable is used for calling delayed stop task)
	},
	{ 	9, // Task ID
		"Blending", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_Blending_functions, // Task function array
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
		g_astTaskIteration_Blending, // Task iteration count (only used when the parent task graph is data flow)
		0, // current run count in iteration
		0, // current iteration
		0, // target iteration (this variable is used for calling delayed stop task)
	},
	{ 	10, // Task ID
		"Sharpen", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_Sharpen_functions, // Task function array
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
		g_astTaskIteration_Sharpen, // Task iteration count (only used when the parent task graph is data flow)
		0, // current run count in iteration
		0, // current iteration
		0, // target iteration (this variable is used for calling delayed stop task)
	},
	{ 	11, // Task ID
		"Merge", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_Merge_functions, // Task function array
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
		g_astTaskIteration_Merge, // Task iteration count (only used when the parent task graph is data flow)
		0, // current run count in iteration
		0, // current iteration
		0, // target iteration (this variable is used for calling delayed stop task)
	},
	{ 	12, // Task ID
		"RGBtoYUV", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_RGBtoYUV_functions, // Task function array
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
		g_astTaskIteration_RGBtoYUV, // Task iteration count (only used when the parent task graph is data flow)
		0, // current run count in iteration
		0, // current iteration
		0, // target iteration (this variable is used for calling delayed stop task)
	},
	{ 	13, // Task ID
		"StoreImage", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_StoreImage_functions, // Task function array
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
		g_astTaskIteration_StoreImage, // Task iteration count (only used when the parent task graph is data flow)
		0, // current run count in iteration
		0, // current iteration
		0, // target iteration (this variable is used for calling delayed stop task)
	},
};


// ##TASK_LIST_TEMPLATE::END

// ##TASK_GRAPH_TEMPLATE::START
STaskGraph g_stGraph_top = {
		GRAPH_TYPE_PROCESS_NETWORK, // Task graph type
		g_astTasks_top, // current task graph's task list
		14, // number of tasks
		NULL, // parent task
};

// ##TASK_GRAPH_TEMPLATE::END

// ##TASK_ID_TO_TASK_MAP_TEMPLATE::START
STaskIdToTaskMap g_astTaskIdToTask[] = {
	{ 	12, // Task ID
		"RGBtoYUV", // Task name
		&g_astTasks_top[12], // Task structure pointer
	},
	{ 	7, // Task ID
		"Hough", // Task name
		&g_astTasks_top[7], // Task structure pointer
	},
	{ 	5, // Task ID
		"Sobel", // Task name
		&g_astTasks_top[5], // Task structure pointer
	},
	{ 	1, // Task ID
		"YUVtoRGB", // Task name
		&g_astTasks_top[1], // Task structure pointer
	},
	{ 	3, // Task ID
		"KNN", // Task name
		&g_astTasks_top[3], // Task structure pointer
	},
	{ 	2, // Task ID
		"NLM", // Task name
		&g_astTasks_top[2], // Task structure pointer
	},
	{ 	9, // Task ID
		"Blending", // Task name
		&g_astTasks_top[9], // Task structure pointer
	},
	{ 	11, // Task ID
		"Merge", // Task name
		&g_astTasks_top[11], // Task structure pointer
	},
	{ 	13, // Task ID
		"StoreImage", // Task name
		&g_astTasks_top[13], // Task structure pointer
	},
	{ 	0, // Task ID
		"LoadImage", // Task name
		&g_astTasks_top[0], // Task structure pointer
	},
	{ 	4, // Task ID
		"Gaussian", // Task name
		&g_astTasks_top[4], // Task structure pointer
	},
	{ 	10, // Task ID
		"Sharpen", // Task name
		&g_astTasks_top[10], // Task structure pointer
	},
	{ 	8, // Task ID
		"DrawLane", // Task name
		&g_astTasks_top[8], // Task structure pointer
	},
	{ 	6, // Task ID
		"NonMax", // Task name
		&g_astTasks_top[6], // Task structure pointer
	},
};
// ##TASK_ID_TO_TASK_MAP_TEMPLATE::END


// ##PROCESSOR_INFO_TEMPLATE::START
SProcessor g_astProcessorInfo[] = {

	{ 	0, // Processor ID
		TRUE, // Processor is CPU?			
		"i7_0", // Processor name
		8, // Processor pool size
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

SMappedGeneralTaskInfo g_astGeneralTaskMappingInfo[] = {
	{	TASK_TYPE_COMPUTATIONAL, // Task type
		&g_astTasks_top[12], // Task ID or composite task information
		0, // Processor ID
		0, // Processor local ID
	},
	{	TASK_TYPE_COMPUTATIONAL, // Task type
		&g_astTasks_top[7], // Task ID or composite task information
		0, // Processor ID
		0, // Processor local ID
	},
	{	TASK_TYPE_COMPUTATIONAL, // Task type
		&g_astTasks_top[5], // Task ID or composite task information
		0, // Processor ID
		0, // Processor local ID
	},
	{	TASK_TYPE_COMPUTATIONAL, // Task type
		&g_astTasks_top[1], // Task ID or composite task information
		0, // Processor ID
		0, // Processor local ID
	},
	{	TASK_TYPE_COMPUTATIONAL, // Task type
		&g_astTasks_top[3], // Task ID or composite task information
		0, // Processor ID
		0, // Processor local ID
	},
	{	TASK_TYPE_COMPUTATIONAL, // Task type
		&g_astTasks_top[2], // Task ID or composite task information
		0, // Processor ID
		0, // Processor local ID
	},
	{	TASK_TYPE_COMPUTATIONAL, // Task type
		&g_astTasks_top[9], // Task ID or composite task information
		0, // Processor ID
		0, // Processor local ID
	},
	{	TASK_TYPE_COMPUTATIONAL, // Task type
		&g_astTasks_top[11], // Task ID or composite task information
		0, // Processor ID
		0, // Processor local ID
	},
	{	TASK_TYPE_COMPUTATIONAL, // Task type
		&g_astTasks_top[13], // Task ID or composite task information
		0, // Processor ID
		0, // Processor local ID
	},
	{	TASK_TYPE_COMPUTATIONAL, // Task type
		&g_astTasks_top[0], // Task ID or composite task information
		0, // Processor ID
		0, // Processor local ID
	},
	{	TASK_TYPE_COMPUTATIONAL, // Task type
		&g_astTasks_top[4], // Task ID or composite task information
		0, // Processor ID
		0, // Processor local ID
	},
	{	TASK_TYPE_COMPUTATIONAL, // Task type
		&g_astTasks_top[10], // Task ID or composite task information
		0, // Processor ID
		0, // Processor local ID
	},
	{	TASK_TYPE_COMPUTATIONAL, // Task type
		&g_astTasks_top[8], // Task ID or composite task information
		0, // Processor ID
		0, // Processor local ID
	},
	{	TASK_TYPE_COMPUTATIONAL, // Task type
		&g_astTasks_top[6], // Task ID or composite task information
		0, // Processor ID
		0, // Processor local ID
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
};

// ##LIBRARY_INFO_TEMPLATE::END


int g_nChannelNum = ARRAYLEN(g_astChannels);
int g_nNumOfTasks_top = ARRAYLEN(g_astTasks_top);
int g_nTaskIdToTaskNum = ARRAYLEN(g_astTaskIdToTask);
int g_nProcessorInfoNum = ARRAYLEN(g_astProcessorInfo);
int g_nLibraryInfoNum = 0;

