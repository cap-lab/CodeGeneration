/* uem_data.c made by UEM Translator */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_data.h>
#include <UKTask.h>
#include <UKModeTransition.h>

SExecutionTime g_stExecutionTime = { 10, TIME_METRIC_SEC } ;

// ##TASK_CODE_TEMPLATE::START
void Deblock_2_Init0(int nTaskId);
void Deblock_2_Go0(int nTaskId);
void Deblock_2_Wrapup0();

void Init_3_Init0(int nTaskId);
void Init_3_Go0(int nTaskId);
void Init_3_Wrapup0();

void Encoder_1_Init0(int nTaskId);
void Encoder_1_Go0(int nTaskId);
void Encoder_1_Wrapup0();

void ME_0_Init0(int nTaskId);
void ME_0_Go0(int nTaskId);
void ME_0_Wrapup0();
void ME_0_Init1(int nTaskId);
void ME_0_Go1(int nTaskId);
void ME_0_Wrapup1();
void ME_0_Init2(int nTaskId);
void ME_0_Go2(int nTaskId);
void ME_0_Wrapup2();
void ME_0_Init3(int nTaskId);
void ME_0_Go3(int nTaskId);
void ME_0_Wrapup3();

void VLC_4_Init0(int nTaskId);
void VLC_4_Go0(int nTaskId);
void VLC_4_Wrapup0();

// ##TASK_CODE_TEMPLATE::END


// ##LIBRARY_INIT_WRAPUP_TEMPLATE::START
// ##LIBRARY_INIT_WRAPUP_TEMPLATE::END


// ##CHANNEL_SIZE_DEFINITION_TEMPLATE::START
#define CHANNEL_0_SIZE (1224828)
#define CHANNEL_1_SIZE (101376)
#define CHANNEL_2_SIZE (396)
#define CHANNEL_3_SIZE (25344)
#define CHANNEL_4_SIZE (19008)
#define CHANNEL_5_SIZE (25344)
#define CHANNEL_6_SIZE (6336)
#define CHANNEL_7_SIZE (6336)
#define CHANNEL_8_SIZE (79596)
#define CHANNEL_9_SIZE (1121076)
#define CHANNEL_10_SIZE (4)
#define CHANNEL_11_SIZE (396)
#define CHANNEL_12_SIZE (396)
#define CHANNEL_13_SIZE (396)
#define CHANNEL_14_SIZE (4)
#define CHANNEL_15_SIZE (396)
#define CHANNEL_16_SIZE (396)
#define CHANNEL_17_SIZE (39204)
#define CHANNEL_18_SIZE (396)
#define CHANNEL_19_SIZE (4)
#define CHANNEL_20_SIZE (396)
#define CHANNEL_21_SIZE (396)
#define CHANNEL_22_SIZE (396)
#define CHANNEL_23_SIZE (3960)
#define CHANNEL_24_SIZE (4)
#define CHANNEL_25_SIZE (1596)
#define CHANNEL_26_SIZE (3168)
#define CHANNEL_27_SIZE (47520)
#define CHANNEL_28_SIZE (6336)
#define CHANNEL_29_SIZE (396)
#define CHANNEL_30_SIZE (396)
#define CHANNEL_31_SIZE (101376)
#define CHANNEL_32_SIZE (95040)
#define CHANNEL_33_SIZE (19008)
#define CHANNEL_34_SIZE (1224828)
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
char s_pChannel_15_buffer[CHANNEL_15_SIZE];
char s_pChannel_16_buffer[CHANNEL_16_SIZE];
char s_pChannel_17_buffer[CHANNEL_17_SIZE];
char s_pChannel_18_buffer[CHANNEL_18_SIZE];
char s_pChannel_19_buffer[CHANNEL_19_SIZE];
char s_pChannel_20_buffer[CHANNEL_20_SIZE];
char s_pChannel_21_buffer[CHANNEL_21_SIZE];
char s_pChannel_22_buffer[CHANNEL_22_SIZE];
char s_pChannel_23_buffer[CHANNEL_23_SIZE];
char s_pChannel_24_buffer[CHANNEL_24_SIZE];
char s_pChannel_25_buffer[CHANNEL_25_SIZE];
char s_pChannel_26_buffer[CHANNEL_26_SIZE];
char s_pChannel_27_buffer[CHANNEL_27_SIZE];
char s_pChannel_28_buffer[CHANNEL_28_SIZE];
char s_pChannel_29_buffer[CHANNEL_29_SIZE];
char s_pChannel_30_buffer[CHANNEL_30_SIZE];
char s_pChannel_31_buffer[CHANNEL_31_SIZE];
char s_pChannel_32_buffer[CHANNEL_32_SIZE];
char s_pChannel_33_buffer[CHANNEL_33_SIZE];
char s_pChannel_34_buffer[CHANNEL_34_SIZE];
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
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_0_buffer, // Chunk start pointer
		s_pChannel_0_buffer, // Data start pointer
		s_pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
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
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_1_buffer, // Chunk start pointer
		s_pChannel_1_buffer, // Data start pointer
		s_pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
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
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_8_buffer, // Chunk start pointer
		s_pChannel_8_buffer, // Data start pointer
		s_pChannel_8_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
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
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_9_buffer, // Chunk start pointer
		s_pChannel_9_buffer, // Data start pointer
		s_pChannel_9_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
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

SChunk g_astChunk_channel_15_out[] = {
	{
		s_pChannel_15_buffer, // Chunk start pointer
		s_pChannel_15_buffer, // Data start pointer
		s_pChannel_15_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_15_in[] = {
	{
		s_pChannel_15_buffer, // Chunk start pointer
		s_pChannel_15_buffer, // Data start pointer
		s_pChannel_15_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_16_out[] = {
	{
		s_pChannel_16_buffer, // Chunk start pointer
		s_pChannel_16_buffer, // Data start pointer
		s_pChannel_16_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_16_in[] = {
	{
		s_pChannel_16_buffer, // Chunk start pointer
		s_pChannel_16_buffer, // Data start pointer
		s_pChannel_16_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_17_out[] = {
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_17_in[] = {
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_17_buffer, // Chunk start pointer
		s_pChannel_17_buffer, // Data start pointer
		s_pChannel_17_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_18_out[] = {
	{
		s_pChannel_18_buffer, // Chunk start pointer
		s_pChannel_18_buffer, // Data start pointer
		s_pChannel_18_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_18_in[] = {
	{
		s_pChannel_18_buffer, // Chunk start pointer
		s_pChannel_18_buffer, // Data start pointer
		s_pChannel_18_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_19_out[] = {
	{
		s_pChannel_19_buffer, // Chunk start pointer
		s_pChannel_19_buffer, // Data start pointer
		s_pChannel_19_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_19_in[] = {
	{
		s_pChannel_19_buffer, // Chunk start pointer
		s_pChannel_19_buffer, // Data start pointer
		s_pChannel_19_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_20_out[] = {
	{
		s_pChannel_20_buffer, // Chunk start pointer
		s_pChannel_20_buffer, // Data start pointer
		s_pChannel_20_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_20_in[] = {
	{
		s_pChannel_20_buffer, // Chunk start pointer
		s_pChannel_20_buffer, // Data start pointer
		s_pChannel_20_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_21_out[] = {
	{
		s_pChannel_21_buffer, // Chunk start pointer
		s_pChannel_21_buffer, // Data start pointer
		s_pChannel_21_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_21_in[] = {
	{
		s_pChannel_21_buffer, // Chunk start pointer
		s_pChannel_21_buffer, // Data start pointer
		s_pChannel_21_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_22_out[] = {
	{
		s_pChannel_22_buffer, // Chunk start pointer
		s_pChannel_22_buffer, // Data start pointer
		s_pChannel_22_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_22_in[] = {
	{
		s_pChannel_22_buffer, // Chunk start pointer
		s_pChannel_22_buffer, // Data start pointer
		s_pChannel_22_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_23_out[] = {
	{
		s_pChannel_23_buffer, // Chunk start pointer
		s_pChannel_23_buffer, // Data start pointer
		s_pChannel_23_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_23_in[] = {
	{
		s_pChannel_23_buffer, // Chunk start pointer
		s_pChannel_23_buffer, // Data start pointer
		s_pChannel_23_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_24_out[] = {
	{
		s_pChannel_24_buffer, // Chunk start pointer
		s_pChannel_24_buffer, // Data start pointer
		s_pChannel_24_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_24_in[] = {
	{
		s_pChannel_24_buffer, // Chunk start pointer
		s_pChannel_24_buffer, // Data start pointer
		s_pChannel_24_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_25_out[] = {
	{
		s_pChannel_25_buffer, // Chunk start pointer
		s_pChannel_25_buffer, // Data start pointer
		s_pChannel_25_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_25_in[] = {
	{
		s_pChannel_25_buffer, // Chunk start pointer
		s_pChannel_25_buffer, // Data start pointer
		s_pChannel_25_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_26_out[] = {
	{
		s_pChannel_26_buffer, // Chunk start pointer
		s_pChannel_26_buffer, // Data start pointer
		s_pChannel_26_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_26_in[] = {
	{
		s_pChannel_26_buffer, // Chunk start pointer
		s_pChannel_26_buffer, // Data start pointer
		s_pChannel_26_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_27_out[] = {
	{
		s_pChannel_27_buffer, // Chunk start pointer
		s_pChannel_27_buffer, // Data start pointer
		s_pChannel_27_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_27_in[] = {
	{
		s_pChannel_27_buffer, // Chunk start pointer
		s_pChannel_27_buffer, // Data start pointer
		s_pChannel_27_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_28_out[] = {
	{
		s_pChannel_28_buffer, // Chunk start pointer
		s_pChannel_28_buffer, // Data start pointer
		s_pChannel_28_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_28_in[] = {
	{
		s_pChannel_28_buffer, // Chunk start pointer
		s_pChannel_28_buffer, // Data start pointer
		s_pChannel_28_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_29_out[] = {
	{
		s_pChannel_29_buffer, // Chunk start pointer
		s_pChannel_29_buffer, // Data start pointer
		s_pChannel_29_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_29_in[] = {
	{
		s_pChannel_29_buffer, // Chunk start pointer
		s_pChannel_29_buffer, // Data start pointer
		s_pChannel_29_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_30_out[] = {
	{
		s_pChannel_30_buffer, // Chunk start pointer
		s_pChannel_30_buffer, // Data start pointer
		s_pChannel_30_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_30_in[] = {
	{
		s_pChannel_30_buffer, // Chunk start pointer
		s_pChannel_30_buffer, // Data start pointer
		s_pChannel_30_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_31_out[] = {
	{
		s_pChannel_31_buffer, // Chunk start pointer
		s_pChannel_31_buffer, // Data start pointer
		s_pChannel_31_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_31_in[] = {
	{
		s_pChannel_31_buffer, // Chunk start pointer
		s_pChannel_31_buffer, // Data start pointer
		s_pChannel_31_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_32_out[] = {
	{
		s_pChannel_32_buffer, // Chunk start pointer
		s_pChannel_32_buffer, // Data start pointer
		s_pChannel_32_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_32_in[] = {
	{
		s_pChannel_32_buffer, // Chunk start pointer
		s_pChannel_32_buffer, // Data start pointer
		s_pChannel_32_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_33_out[] = {
	{
		s_pChannel_33_buffer, // Chunk start pointer
		s_pChannel_33_buffer, // Data start pointer
		s_pChannel_33_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_33_in[] = {
	{
		s_pChannel_33_buffer, // Chunk start pointer
		s_pChannel_33_buffer, // Data start pointer
		s_pChannel_33_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_34_out[] = {
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_34_in[] = {
	{
		s_pChannel_34_buffer, // Chunk start pointer
		s_pChannel_34_buffer, // Data start pointer
		s_pChannel_34_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

// ##CHUNK_DEFINITION_TEMPLATE::END
//portSampleRateList


// ##PORT_SAMPLE_RATE_TEMPLATE::START
SPortSampleRate g_astPortSampleRate_Encoder_1_p52[] = {
	{ 	"Default", // Mode name
		99, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_ME_0_p105[] = {
	{ 	"Default", // Mode name
		99, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Encoder_1_p53[] = {
	{ 	"Default", // Mode name
		99, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_ME_0_p109[] = {
	{ 	"Default", // Mode name
		99, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Deblock_2_p40[] = {
	{ 	"Default", // Mode name
		99, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Encoder_1_p54[] = {
	{ 	"Default", // Mode name
		99, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Deblock_2_p23[] = {
	{ 	"Default", // Mode name
		99, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Encoder_1_p20[] = {
	{ 	"Default", // Mode name
		99, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Deblock_2_p27[] = {
	{ 	"Default", // Mode name
		99, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Encoder_1_p40[] = {
	{ 	"Default", // Mode name
		99, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Deblock_2_p22[] = {
	{ 	"Default", // Mode name
		99, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Encoder_1_p5[] = {
	{ 	"Default", // Mode name
		99, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Deblock_2_p24[] = {
	{ 	"Default", // Mode name
		99, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Encoder_1_p28[] = {
	{ 	"Default", // Mode name
		99, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Deblock_2_p25[] = {
	{ 	"Default", // Mode name
		99, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Encoder_1_p29[] = {
	{ 	"Default", // Mode name
		99, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Deblock_2_p41[] = {
	{ 	"Default", // Mode name
		99, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_ME_0_p107[] = {
	{ 	"Default", // Mode name
		99, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_ME_0_p101[] = {
	{ 	"Default", // Mode name
		99, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Deblock_2_p42[] = {
	{ 	"Default", // Mode name
		99, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Encoder_1_p18[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Init_3_p58[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Encoder_1_p7[] = {
	{ 	"Default", // Mode name
		99, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Init_3_p55[] = {
	{ 	"Default", // Mode name
		99, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Encoder_1_p30[] = {
	{ 	"Default", // Mode name
		99, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Init_3_p59[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Encoder_1_p22[] = {
	{ 	"Default", // Mode name
		99, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Init_3_p60[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Encoder_1_p6[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Init_3_p49[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Encoder_1_p4[] = {
	{ 	"Default", // Mode name
		99, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Init_3_p50[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Encoder_1_p3[] = {
	{ 	"Default", // Mode name
		99, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Init_3_p51[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_ME_0_p102[] = {
	{ 	"Default", // Mode name
		99, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Init_3_p71[] = {
	{ 	"Default", // Mode name
		99, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Deblock_2_p18[] = {
	{ 	"Default", // Mode name
		99, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Init_3_p61[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Deblock_2_p33[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Init_3_p62[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Encoder_1_p19[] = {
	{ 	"Default", // Mode name
		99, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Init_3_p52[] = {
	{ 	"Default", // Mode name
		99, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Encoder_1_p17[] = {
	{ 	"Default", // Mode name
		99, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Init_3_p53[] = {
	{ 	"Default", // Mode name
		99, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Encoder_1_p15[] = {
	{ 	"Default", // Mode name
		99, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Init_3_p54[] = {
	{ 	"Default", // Mode name
		99, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_VLC_4_p17[] = {
	{ 	"Default", // Mode name
		99, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Init_3_p63[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_VLC_4_p18[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Init_3_p67[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_VLC_4_p0[] = {
	{ 	"Default", // Mode name
		99, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Init_3_p68[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_VLC_4_p14[] = {
	{ 	"Default", // Mode name
		99, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Encoder_1_p43[] = {
	{ 	"Default", // Mode name
		99, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_VLC_4_p16[] = {
	{ 	"Default", // Mode name
		99, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Encoder_1_p42[] = {
	{ 	"Default", // Mode name
		99, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_VLC_4_p12[] = {
	{ 	"Default", // Mode name
		99, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Encoder_1_p41[] = {
	{ 	"Default", // Mode name
		99, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_VLC_4_p7[] = {
	{ 	"Default", // Mode name
		99, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Encoder_1_p35[] = {
	{ 	"Default", // Mode name
		99, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_VLC_4_p8[] = {
	{ 	"Default", // Mode name
		99, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Encoder_1_p36[] = {
	{ 	"Default", // Mode name
		99, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_VLC_4_p13[] = {
	{ 	"Default", // Mode name
		99, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Encoder_1_p37[] = {
	{ 	"Default", // Mode name
		99, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_VLC_4_p15[] = {
	{ 	"Default", // Mode name
		99, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Encoder_1_p38[] = {
	{ 	"Default", // Mode name
		99, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_VLC_4_p6[] = {
	{ 	"Default", // Mode name
		99, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Encoder_1_p39[] = {
	{ 	"Default", // Mode name
		99, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_VLC_4_p60[] = {
	{ 	"Default", // Mode name
		99, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_ME_0_p106[] = {
	{ 	"Default", // Mode name
		99, // Sample rate
		1, // Available number of data
	},
};

// ##PORT_SAMPLE_RATE_TEMPLATE::END


// ##PORT_ARRAY_TEMPLATE::START
SPort g_astPortInfo[] = {
	{
		1, // Task ID
		"p52", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Encoder_1_p52, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		12372, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		0, // Task ID
		"p105", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_ME_0_p105, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		12372, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		1, // Task ID
		"p53", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Encoder_1_p53, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		396, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		0, // Task ID
		"p109", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_ME_0_p109, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		396, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		2, // Task ID
		"p40", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Deblock_2_p40, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		1, // Task ID
		"p54", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Encoder_1_p54, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		2, // Task ID
		"p23", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Deblock_2_p23, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		256, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		1, // Task ID
		"p20", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Encoder_1_p20, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		256, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		2, // Task ID
		"p27", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Deblock_2_p27, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		192, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		1, // Task ID
		"p40", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Encoder_1_p40, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		192, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		2, // Task ID
		"p22", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Deblock_2_p22, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		256, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		1, // Task ID
		"p5", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Encoder_1_p5, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		256, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		2, // Task ID
		"p24", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Deblock_2_p24, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		64, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		1, // Task ID
		"p28", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Encoder_1_p28, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		64, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		2, // Task ID
		"p25", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Deblock_2_p25, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		64, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		1, // Task ID
		"p29", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Encoder_1_p29, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		64, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		2, // Task ID
		"p41", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Deblock_2_p41, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		804, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		0, // Task ID
		"p107", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_ME_0_p107, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		804, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		0, // Task ID
		"p101", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_ME_0_p101, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		11324, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		2, // Task ID
		"p42", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Deblock_2_p42, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		11324, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		1, // Task ID
		"p18", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Encoder_1_p18, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		3, // Task ID
		"p58", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Init_3_p58, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		1, // Task ID
		"p7", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Encoder_1_p7, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		3, // Task ID
		"p55", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Init_3_p55, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		1, // Task ID
		"p30", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Encoder_1_p30, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		3, // Task ID
		"p59", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Init_3_p59, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		1, // Task ID
		"p22", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Encoder_1_p22, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		3, // Task ID
		"p60", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Init_3_p60, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		1, // Task ID
		"p6", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Encoder_1_p6, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		3, // Task ID
		"p49", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Init_3_p49, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		1, // Task ID
		"p4", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Encoder_1_p4, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		3, // Task ID
		"p50", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Init_3_p50, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		1, // Task ID
		"p3", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Encoder_1_p3, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		3, // Task ID
		"p51", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Init_3_p51, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		0, // Task ID
		"p102", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_ME_0_p102, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		396, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		3, // Task ID
		"p71", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Init_3_p71, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		396, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		2, // Task ID
		"p18", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Deblock_2_p18, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		3, // Task ID
		"p61", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Init_3_p61, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		2, // Task ID
		"p33", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Deblock_2_p33, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		3, // Task ID
		"p62", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Init_3_p62, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		1, // Task ID
		"p19", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Encoder_1_p19, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		3, // Task ID
		"p52", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Init_3_p52, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		1, // Task ID
		"p17", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Encoder_1_p17, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		3, // Task ID
		"p53", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Init_3_p53, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		1, // Task ID
		"p15", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Encoder_1_p15, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		3, // Task ID
		"p54", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Init_3_p54, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		4, // Task ID
		"p17", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_VLC_4_p17, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		40, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		3, // Task ID
		"p63", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Init_3_p63, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		40, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		4, // Task ID
		"p18", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_VLC_4_p18, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		3, // Task ID
		"p67", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Init_3_p67, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		4, // Task ID
		"p0", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_VLC_4_p0, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		3, // Task ID
		"p68", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Init_3_p68, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		4, // Task ID
		"p14", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_VLC_4_p14, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		32, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		1, // Task ID
		"p43", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Encoder_1_p43, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		32, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		4, // Task ID
		"p16", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_VLC_4_p16, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		480, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		1, // Task ID
		"p42", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Encoder_1_p42, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		480, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		4, // Task ID
		"p12", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_VLC_4_p12, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		64, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		1, // Task ID
		"p41", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Encoder_1_p41, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		64, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		4, // Task ID
		"p7", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_VLC_4_p7, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		1, // Task ID
		"p35", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Encoder_1_p35, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		4, // Task ID
		"p8", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_VLC_4_p8, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		1, // Task ID
		"p36", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Encoder_1_p36, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		4, // Task ID
		"p13", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_VLC_4_p13, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		1024, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		1, // Task ID
		"p37", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Encoder_1_p37, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		1024, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		4, // Task ID
		"p15", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_VLC_4_p15, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		960, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		1, // Task ID
		"p38", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Encoder_1_p38, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		960, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		4, // Task ID
		"p6", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_VLC_4_p6, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		192, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		1, // Task ID
		"p39", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_Encoder_1_p39, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		192, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		4, // Task ID
		"p60", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_VLC_4_p60, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		12372, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		0, // Task ID
		"p106", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_ME_0_p106, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		12372, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
};
// ##PORT_ARRAY_TEMPLATE::END


// ##LOOP_STRUCTURE_TEMPLATE::START
SLoopInfo g_stLoopStruct_ME_0 = {
	LOOP_TYPE_DATA,
	99,
	0,
};

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
	{ 1, 0, NULL, NULL, },
	{ 2, 0, NULL, NULL, },
	{ 3, 0, NULL, NULL, },
	{ 4, 0, NULL, NULL, },
	{ 5, 0, NULL, NULL, },
	{ 6, 0, NULL, NULL, },
	{ 7, 0, NULL, NULL, },
	{ 8, 0, NULL, NULL, },
	{ 9, 0, NULL, NULL, },
	{ 10, 0, NULL, NULL, },
	{ 11, 0, NULL, NULL, },
	{ 12, 0, NULL, NULL, },
	{ 13, 0, NULL, NULL, },
	{ 14, 0, NULL, NULL, },
	{ 15, 0, NULL, NULL, },
	{ 16, 0, NULL, NULL, },
	{ 17, 0, NULL, NULL, },
	{ 18, 0, NULL, NULL, },
	{ 19, 0, NULL, NULL, },
	{ 20, 0, NULL, NULL, },
	{ 21, 0, NULL, NULL, },
	{ 22, 0, NULL, NULL, },
	{ 23, 0, NULL, NULL, },
	{ 24, 0, NULL, NULL, },
	{ 25, 0, NULL, NULL, },
	{ 26, 0, NULL, NULL, },
	{ 27, 0, NULL, NULL, },
	{ 28, 0, NULL, NULL, },
	{ 29, 0, NULL, NULL, },
	{ 30, 0, NULL, NULL, },
	{ 31, 0, NULL, NULL, },
	{ 32, 0, NULL, NULL, },
	{ 33, 0, NULL, NULL, },
	{ 34, 0, NULL, NULL, },
	{ 35, 0, NULL, NULL, },
	{ 36, 0, NULL, NULL, },
	{ 37, 0, NULL, NULL, },
	{ 38, 0, NULL, NULL, },
	{ 39, 0, NULL, NULL, },
	{ 40, 0, NULL, NULL, },
	{ 41, 0, NULL, NULL, },
	{ 42, 0, NULL, NULL, },
	{ 43, 0, NULL, NULL, },
	{ 44, 0, NULL, NULL, },
	{ 45, 0, NULL, NULL, },
	{ 46, 0, NULL, NULL, },
	{ 47, 0, NULL, NULL, },
	{ 48, 0, NULL, NULL, },
	{ 49, 0, NULL, NULL, },
	{ 50, 0, NULL, NULL, },
	{ 51, 0, NULL, NULL, },
	{ 52, 0, NULL, NULL, },
	{ 53, 0, NULL, NULL, },
	{ 54, 0, NULL, NULL, },
	{ 55, 0, NULL, NULL, },
	{ 56, 0, NULL, NULL, },
	{ 57, 0, NULL, NULL, },
	{ 58, 0, NULL, NULL, },
	{ 59, 0, NULL, NULL, },
	{ 60, 0, NULL, NULL, },
	{ 61, 0, NULL, NULL, },
	{ 62, 0, NULL, NULL, },
	{ 63, 0, NULL, NULL, },
	{ 64, 0, NULL, NULL, },
	{ 65, 0, NULL, NULL, },
	{ 66, 0, NULL, NULL, },
	{ 67, 0, NULL, NULL, },
	{ 68, 0, NULL, NULL, },
	{ 69, 0, NULL, NULL, },
	{ 70, 0, NULL, NULL, },
	{ 71, 0, NULL, NULL, },
	{ 72, 0, NULL, NULL, },
	{ 73, 0, NULL, NULL, },
	{ 74, 0, NULL, NULL, },
	{ 75, 0, NULL, NULL, },
	{ 76, 0, NULL, NULL, },
	{ 77, 0, NULL, NULL, },
	{ 78, 0, NULL, NULL, },
	{ 79, 0, NULL, NULL, },
	{ 80, 0, NULL, NULL, },
	{ 81, 0, NULL, NULL, },
	{ 82, 0, NULL, NULL, },
	{ 83, 0, NULL, NULL, },
	{ 84, 0, NULL, NULL, },
	{ 85, 0, NULL, NULL, },
	{ 86, 0, NULL, NULL, },
	{ 87, 0, NULL, NULL, },
	{ 88, 0, NULL, NULL, },
	{ 89, 0, NULL, NULL, },
	{ 90, 0, NULL, NULL, },
	{ 91, 0, NULL, NULL, },
	{ 92, 0, NULL, NULL, },
	{ 93, 0, NULL, NULL, },
	{ 94, 0, NULL, NULL, },
	{ 95, 0, NULL, NULL, },
	{ 96, 0, NULL, NULL, },
	{ 97, 0, NULL, NULL, },
	{ 98, 0, NULL, NULL, },
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
SAvailableChunk g_astAvailableInputChunk_channel_15[] = {
	{ 0, 0, NULL, NULL, },
};
SAvailableChunk g_astAvailableInputChunk_channel_16[] = {
	{ 0, 0, NULL, NULL, },
};
SAvailableChunk g_astAvailableInputChunk_channel_17[] = {
	{ 0, 0, NULL, NULL, },
	{ 1, 0, NULL, NULL, },
	{ 2, 0, NULL, NULL, },
	{ 3, 0, NULL, NULL, },
	{ 4, 0, NULL, NULL, },
	{ 5, 0, NULL, NULL, },
	{ 6, 0, NULL, NULL, },
	{ 7, 0, NULL, NULL, },
	{ 8, 0, NULL, NULL, },
	{ 9, 0, NULL, NULL, },
	{ 10, 0, NULL, NULL, },
	{ 11, 0, NULL, NULL, },
	{ 12, 0, NULL, NULL, },
	{ 13, 0, NULL, NULL, },
	{ 14, 0, NULL, NULL, },
	{ 15, 0, NULL, NULL, },
	{ 16, 0, NULL, NULL, },
	{ 17, 0, NULL, NULL, },
	{ 18, 0, NULL, NULL, },
	{ 19, 0, NULL, NULL, },
	{ 20, 0, NULL, NULL, },
	{ 21, 0, NULL, NULL, },
	{ 22, 0, NULL, NULL, },
	{ 23, 0, NULL, NULL, },
	{ 24, 0, NULL, NULL, },
	{ 25, 0, NULL, NULL, },
	{ 26, 0, NULL, NULL, },
	{ 27, 0, NULL, NULL, },
	{ 28, 0, NULL, NULL, },
	{ 29, 0, NULL, NULL, },
	{ 30, 0, NULL, NULL, },
	{ 31, 0, NULL, NULL, },
	{ 32, 0, NULL, NULL, },
	{ 33, 0, NULL, NULL, },
	{ 34, 0, NULL, NULL, },
	{ 35, 0, NULL, NULL, },
	{ 36, 0, NULL, NULL, },
	{ 37, 0, NULL, NULL, },
	{ 38, 0, NULL, NULL, },
	{ 39, 0, NULL, NULL, },
	{ 40, 0, NULL, NULL, },
	{ 41, 0, NULL, NULL, },
	{ 42, 0, NULL, NULL, },
	{ 43, 0, NULL, NULL, },
	{ 44, 0, NULL, NULL, },
	{ 45, 0, NULL, NULL, },
	{ 46, 0, NULL, NULL, },
	{ 47, 0, NULL, NULL, },
	{ 48, 0, NULL, NULL, },
	{ 49, 0, NULL, NULL, },
	{ 50, 0, NULL, NULL, },
	{ 51, 0, NULL, NULL, },
	{ 52, 0, NULL, NULL, },
	{ 53, 0, NULL, NULL, },
	{ 54, 0, NULL, NULL, },
	{ 55, 0, NULL, NULL, },
	{ 56, 0, NULL, NULL, },
	{ 57, 0, NULL, NULL, },
	{ 58, 0, NULL, NULL, },
	{ 59, 0, NULL, NULL, },
	{ 60, 0, NULL, NULL, },
	{ 61, 0, NULL, NULL, },
	{ 62, 0, NULL, NULL, },
	{ 63, 0, NULL, NULL, },
	{ 64, 0, NULL, NULL, },
	{ 65, 0, NULL, NULL, },
	{ 66, 0, NULL, NULL, },
	{ 67, 0, NULL, NULL, },
	{ 68, 0, NULL, NULL, },
	{ 69, 0, NULL, NULL, },
	{ 70, 0, NULL, NULL, },
	{ 71, 0, NULL, NULL, },
	{ 72, 0, NULL, NULL, },
	{ 73, 0, NULL, NULL, },
	{ 74, 0, NULL, NULL, },
	{ 75, 0, NULL, NULL, },
	{ 76, 0, NULL, NULL, },
	{ 77, 0, NULL, NULL, },
	{ 78, 0, NULL, NULL, },
	{ 79, 0, NULL, NULL, },
	{ 80, 0, NULL, NULL, },
	{ 81, 0, NULL, NULL, },
	{ 82, 0, NULL, NULL, },
	{ 83, 0, NULL, NULL, },
	{ 84, 0, NULL, NULL, },
	{ 85, 0, NULL, NULL, },
	{ 86, 0, NULL, NULL, },
	{ 87, 0, NULL, NULL, },
	{ 88, 0, NULL, NULL, },
	{ 89, 0, NULL, NULL, },
	{ 90, 0, NULL, NULL, },
	{ 91, 0, NULL, NULL, },
	{ 92, 0, NULL, NULL, },
	{ 93, 0, NULL, NULL, },
	{ 94, 0, NULL, NULL, },
	{ 95, 0, NULL, NULL, },
	{ 96, 0, NULL, NULL, },
	{ 97, 0, NULL, NULL, },
	{ 98, 0, NULL, NULL, },
};
SAvailableChunk g_astAvailableInputChunk_channel_18[] = {
	{ 0, 0, NULL, NULL, },
};
SAvailableChunk g_astAvailableInputChunk_channel_19[] = {
	{ 0, 0, NULL, NULL, },
};
SAvailableChunk g_astAvailableInputChunk_channel_20[] = {
	{ 0, 0, NULL, NULL, },
};
SAvailableChunk g_astAvailableInputChunk_channel_21[] = {
	{ 0, 0, NULL, NULL, },
};
SAvailableChunk g_astAvailableInputChunk_channel_22[] = {
	{ 0, 0, NULL, NULL, },
};
SAvailableChunk g_astAvailableInputChunk_channel_23[] = {
	{ 0, 0, NULL, NULL, },
};
SAvailableChunk g_astAvailableInputChunk_channel_24[] = {
	{ 0, 0, NULL, NULL, },
};
SAvailableChunk g_astAvailableInputChunk_channel_25[] = {
	{ 0, 0, NULL, NULL, },
};
SAvailableChunk g_astAvailableInputChunk_channel_26[] = {
	{ 0, 0, NULL, NULL, },
};
SAvailableChunk g_astAvailableInputChunk_channel_27[] = {
	{ 0, 0, NULL, NULL, },
};
SAvailableChunk g_astAvailableInputChunk_channel_28[] = {
	{ 0, 0, NULL, NULL, },
};
SAvailableChunk g_astAvailableInputChunk_channel_29[] = {
	{ 0, 0, NULL, NULL, },
};
SAvailableChunk g_astAvailableInputChunk_channel_30[] = {
	{ 0, 0, NULL, NULL, },
};
SAvailableChunk g_astAvailableInputChunk_channel_31[] = {
	{ 0, 0, NULL, NULL, },
};
SAvailableChunk g_astAvailableInputChunk_channel_32[] = {
	{ 0, 0, NULL, NULL, },
};
SAvailableChunk g_astAvailableInputChunk_channel_33[] = {
	{ 0, 0, NULL, NULL, },
};
SAvailableChunk g_astAvailableInputChunk_channel_34[] = {
	{ 0, 0, NULL, NULL, },
};
// ##AVAILABLE_CHUNK_LIST_TEMPLATE::END

// ##TASK_PARAMETER_TEMPLATE::START
// ##TASK_PARAMETER_TEMPLATE::END

// ##TASK_FUNCTION_LIST::START
STaskFunctions g_ast_Deblock_2_functions[] = {
	{
		Deblock_2_Init0, // Task init function
		Deblock_2_Go0, // Task go function
		Deblock_2_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_Init_3_functions[] = {
	{
		Init_3_Init0, // Task init function
		Init_3_Go0, // Task go function
		Init_3_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_Encoder_1_functions[] = {
	{
		Encoder_1_Init0, // Task init function
		Encoder_1_Go0, // Task go function
		Encoder_1_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_ME_0_functions[] = {
	{
		ME_0_Init0, // Task init function
		ME_0_Go0, // Task go function
		ME_0_Wrapup0, // Task wrapup function
	},
	{
		ME_0_Init1, // Task init function
		ME_0_Go1, // Task go function
		ME_0_Wrapup1, // Task wrapup function
	},
	{
		ME_0_Init2, // Task init function
		ME_0_Go2, // Task go function
		ME_0_Wrapup2, // Task wrapup function
	},
	{
		ME_0_Init3, // Task init function
		ME_0_Go3, // Task go function
		ME_0_Wrapup3, // Task wrapup function
	},
};

STaskFunctions g_ast_VLC_4_functions[] = {
	{
		VLC_4_Init0, // Task init function
		VLC_4_Go0, // Task go function
		VLC_4_Wrapup0, // Task wrapup function
	},
};

// ##TASK_FUNCTION_LIST::END


// ##CHANNEL_LIST_TEMPLATE::START
SChannel g_astChannels[] = {
	{
		0, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_OUTPUT_ARRAY, // Channel type
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
			"p52", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Encoder_1_p52, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			12372, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			0, // Task ID
			"p105", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_ME_0_p105, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			12372, // Sample size
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
		1, // maximum input port chunk size for all port sample rate cases
		NULL, // Chunk list head
		NULL, // Chunk list tail
		0, // Initial data length 
	},
	{
		1, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_OUTPUT_ARRAY, // Channel type
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
			"p53", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Encoder_1_p53, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			396, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			0, // Task ID
			"p109", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_ME_0_p109, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			396, // Sample size
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
		1, // maximum input port chunk size for all port sample rate cases
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
			2, // Task ID
			"p40", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Deblock_2_p40, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			1, // Task ID
			"p54", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Encoder_1_p54, // Array of sample rate list
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
		1, // maximum input port chunk size for all port sample rate cases
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
			2, // Task ID
			"p23", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Deblock_2_p23, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			256, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			1, // Task ID
			"p20", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Encoder_1_p20, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			256, // Sample size
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
		1, // maximum input port chunk size for all port sample rate cases
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
			"p27", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Deblock_2_p27, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			192, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			1, // Task ID
			"p40", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Encoder_1_p40, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			192, // Sample size
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
		1, // maximum input port chunk size for all port sample rate cases
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
			"p22", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Deblock_2_p22, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			256, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			1, // Task ID
			"p5", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Encoder_1_p5, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			256, // Sample size
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
		1, // maximum input port chunk size for all port sample rate cases
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
			2, // Task ID
			"p24", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Deblock_2_p24, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			64, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			1, // Task ID
			"p28", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Encoder_1_p28, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			64, // Sample size
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
		1, // maximum input port chunk size for all port sample rate cases
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
			2, // Task ID
			"p25", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Deblock_2_p25, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			64, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			1, // Task ID
			"p29", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Encoder_1_p29, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			64, // Sample size
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
		1, // maximum input port chunk size for all port sample rate cases
		NULL, // Chunk list head
		NULL, // Chunk list tail
		0, // Initial data length 
	},
	{
		8, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_OUTPUT_ARRAY, // Channel type
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
			2, // Task ID
			"p41", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Deblock_2_p41, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			804, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			0, // Task ID
			"p107", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_ME_0_p107, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			804, // Sample size
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
		1, // maximum input port chunk size for all port sample rate cases
		NULL, // Chunk list head
		NULL, // Chunk list tail
		0, // Initial data length 
	},
	{
		9, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_INPUT_ARRAY, // Channel type
		s_pChannel_9_buffer, // Channel buffer pointer
		CHANNEL_9_SIZE, // Channel size
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
			0, // Task ID
			"p101", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_ME_0_p101, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			11324, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			2, // Task ID
			"p42", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Deblock_2_p42, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			11324, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
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
		99, // maximum input port chunk size for all port sample rate cases
		NULL, // Chunk list head
		NULL, // Chunk list tail
		11324*99, // Initial data length
	},
	{
		10, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_GENERAL, // Channel type
		s_pChannel_10_buffer, // Channel buffer pointer
		CHANNEL_10_SIZE, // Channel size
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
			1, // Task ID
			"p18", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Encoder_1_p18, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			3, // Task ID
			"p58", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Init_3_p58, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
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
		0, // Initial data length 
	},
	{
		11, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_GENERAL, // Channel type
		s_pChannel_11_buffer, // Channel buffer pointer
		CHANNEL_11_SIZE, // Channel size
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
			1, // Task ID
			"p7", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Encoder_1_p7, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			3, // Task ID
			"p55", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Init_3_p55, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
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
		0, // Initial data length 
	},
	{
		12, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_GENERAL, // Channel type
		s_pChannel_12_buffer, // Channel buffer pointer
		CHANNEL_12_SIZE, // Channel size
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
			1, // Task ID
			"p30", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Encoder_1_p30, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			3, // Task ID
			"p59", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Init_3_p59, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
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
		0, // Initial data length 
	},
	{
		13, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_GENERAL, // Channel type
		s_pChannel_13_buffer, // Channel buffer pointer
		CHANNEL_13_SIZE, // Channel size
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
			1, // Task ID
			"p22", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Encoder_1_p22, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			3, // Task ID
			"p60", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Init_3_p60, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
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
		0, // Initial data length 
	},
	{
		14, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_GENERAL, // Channel type
		s_pChannel_14_buffer, // Channel buffer pointer
		CHANNEL_14_SIZE, // Channel size
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
			1, // Task ID
			"p6", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Encoder_1_p6, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			3, // Task ID
			"p49", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Init_3_p49, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
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
		0, // Initial data length 
	},
	{
		15, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_GENERAL, // Channel type
		s_pChannel_15_buffer, // Channel buffer pointer
		CHANNEL_15_SIZE, // Channel size
		s_pChannel_15_buffer, // Channel data start
		s_pChannel_15_buffer, // Channel data end
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
			"p4", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Encoder_1_p4, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			3, // Task ID
			"p50", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Init_3_p50, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
		{
			g_astChunk_channel_15_in, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Input chunk information
		{
			g_astChunk_channel_15_out, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Output chunk information
		CHUNK_NUM_NOT_INITIALIZED, // Written output chunk number
		g_astAvailableInputChunk_channel_15, // Available chunk list
		1, // maximum input port chunk size for all port sample rate cases
		NULL, // Chunk list head
		NULL, // Chunk list tail
		0, // Initial data length 
	},
	{
		16, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_GENERAL, // Channel type
		s_pChannel_16_buffer, // Channel buffer pointer
		CHANNEL_16_SIZE, // Channel size
		s_pChannel_16_buffer, // Channel data start
		s_pChannel_16_buffer, // Channel data end
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
			"p3", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Encoder_1_p3, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			3, // Task ID
			"p51", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Init_3_p51, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
		{
			g_astChunk_channel_16_in, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Input chunk information
		{
			g_astChunk_channel_16_out, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Output chunk information
		CHUNK_NUM_NOT_INITIALIZED, // Written output chunk number
		g_astAvailableInputChunk_channel_16, // Available chunk list
		1, // maximum input port chunk size for all port sample rate cases
		NULL, // Chunk list head
		NULL, // Chunk list tail
		0, // Initial data length 
	},
	{
		17, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_INPUT_ARRAY, // Channel type
		s_pChannel_17_buffer, // Channel buffer pointer
		CHANNEL_17_SIZE, // Channel size
		s_pChannel_17_buffer, // Channel data start
		s_pChannel_17_buffer, // Channel data end
		0, // Channel data length
		0, // Read reference count
		0, // Write reference count
		FALSE, // Read exit setting
		FALSE, // Write exit setting
		NULL, // Mutex
		NULL, // Read available notice event
		NULL, // Write available notice event
		{
			0, // Task ID
			"p102", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_ME_0_p102, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			396, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			3, // Task ID
			"p71", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Init_3_p71, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			396, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
		{
			g_astChunk_channel_17_in, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Input chunk information
		{
			g_astChunk_channel_17_out, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Output chunk information
		CHUNK_NUM_NOT_INITIALIZED, // Written output chunk number
		g_astAvailableInputChunk_channel_17, // Available chunk list
		99, // maximum input port chunk size for all port sample rate cases
		NULL, // Chunk list head
		NULL, // Chunk list tail
		0, // Initial data length 
	},
	{
		18, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_GENERAL, // Channel type
		s_pChannel_18_buffer, // Channel buffer pointer
		CHANNEL_18_SIZE, // Channel size
		s_pChannel_18_buffer, // Channel data start
		s_pChannel_18_buffer, // Channel data end
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
			"p18", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Deblock_2_p18, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			3, // Task ID
			"p61", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Init_3_p61, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
		{
			g_astChunk_channel_18_in, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Input chunk information
		{
			g_astChunk_channel_18_out, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Output chunk information
		CHUNK_NUM_NOT_INITIALIZED, // Written output chunk number
		g_astAvailableInputChunk_channel_18, // Available chunk list
		1, // maximum input port chunk size for all port sample rate cases
		NULL, // Chunk list head
		NULL, // Chunk list tail
		0, // Initial data length 
	},
	{
		19, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_GENERAL, // Channel type
		s_pChannel_19_buffer, // Channel buffer pointer
		CHANNEL_19_SIZE, // Channel size
		s_pChannel_19_buffer, // Channel data start
		s_pChannel_19_buffer, // Channel data end
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
			"p33", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Deblock_2_p33, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			3, // Task ID
			"p62", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Init_3_p62, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
		{
			g_astChunk_channel_19_in, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Input chunk information
		{
			g_astChunk_channel_19_out, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Output chunk information
		CHUNK_NUM_NOT_INITIALIZED, // Written output chunk number
		g_astAvailableInputChunk_channel_19, // Available chunk list
		1, // maximum input port chunk size for all port sample rate cases
		NULL, // Chunk list head
		NULL, // Chunk list tail
		0, // Initial data length 
	},
	{
		20, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_GENERAL, // Channel type
		s_pChannel_20_buffer, // Channel buffer pointer
		CHANNEL_20_SIZE, // Channel size
		s_pChannel_20_buffer, // Channel data start
		s_pChannel_20_buffer, // Channel data end
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
			"p19", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Encoder_1_p19, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			3, // Task ID
			"p52", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Init_3_p52, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
		{
			g_astChunk_channel_20_in, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Input chunk information
		{
			g_astChunk_channel_20_out, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Output chunk information
		CHUNK_NUM_NOT_INITIALIZED, // Written output chunk number
		g_astAvailableInputChunk_channel_20, // Available chunk list
		1, // maximum input port chunk size for all port sample rate cases
		NULL, // Chunk list head
		NULL, // Chunk list tail
		0, // Initial data length 
	},
	{
		21, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_GENERAL, // Channel type
		s_pChannel_21_buffer, // Channel buffer pointer
		CHANNEL_21_SIZE, // Channel size
		s_pChannel_21_buffer, // Channel data start
		s_pChannel_21_buffer, // Channel data end
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
			"p17", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Encoder_1_p17, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			3, // Task ID
			"p53", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Init_3_p53, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
		{
			g_astChunk_channel_21_in, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Input chunk information
		{
			g_astChunk_channel_21_out, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Output chunk information
		CHUNK_NUM_NOT_INITIALIZED, // Written output chunk number
		g_astAvailableInputChunk_channel_21, // Available chunk list
		1, // maximum input port chunk size for all port sample rate cases
		NULL, // Chunk list head
		NULL, // Chunk list tail
		0, // Initial data length 
	},
	{
		22, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_GENERAL, // Channel type
		s_pChannel_22_buffer, // Channel buffer pointer
		CHANNEL_22_SIZE, // Channel size
		s_pChannel_22_buffer, // Channel data start
		s_pChannel_22_buffer, // Channel data end
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
			"p15", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Encoder_1_p15, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			3, // Task ID
			"p54", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Init_3_p54, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
		{
			g_astChunk_channel_22_in, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Input chunk information
		{
			g_astChunk_channel_22_out, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Output chunk information
		CHUNK_NUM_NOT_INITIALIZED, // Written output chunk number
		g_astAvailableInputChunk_channel_22, // Available chunk list
		1, // maximum input port chunk size for all port sample rate cases
		NULL, // Chunk list head
		NULL, // Chunk list tail
		0, // Initial data length 
	},
	{
		23, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_GENERAL, // Channel type
		s_pChannel_23_buffer, // Channel buffer pointer
		CHANNEL_23_SIZE, // Channel size
		s_pChannel_23_buffer, // Channel data start
		s_pChannel_23_buffer, // Channel data end
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
			"p17", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_VLC_4_p17, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			40, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			3, // Task ID
			"p63", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Init_3_p63, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			40, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
		{
			g_astChunk_channel_23_in, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Input chunk information
		{
			g_astChunk_channel_23_out, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Output chunk information
		CHUNK_NUM_NOT_INITIALIZED, // Written output chunk number
		g_astAvailableInputChunk_channel_23, // Available chunk list
		1, // maximum input port chunk size for all port sample rate cases
		NULL, // Chunk list head
		NULL, // Chunk list tail
		0, // Initial data length 
	},
	{
		24, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_GENERAL, // Channel type
		s_pChannel_24_buffer, // Channel buffer pointer
		CHANNEL_24_SIZE, // Channel size
		s_pChannel_24_buffer, // Channel data start
		s_pChannel_24_buffer, // Channel data end
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
			"p18", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_VLC_4_p18, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			3, // Task ID
			"p67", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Init_3_p67, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
		{
			g_astChunk_channel_24_in, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Input chunk information
		{
			g_astChunk_channel_24_out, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Output chunk information
		CHUNK_NUM_NOT_INITIALIZED, // Written output chunk number
		g_astAvailableInputChunk_channel_24, // Available chunk list
		1, // maximum input port chunk size for all port sample rate cases
		NULL, // Chunk list head
		NULL, // Chunk list tail
		0, // Initial data length 
	},
	{
		25, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_GENERAL, // Channel type
		s_pChannel_25_buffer, // Channel buffer pointer
		CHANNEL_25_SIZE, // Channel size
		s_pChannel_25_buffer, // Channel data start
		s_pChannel_25_buffer, // Channel data end
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
			"p0", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_VLC_4_p0, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			3, // Task ID
			"p68", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Init_3_p68, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
		{
			g_astChunk_channel_25_in, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Input chunk information
		{
			g_astChunk_channel_25_out, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Output chunk information
		CHUNK_NUM_NOT_INITIALIZED, // Written output chunk number
		g_astAvailableInputChunk_channel_25, // Available chunk list
		1, // maximum input port chunk size for all port sample rate cases
		NULL, // Chunk list head
		NULL, // Chunk list tail
		0, // Initial data length 
	},
	{
		26, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_GENERAL, // Channel type
		s_pChannel_26_buffer, // Channel buffer pointer
		CHANNEL_26_SIZE, // Channel size
		s_pChannel_26_buffer, // Channel data start
		s_pChannel_26_buffer, // Channel data end
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
			"p14", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_VLC_4_p14, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			32, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			1, // Task ID
			"p43", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Encoder_1_p43, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			32, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
		{
			g_astChunk_channel_26_in, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Input chunk information
		{
			g_astChunk_channel_26_out, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Output chunk information
		CHUNK_NUM_NOT_INITIALIZED, // Written output chunk number
		g_astAvailableInputChunk_channel_26, // Available chunk list
		1, // maximum input port chunk size for all port sample rate cases
		NULL, // Chunk list head
		NULL, // Chunk list tail
		0, // Initial data length 
	},
	{
		27, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_GENERAL, // Channel type
		s_pChannel_27_buffer, // Channel buffer pointer
		CHANNEL_27_SIZE, // Channel size
		s_pChannel_27_buffer, // Channel data start
		s_pChannel_27_buffer, // Channel data end
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
			"p16", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_VLC_4_p16, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			480, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			1, // Task ID
			"p42", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Encoder_1_p42, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			480, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
		{
			g_astChunk_channel_27_in, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Input chunk information
		{
			g_astChunk_channel_27_out, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Output chunk information
		CHUNK_NUM_NOT_INITIALIZED, // Written output chunk number
		g_astAvailableInputChunk_channel_27, // Available chunk list
		1, // maximum input port chunk size for all port sample rate cases
		NULL, // Chunk list head
		NULL, // Chunk list tail
		0, // Initial data length 
	},
	{
		28, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_GENERAL, // Channel type
		s_pChannel_28_buffer, // Channel buffer pointer
		CHANNEL_28_SIZE, // Channel size
		s_pChannel_28_buffer, // Channel data start
		s_pChannel_28_buffer, // Channel data end
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
			"p12", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_VLC_4_p12, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			64, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			1, // Task ID
			"p41", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Encoder_1_p41, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			64, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
		{
			g_astChunk_channel_28_in, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Input chunk information
		{
			g_astChunk_channel_28_out, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Output chunk information
		CHUNK_NUM_NOT_INITIALIZED, // Written output chunk number
		g_astAvailableInputChunk_channel_28, // Available chunk list
		1, // maximum input port chunk size for all port sample rate cases
		NULL, // Chunk list head
		NULL, // Chunk list tail
		0, // Initial data length 
	},
	{
		29, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_GENERAL, // Channel type
		s_pChannel_29_buffer, // Channel buffer pointer
		CHANNEL_29_SIZE, // Channel size
		s_pChannel_29_buffer, // Channel data start
		s_pChannel_29_buffer, // Channel data end
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
			"p7", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_VLC_4_p7, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			1, // Task ID
			"p35", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Encoder_1_p35, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
		{
			g_astChunk_channel_29_in, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Input chunk information
		{
			g_astChunk_channel_29_out, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Output chunk information
		CHUNK_NUM_NOT_INITIALIZED, // Written output chunk number
		g_astAvailableInputChunk_channel_29, // Available chunk list
		1, // maximum input port chunk size for all port sample rate cases
		NULL, // Chunk list head
		NULL, // Chunk list tail
		0, // Initial data length 
	},
	{
		30, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_GENERAL, // Channel type
		s_pChannel_30_buffer, // Channel buffer pointer
		CHANNEL_30_SIZE, // Channel size
		s_pChannel_30_buffer, // Channel data start
		s_pChannel_30_buffer, // Channel data end
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
			"p8", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_VLC_4_p8, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			1, // Task ID
			"p36", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Encoder_1_p36, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
		{
			g_astChunk_channel_30_in, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Input chunk information
		{
			g_astChunk_channel_30_out, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Output chunk information
		CHUNK_NUM_NOT_INITIALIZED, // Written output chunk number
		g_astAvailableInputChunk_channel_30, // Available chunk list
		1, // maximum input port chunk size for all port sample rate cases
		NULL, // Chunk list head
		NULL, // Chunk list tail
		0, // Initial data length 
	},
	{
		31, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_GENERAL, // Channel type
		s_pChannel_31_buffer, // Channel buffer pointer
		CHANNEL_31_SIZE, // Channel size
		s_pChannel_31_buffer, // Channel data start
		s_pChannel_31_buffer, // Channel data end
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
			"p13", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_VLC_4_p13, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			1024, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			1, // Task ID
			"p37", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Encoder_1_p37, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			1024, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
		{
			g_astChunk_channel_31_in, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Input chunk information
		{
			g_astChunk_channel_31_out, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Output chunk information
		CHUNK_NUM_NOT_INITIALIZED, // Written output chunk number
		g_astAvailableInputChunk_channel_31, // Available chunk list
		1, // maximum input port chunk size for all port sample rate cases
		NULL, // Chunk list head
		NULL, // Chunk list tail
		0, // Initial data length 
	},
	{
		32, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_GENERAL, // Channel type
		s_pChannel_32_buffer, // Channel buffer pointer
		CHANNEL_32_SIZE, // Channel size
		s_pChannel_32_buffer, // Channel data start
		s_pChannel_32_buffer, // Channel data end
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
			"p15", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_VLC_4_p15, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			960, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			1, // Task ID
			"p38", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Encoder_1_p38, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			960, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
		{
			g_astChunk_channel_32_in, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Input chunk information
		{
			g_astChunk_channel_32_out, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Output chunk information
		CHUNK_NUM_NOT_INITIALIZED, // Written output chunk number
		g_astAvailableInputChunk_channel_32, // Available chunk list
		1, // maximum input port chunk size for all port sample rate cases
		NULL, // Chunk list head
		NULL, // Chunk list tail
		0, // Initial data length 
	},
	{
		33, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_GENERAL, // Channel type
		s_pChannel_33_buffer, // Channel buffer pointer
		CHANNEL_33_SIZE, // Channel size
		s_pChannel_33_buffer, // Channel data start
		s_pChannel_33_buffer, // Channel data end
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
			"p6", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_VLC_4_p6, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			192, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			1, // Task ID
			"p39", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_Encoder_1_p39, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			192, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
		{
			g_astChunk_channel_33_in, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Input chunk information
		{
			g_astChunk_channel_33_out, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Output chunk information
		CHUNK_NUM_NOT_INITIALIZED, // Written output chunk number
		g_astAvailableInputChunk_channel_33, // Available chunk list
		1, // maximum input port chunk size for all port sample rate cases
		NULL, // Chunk list head
		NULL, // Chunk list tail
		0, // Initial data length 
	},
	{
		34, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_OUTPUT_ARRAY, // Channel type
		s_pChannel_34_buffer, // Channel buffer pointer
		CHANNEL_34_SIZE, // Channel size
		s_pChannel_34_buffer, // Channel data start
		s_pChannel_34_buffer, // Channel data end
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
			"p60", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_VLC_4_p60, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			12372, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			0, // Task ID
			"p106", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_ME_0_p106, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			12372, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
		{
			g_astChunk_channel_34_in, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Input chunk information
		{
			g_astChunk_channel_34_out, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Output chunk information
		CHUNK_NUM_NOT_INITIALIZED, // Written output chunk number
		g_astAvailableInputChunk_channel_34, // Available chunk list
		1, // maximum input port chunk size for all port sample rate cases
		NULL, // Chunk list head
		NULL, // Chunk list tail
		0, // Initial data length 
	},
};
// ##CHANNEL_LIST_TEMPLATE::END


// ##TASK_LIST_TEMPLATE::START
STask g_astTasks_top[] = {
	{ 	0, // Task ID
		"ME_0", // Task name
		TASK_TYPE_LOOP, // Task Type
		g_ast_ME_0_functions, // Task function array
		4, // Task function array number
		RUN_CONDITION_TIME_DRIVEN, // Run condition
		1, // Run rate
		10, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // Subgraph
		&g_stGraph_top, // Parent task graph
		NULL, // MTM information
		&g_stLoopStruct_ME_0, // Loop information
		NULL, // Task parameter information
		0, // Task parameter number
		FALSE, // Statically scheduled or not
		0,	  // Throughput constraint
		NULL, // Mutex
		NULL, // Conditional variable
	},
	{ 	1, // Task ID
		"Encoder_1", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_Encoder_1_functions, // Task function array
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
		0, // Task parameter number
		FALSE, // Statically scheduled or not
		0,	  // Throughput constraint
		NULL, // Mutex
		NULL, // Conditional variable
	},
	{ 	2, // Task ID
		"Deblock_2", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_Deblock_2_functions, // Task function array
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
		0, // Task parameter number
		FALSE, // Statically scheduled or not
		0,	  // Throughput constraint
		NULL, // Mutex
		NULL, // Conditional variable
	},
	{ 	3, // Task ID
		"Init_3", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_Init_3_functions, // Task function array
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
		0, // Task parameter number
		FALSE, // Statically scheduled or not
		0,	  // Throughput constraint
		NULL, // Mutex
		NULL, // Conditional variable
	},
	{ 	4, // Task ID
		"VLC_4", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_VLC_4_functions, // Task function array
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
		5, // number of tasks
		NULL, // parent task
};

// ##TASK_GRAPH_TEMPLATE::END

// ##TASK_ID_TO_TASK_MAP_TEMPLATE::START
STaskIdToTaskMap g_astTaskIdToTask[] = {
	{ 	2, // Task ID
		"Deblock_2", // Task name
		&g_astTasks_top[2], // Task structure pointer
	},
	{ 	3, // Task ID
		"Init_3", // Task name
		&g_astTasks_top[3], // Task structure pointer
	},
	{ 	1, // Task ID
		"Encoder_1", // Task name
		&g_astTasks_top[1], // Task structure pointer
	},
	{ 	0, // Task ID
		"ME_0", // Task name
		&g_astTasks_top[0], // Task structure pointer
	},
	{ 	4, // Task ID
		"VLC_4", // Task name
		&g_astTasks_top[4], // Task structure pointer
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
		&g_astTasks_top[2], // Task ID or composite task information
		0, // Processor ID
		0, // Processor local ID
	},
	{	TASK_TYPE_COMPUTATIONAL, // Task type
		&g_astTasks_top[3], // Task ID or composite task information
		0, // Processor ID
		0, // Processor local ID
	},
	{	TASK_TYPE_COMPUTATIONAL, // Task type
		&g_astTasks_top[1], // Task ID or composite task information
		0, // Processor ID
		0, // Processor local ID
	},
	{	TASK_TYPE_LOOP, // Task type
		&g_astTasks_top[0], // Task ID or composite task information
		0, // Processor ID
		1, // Processor local ID
	},
	{	TASK_TYPE_LOOP, // Task type
		&g_astTasks_top[0], // Task ID or composite task information
		0, // Processor ID
		2, // Processor local ID
	},
	{	TASK_TYPE_LOOP, // Task type
		&g_astTasks_top[0], // Task ID or composite task information
		0, // Processor ID
		3, // Processor local ID
	},
	{	TASK_TYPE_LOOP, // Task type
		&g_astTasks_top[0], // Task ID or composite task information
		0, // Processor ID
		4, // Processor local ID
	},
	{	TASK_TYPE_COMPUTATIONAL, // Task type
		&g_astTasks_top[4], // Task ID or composite task information
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

