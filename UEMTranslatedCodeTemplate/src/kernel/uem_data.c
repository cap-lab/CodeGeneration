/* uem_data.c made by UEM Translator */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_data.h>
#include <UKTask.h>
#include <UKModeTransition.h>

SExecutionTime g_stExecutionTime = { 30, TIME_METRIC_SEC } ;

// ##TASK_CODE_TEMPLATE::START
void x264Enc_Init_3_Init0(int nTaskId);
void x264Enc_Init_3_Go0(int nTaskId);
void x264Enc_Init_3_Wrapup0();

void H264Dec_VIDEO_Decode_Init0(int nTaskId);
void H264Dec_VIDEO_Decode_Go0(int nTaskId);
void H264Dec_VIDEO_Decode_Wrapup0();

void G723Enc_Init0(int nTaskId);
void G723Enc_Go0(int nTaskId);
void G723Enc_Wrapup0();

void x264Enc_Deblock_2_Init0(int nTaskId);
void x264Enc_Deblock_2_Go0(int nTaskId);
void x264Enc_Deblock_2_Wrapup0();

void H264Dec_PHONE_Decode_Init0(int nTaskId);
void H264Dec_PHONE_Decode_Go0(int nTaskId);
void H264Dec_PHONE_Decode_Wrapup0();


void MP3Dec_VLDStream_Init0(int nTaskId);
void MP3Dec_VLDStream_Go0(int nTaskId);
void MP3Dec_VLDStream_Wrapup0();

void x264Enc_Encoder_1_Init0(int nTaskId);
void x264Enc_Encoder_1_Go0(int nTaskId);
void x264Enc_Encoder_1_Wrapup0();

void Control_Init0(int nTaskId);
void Control_Go0(int nTaskId);
void Control_Wrapup0();

void H264Dec_VIDEO_IntraPredY_Init0(int nTaskId);
void H264Dec_VIDEO_IntraPredY_Go0(int nTaskId);
void H264Dec_VIDEO_IntraPredY_Wrapup0();


void H264Dec_VIDEO_IntraPredV_Init0(int nTaskId);
void H264Dec_VIDEO_IntraPredV_Go0(int nTaskId);
void H264Dec_VIDEO_IntraPredV_Wrapup0();

void H264Dec_VIDEO_WriteFileH_Init0(int nTaskId);
void H264Dec_VIDEO_WriteFileH_Go0(int nTaskId);
void H264Dec_VIDEO_WriteFileH_Wrapup0();

void MP3Dec_Antialias_Init0(int nTaskId);
void MP3Dec_Antialias_Go0(int nTaskId);
void MP3Dec_Antialias_Wrapup0();

void MP3Dec_Subband_Init0(int nTaskId);
void MP3Dec_Subband_Go0(int nTaskId);
void MP3Dec_Subband_Wrapup0();

void H264Dec_PHONE_InterPredY_Init0(int nTaskId);
void H264Dec_PHONE_InterPredY_Go0(int nTaskId);
void H264Dec_PHONE_InterPredY_Wrapup0();

void H264Dec_VIDEO_Deblock_Init0(int nTaskId);
void H264Dec_VIDEO_Deblock_Go0(int nTaskId);
void H264Dec_VIDEO_Deblock_Wrapup0();

void x264Enc_ME_0_Init0(int nTaskId);
void x264Enc_ME_0_Go0(int nTaskId);
void x264Enc_ME_0_Wrapup0();

void H264Dec_VIDEO_InterPredU_Init0(int nTaskId);
void H264Dec_VIDEO_InterPredU_Go0(int nTaskId);
void H264Dec_VIDEO_InterPredU_Wrapup0();

void H264Dec_VIDEO_IntraPredU_Init0(int nTaskId);
void H264Dec_VIDEO_IntraPredU_Go0(int nTaskId);
void H264Dec_VIDEO_IntraPredU_Wrapup0();

void UserInput_Init0(int nTaskId);
void UserInput_Go0(int nTaskId);
void UserInput_Wrapup0();


void H264Dec_PHONE_Deblock_Init0(int nTaskId);
void H264Dec_PHONE_Deblock_Go0(int nTaskId);
void H264Dec_PHONE_Deblock_Wrapup0();

void H264Dec_VIDEO_InterPredV_Init0(int nTaskId);
void H264Dec_VIDEO_InterPredV_Go0(int nTaskId);
void H264Dec_VIDEO_InterPredV_Wrapup0();

void H264Dec_PHONE_ReadFileH_Init0(int nTaskId);
void H264Dec_PHONE_ReadFileH_Go0(int nTaskId);
void H264Dec_PHONE_ReadFileH_Wrapup0();

void H264Dec_VIDEO_InterPredY_Init0(int nTaskId);
void H264Dec_VIDEO_InterPredY_Go0(int nTaskId);
void H264Dec_VIDEO_InterPredY_Wrapup0();

void MP3Dec_WriteFileM_Init0(int nTaskId);
void MP3Dec_WriteFileM_Go0(int nTaskId);
void MP3Dec_WriteFileM_Wrapup0();


void H264Dec_PHONE_IntraPredV_Init0(int nTaskId);
void H264Dec_PHONE_IntraPredV_Go0(int nTaskId);
void H264Dec_PHONE_IntraPredV_Wrapup0();

void Interrupt_Init0(int nTaskId);
void Interrupt_Go0(int nTaskId);
void Interrupt_Wrapup0();

void H264Dec_PHONE_IntraPredU_Init0(int nTaskId);
void H264Dec_PHONE_IntraPredU_Go0(int nTaskId);
void H264Dec_PHONE_IntraPredU_Wrapup0();

void H264Dec_VIDEO_ReadFileH_Init0(int nTaskId);
void H264Dec_VIDEO_ReadFileH_Go0(int nTaskId);
void H264Dec_VIDEO_ReadFileH_Wrapup0();

void H264Dec_PHONE_IntraPredY_Init0(int nTaskId);
void H264Dec_PHONE_IntraPredY_Go0(int nTaskId);
void H264Dec_PHONE_IntraPredY_Wrapup0();

void MP3Dec_Stereo_Init0(int nTaskId);
void MP3Dec_Stereo_Go0(int nTaskId);
void MP3Dec_Stereo_Wrapup0();

void MP3Dec_DeQ_Init0(int nTaskId);
void MP3Dec_DeQ_Go0(int nTaskId);
void MP3Dec_DeQ_Wrapup0();

void H264Dec_PHONE_WriteFileH_Init0(int nTaskId);
void H264Dec_PHONE_WriteFileH_Go0(int nTaskId);
void H264Dec_PHONE_WriteFileH_Wrapup0();

void H264Dec_PHONE_InterPredU_Init0(int nTaskId);
void H264Dec_PHONE_InterPredU_Go0(int nTaskId);
void H264Dec_PHONE_InterPredU_Wrapup0();

void H264Dec_PHONE_InterPredV_Init0(int nTaskId);
void H264Dec_PHONE_InterPredV_Go0(int nTaskId);
void H264Dec_PHONE_InterPredV_Wrapup0();

void MP3Dec_Reorder_Init0(int nTaskId);
void MP3Dec_Reorder_Go0(int nTaskId);
void MP3Dec_Reorder_Wrapup0();

void x264Enc_VLC_4_Init0(int nTaskId);
void x264Enc_VLC_4_Go0(int nTaskId);
void x264Enc_VLC_4_Wrapup0();

void G723Dec_Init0(int nTaskId);
void G723Dec_Go0(int nTaskId);
void G723Dec_Wrapup0();

void MP3Dec_Hybrid_Init0(int nTaskId);
void MP3Dec_Hybrid_Go0(int nTaskId);
void MP3Dec_Hybrid_Wrapup0();

// ##TASK_CODE_TEMPLATE::END


// ##LIBRARY_INIT_WRAPUP_TEMPLATE::START
// ##LIBRARY_INIT_WRAPUP_TEMPLATE::END


// ##CHANNEL_SIZE_DEFINITION_TEMPLATE::START
#define CHANNEL_0_SIZE (999999)
#define CHANNEL_1_SIZE (110000000)
#define CHANNEL_2_SIZE (25000000)
#define CHANNEL_3_SIZE (25000000)
#define CHANNEL_4_SIZE (999999)
#define CHANNEL_5_SIZE (999999)
#define CHANNEL_6_SIZE (999999)
#define CHANNEL_7_SIZE (999999)
#define CHANNEL_8_SIZE (999999)
#define CHANNEL_9_SIZE (999999)
#define CHANNEL_10_SIZE (999999)
#define CHANNEL_11_SIZE (110000000)
#define CHANNEL_12_SIZE (25000000)
#define CHANNEL_13_SIZE (25000000)
#define CHANNEL_14_SIZE (999999)
#define CHANNEL_15_SIZE (999999)
#define CHANNEL_16_SIZE (999999)
#define CHANNEL_17_SIZE (9999999)
#define CHANNEL_18_SIZE (9999999)
#define CHANNEL_19_SIZE (13440680)
#define CHANNEL_20_SIZE (9999999)
#define CHANNEL_21_SIZE (11324000)
#define CHANNEL_22_SIZE (44832)
#define CHANNEL_23_SIZE (26880)
#define CHANNEL_24_SIZE (53760)
#define CHANNEL_25_SIZE (21128)
#define CHANNEL_26_SIZE (26880)
#define CHANNEL_27_SIZE (53760)
#define CHANNEL_28_SIZE (53760)
#define CHANNEL_29_SIZE (999999)
#define CHANNEL_30_SIZE (110000000)
#define CHANNEL_31_SIZE (25000000)
#define CHANNEL_32_SIZE (25000000)
#define CHANNEL_33_SIZE (999999)
#define CHANNEL_34_SIZE (999999)
#define CHANNEL_35_SIZE (999999)
#define CHANNEL_36_SIZE (999999)
#define CHANNEL_37_SIZE (999999)
#define CHANNEL_38_SIZE (999999)
#define CHANNEL_39_SIZE (999999)
#define CHANNEL_40_SIZE (110000000)
#define CHANNEL_41_SIZE (25000000)
#define CHANNEL_42_SIZE (25000000)
#define CHANNEL_43_SIZE (999999)
#define CHANNEL_44_SIZE (999999)
#define CHANNEL_45_SIZE (999999)
#define CHANNEL_46_SIZE (4)
#define CHANNEL_47_SIZE (4)
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
char s_pChannel_35_buffer[CHANNEL_35_SIZE];
char s_pChannel_36_buffer[CHANNEL_36_SIZE];
char s_pChannel_37_buffer[CHANNEL_37_SIZE];
char s_pChannel_38_buffer[CHANNEL_38_SIZE];
char s_pChannel_39_buffer[CHANNEL_39_SIZE];
char s_pChannel_40_buffer[CHANNEL_40_SIZE];
char s_pChannel_41_buffer[CHANNEL_41_SIZE];
char s_pChannel_42_buffer[CHANNEL_42_SIZE];
char s_pChannel_43_buffer[CHANNEL_43_SIZE];
char s_pChannel_44_buffer[CHANNEL_44_SIZE];
char s_pChannel_45_buffer[CHANNEL_45_SIZE];
char s_pChannel_46_buffer[CHANNEL_46_SIZE];
char s_pChannel_47_buffer[CHANNEL_47_SIZE];
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

SChunk g_astChunk_channel_35_out[] = {
	{
		s_pChannel_35_buffer, // Chunk start pointer
		s_pChannel_35_buffer, // Data start pointer
		s_pChannel_35_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_35_in[] = {
	{
		s_pChannel_35_buffer, // Chunk start pointer
		s_pChannel_35_buffer, // Data start pointer
		s_pChannel_35_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_36_out[] = {
	{
		s_pChannel_36_buffer, // Chunk start pointer
		s_pChannel_36_buffer, // Data start pointer
		s_pChannel_36_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_36_in[] = {
	{
		s_pChannel_36_buffer, // Chunk start pointer
		s_pChannel_36_buffer, // Data start pointer
		s_pChannel_36_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_37_out[] = {
	{
		s_pChannel_37_buffer, // Chunk start pointer
		s_pChannel_37_buffer, // Data start pointer
		s_pChannel_37_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_37_in[] = {
	{
		s_pChannel_37_buffer, // Chunk start pointer
		s_pChannel_37_buffer, // Data start pointer
		s_pChannel_37_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_38_out[] = {
	{
		s_pChannel_38_buffer, // Chunk start pointer
		s_pChannel_38_buffer, // Data start pointer
		s_pChannel_38_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_38_in[] = {
	{
		s_pChannel_38_buffer, // Chunk start pointer
		s_pChannel_38_buffer, // Data start pointer
		s_pChannel_38_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_39_out[] = {
	{
		s_pChannel_39_buffer, // Chunk start pointer
		s_pChannel_39_buffer, // Data start pointer
		s_pChannel_39_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_39_in[] = {
	{
		s_pChannel_39_buffer, // Chunk start pointer
		s_pChannel_39_buffer, // Data start pointer
		s_pChannel_39_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_40_out[] = {
	{
		s_pChannel_40_buffer, // Chunk start pointer
		s_pChannel_40_buffer, // Data start pointer
		s_pChannel_40_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_40_in[] = {
	{
		s_pChannel_40_buffer, // Chunk start pointer
		s_pChannel_40_buffer, // Data start pointer
		s_pChannel_40_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_41_out[] = {
	{
		s_pChannel_41_buffer, // Chunk start pointer
		s_pChannel_41_buffer, // Data start pointer
		s_pChannel_41_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_41_in[] = {
	{
		s_pChannel_41_buffer, // Chunk start pointer
		s_pChannel_41_buffer, // Data start pointer
		s_pChannel_41_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_42_out[] = {
	{
		s_pChannel_42_buffer, // Chunk start pointer
		s_pChannel_42_buffer, // Data start pointer
		s_pChannel_42_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_42_in[] = {
	{
		s_pChannel_42_buffer, // Chunk start pointer
		s_pChannel_42_buffer, // Data start pointer
		s_pChannel_42_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_43_out[] = {
	{
		s_pChannel_43_buffer, // Chunk start pointer
		s_pChannel_43_buffer, // Data start pointer
		s_pChannel_43_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_43_in[] = {
	{
		s_pChannel_43_buffer, // Chunk start pointer
		s_pChannel_43_buffer, // Data start pointer
		s_pChannel_43_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_44_out[] = {
	{
		s_pChannel_44_buffer, // Chunk start pointer
		s_pChannel_44_buffer, // Data start pointer
		s_pChannel_44_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_44_in[] = {
	{
		s_pChannel_44_buffer, // Chunk start pointer
		s_pChannel_44_buffer, // Data start pointer
		s_pChannel_44_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_45_out[] = {
	{
		s_pChannel_45_buffer, // Chunk start pointer
		s_pChannel_45_buffer, // Data start pointer
		s_pChannel_45_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_45_in[] = {
	{
		s_pChannel_45_buffer, // Chunk start pointer
		s_pChannel_45_buffer, // Data start pointer
		s_pChannel_45_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_46_out[] = {
	{
		s_pChannel_46_buffer, // Chunk start pointer
		s_pChannel_46_buffer, // Data start pointer
		s_pChannel_46_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_46_in[] = {
	{
		s_pChannel_46_buffer, // Chunk start pointer
		s_pChannel_46_buffer, // Data start pointer
		s_pChannel_46_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_47_out[] = {
	{
		s_pChannel_47_buffer, // Chunk start pointer
		s_pChannel_47_buffer, // Data start pointer
		s_pChannel_47_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

SChunk g_astChunk_channel_47_in[] = {
	{
		s_pChannel_47_buffer, // Chunk start pointer
		s_pChannel_47_buffer, // Data start pointer
		s_pChannel_47_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number;
	},
};

// ##CHUNK_DEFINITION_TEMPLATE::END
//portSampleRateList


// ##PORT_SAMPLE_RATE_TEMPLATE::START
SPortSampleRate g_astPortSampleRate_H264Dec_VIDEO_Decode_inFrame[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_VIDEO_ReadFileH_outFrame[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_VIDEO_InterPredY_inMB_Y[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_VIDEO_Decode_outMB_interY[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_VIDEO_InterPredU_inMB_U[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_VIDEO_Decode_outMB_interU[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_VIDEO_InterPredV_inMB_V[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_VIDEO_Decode_outMB_interV[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_VIDEO_Deblock_inFrame_interY[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_VIDEO_InterPredY_outFrame_Y[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_VIDEO_Deblock_inFrame_interU[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_VIDEO_InterPredU_outFrame_U[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_VIDEO_Deblock_inFrame_interV[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_VIDEO_InterPredV_outFrame_V[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_VIDEO_WriteFileH_inFrame[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_VIDEO_Deblock_outFrame[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_VIDEO_InterPredY_inPrevFrame[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_VIDEO_Deblock_outRef_Y[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_VIDEO_InterPredU_inPrevFrame[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_VIDEO_Deblock_outRef_U[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_VIDEO_InterPredV_inPrevFrame[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_VIDEO_Deblock_outRef_V[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_VIDEO_IntraPredY_inMB_Y[] = {
	{ 	"I_Frame", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
	{ 	"P_Frame", // Mode name
		0, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_VIDEO_Decode_outMB_intraY[] = {
	{ 	"I_Frame", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
	{ 	"P_Frame", // Mode name
		0, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_VIDEO_IntraPredU_inMB_U[] = {
	{ 	"I_Frame", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
	{ 	"P_Frame", // Mode name
		0, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_VIDEO_Decode_outMB_intraU[] = {
	{ 	"I_Frame", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
	{ 	"P_Frame", // Mode name
		0, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_VIDEO_IntraPredV_inMB_V[] = {
	{ 	"I_Frame", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
	{ 	"P_Frame", // Mode name
		0, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_VIDEO_Decode_outMB_intraV[] = {
	{ 	"I_Frame", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
	{ 	"P_Frame", // Mode name
		0, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_VIDEO_Deblock_inFrame_intraY[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_VIDEO_IntraPredY_outFrame_Y[] = {
	{ 	"I_Frame", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
	{ 	"P_Frame", // Mode name
		0, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_VIDEO_Deblock_inFrame_intraU[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_VIDEO_IntraPredU_outFrame_U[] = {
	{ 	"I_Frame", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
	{ 	"P_Frame", // Mode name
		0, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_VIDEO_Deblock_inFrame_intraV[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_VIDEO_IntraPredV_outFrame_V[] = {
	{ 	"I_Frame", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
	{ 	"P_Frame", // Mode name
		0, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_x264Enc_ME_0_p101[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_x264Enc_Init_3_p71[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_x264Enc_Encoder_1_p52[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_x264Enc_ME_0_p105[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_x264Enc_VLC_4_p14[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_x264Enc_Encoder_1_p43[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_x264Enc_Deblock_2_p24[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_x264Enc_Encoder_1_p28[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_x264Enc_ME_0_p102[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_x264Enc_Deblock_2_p42[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_MP3Dec_DeQ_input[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_MP3Dec_VLDStream_output[] = {
	{ 	"Default", // Mode name
		2, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_MP3Dec_Hybrid_input[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_MP3Dec_Antialias_output[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_MP3Dec_Subband_input[] = {
	{ 	"Default", // Mode name
		2, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_MP3Dec_Hybrid_output[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_MP3Dec_WriteFileM_input[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_MP3Dec_Subband_output[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_MP3Dec_Reorder_input[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_MP3Dec_DeQ_output[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_MP3Dec_Stereo_input[] = {
	{ 	"Default", // Mode name
		2, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_MP3Dec_Reorder_output[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_MP3Dec_Antialias_input[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_MP3Dec_Stereo_output[] = {
	{ 	"Default", // Mode name
		2, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_PHONE_Decode_inFrame[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_PHONE_ReadFileH_outFrame[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_PHONE_InterPredY_inMB_Y[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_PHONE_Decode_outMB_interY[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_PHONE_InterPredU_inMB_U[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_PHONE_Decode_outMB_interU[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_PHONE_InterPredV_inMB_V[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_PHONE_Decode_outMB_interV[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_PHONE_Deblock_inFrame_interY[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_PHONE_InterPredY_outFrame_Y[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_PHONE_Deblock_inFrame_interU[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_PHONE_InterPredU_outFrame_U[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_PHONE_Deblock_inFrame_interV[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_PHONE_InterPredV_outFrame_V[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_PHONE_WriteFileH_inFrame[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_PHONE_Deblock_outFrame[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_PHONE_InterPredY_inPrevFrame[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_PHONE_Deblock_outRef_Y[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_PHONE_InterPredU_inPrevFrame[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_PHONE_Deblock_outRef_U[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_PHONE_InterPredV_inPrevFrame[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_PHONE_Deblock_outRef_V[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_PHONE_IntraPredY_inMB_Y[] = {
	{ 	"I_Frame", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
	{ 	"P_Frame", // Mode name
		0, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_PHONE_Decode_outMB_intraY[] = {
	{ 	"I_Frame", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
	{ 	"P_Frame", // Mode name
		0, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_PHONE_IntraPredU_inMB_U[] = {
	{ 	"I_Frame", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
	{ 	"P_Frame", // Mode name
		0, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_PHONE_Decode_outMB_intraU[] = {
	{ 	"I_Frame", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
	{ 	"P_Frame", // Mode name
		0, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_PHONE_IntraPredV_inMB_V[] = {
	{ 	"I_Frame", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
	{ 	"P_Frame", // Mode name
		0, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_PHONE_Decode_outMB_intraV[] = {
	{ 	"I_Frame", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
	{ 	"P_Frame", // Mode name
		0, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_PHONE_Deblock_inFrame_intraY[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_PHONE_IntraPredY_outFrame_Y[] = {
	{ 	"I_Frame", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
	{ 	"P_Frame", // Mode name
		0, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_PHONE_Deblock_inFrame_intraU[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_PHONE_IntraPredU_outFrame_U[] = {
	{ 	"I_Frame", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
	{ 	"P_Frame", // Mode name
		0, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_PHONE_Deblock_inFrame_intraV[] = {
	{ 	"Default", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_H264Dec_PHONE_IntraPredV_outFrame_V[] = {
	{ 	"I_Frame", // Mode name
		1, // Sample rate
		1, // Available number of data
	},
	{ 	"P_Frame", // Mode name
		0, // Sample rate
		1, // Available number of data
	},
};

SPortSampleRate g_astPortSampleRate_Control_in_UserInput[] = {
};

SPortSampleRate g_astPortSampleRate_UserInput_out[] = {
};

SPortSampleRate g_astPortSampleRate_Control_in_Interrupt[] = {
};

SPortSampleRate g_astPortSampleRate_Interrupt_out[] = {
};

// ##PORT_SAMPLE_RATE_TEMPLATE::END


// ##PORT_ARRAY_TEMPLATE::START
SPort g_astPortInfo[] = {
	{
		2, // Task ID
		"inFrame", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_H264Dec_VIDEO_Decode_inFrame, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		1, // Task ID
		"outFrame", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_H264Dec_VIDEO_ReadFileH_outFrame, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		3, // Task ID
		"inMB_Y", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_H264Dec_VIDEO_InterPredY_inMB_Y, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		2, // Task ID
		"outMB_interY", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_H264Dec_VIDEO_Decode_outMB_interY, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		4, // Task ID
		"inMB_U", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_H264Dec_VIDEO_InterPredU_inMB_U, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		2, // Task ID
		"outMB_interU", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_H264Dec_VIDEO_Decode_outMB_interU, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		5, // Task ID
		"inMB_V", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_H264Dec_VIDEO_InterPredV_inMB_V, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		2, // Task ID
		"outMB_interV", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_H264Dec_VIDEO_Decode_outMB_interV, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		6, // Task ID
		"inFrame_interY", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_H264Dec_VIDEO_Deblock_inFrame_interY, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		3, // Task ID
		"outFrame_Y", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_H264Dec_VIDEO_InterPredY_outFrame_Y, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		6, // Task ID
		"inFrame_interU", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_H264Dec_VIDEO_Deblock_inFrame_interU, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		4, // Task ID
		"outFrame_U", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_H264Dec_VIDEO_InterPredU_outFrame_U, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		6, // Task ID
		"inFrame_interV", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_H264Dec_VIDEO_Deblock_inFrame_interV, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		5, // Task ID
		"outFrame_V", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_H264Dec_VIDEO_InterPredV_outFrame_V, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		7, // Task ID
		"inFrame", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_H264Dec_VIDEO_WriteFileH_inFrame, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		6, // Task ID
		"outFrame", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_H264Dec_VIDEO_Deblock_outFrame, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		3, // Task ID
		"inPrevFrame", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_H264Dec_VIDEO_InterPredY_inPrevFrame, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		6, // Task ID
		"outRef_Y", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_H264Dec_VIDEO_Deblock_outRef_Y, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		4, // Task ID
		"inPrevFrame", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_H264Dec_VIDEO_InterPredU_inPrevFrame, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		6, // Task ID
		"outRef_U", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_H264Dec_VIDEO_Deblock_outRef_U, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		5, // Task ID
		"inPrevFrame", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_H264Dec_VIDEO_InterPredV_inPrevFrame, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		6, // Task ID
		"outRef_V", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_H264Dec_VIDEO_Deblock_outRef_V, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		8, // Task ID
		"inMB_Y", // Port name
		PORT_SAMPLE_RATE_MULTIPLE, // Port sample rate type
		g_astPortSampleRate_H264Dec_VIDEO_IntraPredY_inMB_Y, // Array of sample rate list
		2, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		2, // Task ID
		"outMB_intraY", // Port name
		PORT_SAMPLE_RATE_MULTIPLE, // Port sample rate type
		g_astPortSampleRate_H264Dec_VIDEO_Decode_outMB_intraY, // Array of sample rate list
		2, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		9, // Task ID
		"inMB_U", // Port name
		PORT_SAMPLE_RATE_MULTIPLE, // Port sample rate type
		g_astPortSampleRate_H264Dec_VIDEO_IntraPredU_inMB_U, // Array of sample rate list
		2, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		2, // Task ID
		"outMB_intraU", // Port name
		PORT_SAMPLE_RATE_MULTIPLE, // Port sample rate type
		g_astPortSampleRate_H264Dec_VIDEO_Decode_outMB_intraU, // Array of sample rate list
		2, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		10, // Task ID
		"inMB_V", // Port name
		PORT_SAMPLE_RATE_MULTIPLE, // Port sample rate type
		g_astPortSampleRate_H264Dec_VIDEO_IntraPredV_inMB_V, // Array of sample rate list
		2, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		2, // Task ID
		"outMB_intraV", // Port name
		PORT_SAMPLE_RATE_MULTIPLE, // Port sample rate type
		g_astPortSampleRate_H264Dec_VIDEO_Decode_outMB_intraV, // Array of sample rate list
		2, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		6, // Task ID
		"inFrame_intraY", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_H264Dec_VIDEO_Deblock_inFrame_intraY, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		8, // Task ID
		"outFrame_Y", // Port name
		PORT_SAMPLE_RATE_MULTIPLE, // Port sample rate type
		g_astPortSampleRate_H264Dec_VIDEO_IntraPredY_outFrame_Y, // Array of sample rate list
		2, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		6, // Task ID
		"inFrame_intraU", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_H264Dec_VIDEO_Deblock_inFrame_intraU, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		9, // Task ID
		"outFrame_U", // Port name
		PORT_SAMPLE_RATE_MULTIPLE, // Port sample rate type
		g_astPortSampleRate_H264Dec_VIDEO_IntraPredU_outFrame_U, // Array of sample rate list
		2, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		6, // Task ID
		"inFrame_intraV", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_H264Dec_VIDEO_Deblock_inFrame_intraV, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		10, // Task ID
		"outFrame_V", // Port name
		PORT_SAMPLE_RATE_MULTIPLE, // Port sample rate type
		g_astPortSampleRate_H264Dec_VIDEO_IntraPredV_outFrame_V, // Array of sample rate list
		2, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		12, // Task ID
		"p101", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_x264Enc_ME_0_p101, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		11324, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		15, // Task ID
		"p71", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_x264Enc_Init_3_p71, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		13, // Task ID
		"p52", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_x264Enc_Encoder_1_p52, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		12372, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		12, // Task ID
		"p105", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_x264Enc_ME_0_p105, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		12372, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		16, // Task ID
		"p14", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_x264Enc_VLC_4_p14, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		32, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		13, // Task ID
		"p43", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_x264Enc_Encoder_1_p43, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		64, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		14, // Task ID
		"p24", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_x264Enc_Deblock_2_p24, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		64, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		13, // Task ID
		"p28", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_x264Enc_Encoder_1_p28, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		32, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		12, // Task ID
		"p102", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_x264Enc_ME_0_p102, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		396, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		14, // Task ID
		"p42", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_x264Enc_Deblock_2_p42, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		11324, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		19, // Task ID
		"input", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_MP3Dec_DeQ_input, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		18, // Task ID
		"output", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_MP3Dec_VLDStream_output, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		23, // Task ID
		"input", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_MP3Dec_Hybrid_input, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		22, // Task ID
		"output", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_MP3Dec_Antialias_output, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		24, // Task ID
		"input", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_MP3Dec_Subband_input, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		23, // Task ID
		"output", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_MP3Dec_Hybrid_output, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		25, // Task ID
		"input", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_MP3Dec_WriteFileM_input, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		24, // Task ID
		"output", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_MP3Dec_Subband_output, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		21, // Task ID
		"input", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_MP3Dec_Reorder_input, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		19, // Task ID
		"output", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_MP3Dec_DeQ_output, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		20, // Task ID
		"input", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_MP3Dec_Stereo_input, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		21, // Task ID
		"output", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_MP3Dec_Reorder_output, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		22, // Task ID
		"input", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_MP3Dec_Antialias_input, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		20, // Task ID
		"output", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_MP3Dec_Stereo_output, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		31, // Task ID
		"inFrame", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_H264Dec_PHONE_Decode_inFrame, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		30, // Task ID
		"outFrame", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_H264Dec_PHONE_ReadFileH_outFrame, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		32, // Task ID
		"inMB_Y", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_H264Dec_PHONE_InterPredY_inMB_Y, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		31, // Task ID
		"outMB_interY", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_H264Dec_PHONE_Decode_outMB_interY, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		33, // Task ID
		"inMB_U", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_H264Dec_PHONE_InterPredU_inMB_U, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		31, // Task ID
		"outMB_interU", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_H264Dec_PHONE_Decode_outMB_interU, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		34, // Task ID
		"inMB_V", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_H264Dec_PHONE_InterPredV_inMB_V, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		31, // Task ID
		"outMB_interV", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_H264Dec_PHONE_Decode_outMB_interV, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		35, // Task ID
		"inFrame_interY", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_H264Dec_PHONE_Deblock_inFrame_interY, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		32, // Task ID
		"outFrame_Y", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_H264Dec_PHONE_InterPredY_outFrame_Y, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		35, // Task ID
		"inFrame_interU", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_H264Dec_PHONE_Deblock_inFrame_interU, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		33, // Task ID
		"outFrame_U", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_H264Dec_PHONE_InterPredU_outFrame_U, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		35, // Task ID
		"inFrame_interV", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_H264Dec_PHONE_Deblock_inFrame_interV, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		34, // Task ID
		"outFrame_V", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_H264Dec_PHONE_InterPredV_outFrame_V, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		36, // Task ID
		"inFrame", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_H264Dec_PHONE_WriteFileH_inFrame, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		35, // Task ID
		"outFrame", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_H264Dec_PHONE_Deblock_outFrame, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		32, // Task ID
		"inPrevFrame", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_H264Dec_PHONE_InterPredY_inPrevFrame, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		35, // Task ID
		"outRef_Y", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_H264Dec_PHONE_Deblock_outRef_Y, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		33, // Task ID
		"inPrevFrame", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_H264Dec_PHONE_InterPredU_inPrevFrame, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		35, // Task ID
		"outRef_U", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_H264Dec_PHONE_Deblock_outRef_U, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		34, // Task ID
		"inPrevFrame", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_H264Dec_PHONE_InterPredV_inPrevFrame, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		35, // Task ID
		"outRef_V", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_H264Dec_PHONE_Deblock_outRef_V, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		37, // Task ID
		"inMB_Y", // Port name
		PORT_SAMPLE_RATE_MULTIPLE, // Port sample rate type
		g_astPortSampleRate_H264Dec_PHONE_IntraPredY_inMB_Y, // Array of sample rate list
		2, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		31, // Task ID
		"outMB_intraY", // Port name
		PORT_SAMPLE_RATE_MULTIPLE, // Port sample rate type
		g_astPortSampleRate_H264Dec_PHONE_Decode_outMB_intraY, // Array of sample rate list
		2, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		38, // Task ID
		"inMB_U", // Port name
		PORT_SAMPLE_RATE_MULTIPLE, // Port sample rate type
		g_astPortSampleRate_H264Dec_PHONE_IntraPredU_inMB_U, // Array of sample rate list
		2, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		31, // Task ID
		"outMB_intraU", // Port name
		PORT_SAMPLE_RATE_MULTIPLE, // Port sample rate type
		g_astPortSampleRate_H264Dec_PHONE_Decode_outMB_intraU, // Array of sample rate list
		2, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		39, // Task ID
		"inMB_V", // Port name
		PORT_SAMPLE_RATE_MULTIPLE, // Port sample rate type
		g_astPortSampleRate_H264Dec_PHONE_IntraPredV_inMB_V, // Array of sample rate list
		2, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		31, // Task ID
		"outMB_intraV", // Port name
		PORT_SAMPLE_RATE_MULTIPLE, // Port sample rate type
		g_astPortSampleRate_H264Dec_PHONE_Decode_outMB_intraV, // Array of sample rate list
		2, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		35, // Task ID
		"inFrame_intraY", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_H264Dec_PHONE_Deblock_inFrame_intraY, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		37, // Task ID
		"outFrame_Y", // Port name
		PORT_SAMPLE_RATE_MULTIPLE, // Port sample rate type
		g_astPortSampleRate_H264Dec_PHONE_IntraPredY_outFrame_Y, // Array of sample rate list
		2, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		35, // Task ID
		"inFrame_intraU", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_H264Dec_PHONE_Deblock_inFrame_intraU, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		38, // Task ID
		"outFrame_U", // Port name
		PORT_SAMPLE_RATE_MULTIPLE, // Port sample rate type
		g_astPortSampleRate_H264Dec_PHONE_IntraPredU_outFrame_U, // Array of sample rate list
		2, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		35, // Task ID
		"inFrame_intraV", // Port name
		PORT_SAMPLE_RATE_FIXED, // Port sample rate type
		g_astPortSampleRate_H264Dec_PHONE_Deblock_inFrame_intraV, // Array of sample rate list
		1, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		39, // Task ID
		"outFrame_V", // Port name
		PORT_SAMPLE_RATE_MULTIPLE, // Port sample rate type
		g_astPortSampleRate_H264Dec_PHONE_IntraPredV_outFrame_V, // Array of sample rate list
		2, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		41, // Task ID
		"in_UserInput", // Port name
		PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
		g_astPortSampleRate_Control_in_UserInput, // Array of sample rate list
		0, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		28, // Task ID
		"out", // Port name
		PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
		g_astPortSampleRate_UserInput_out, // Array of sample rate list
		0, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		41, // Task ID
		"in_Interrupt", // Port name
		PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
		g_astPortSampleRate_Control_in_Interrupt, // Array of sample rate list
		0, // Array element number of sample rate list
		0, //Selected sample rate index
		4, // Sample size
		PORT_TYPE_QUEUE, // Port type
		NULL, // Pointer to Subgraph port
	}, // Port information		
	{
		40, // Task ID
		"out", // Port name
		PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
		g_astPortSampleRate_Interrupt_out, // Array of sample rate list
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
STask g_astTasks_top[];
STask g_astTasks_MP3Dec[];
STask g_astTasks_H264Dec_VIDEO[];
STask g_astTasks_x264Enc[];
STask g_astTasks_H264Dec_PHONE[];
// ##TASK_LIST_DECLARATION_TEMPLATE::END


// ##TASK_GRAPH_DECLARATION_TEMPLATE::START
STaskGraph g_stGraph_top;
STaskGraph g_stGraph_MP3Dec;
STaskGraph g_stGraph_H264Dec_VIDEO;
STaskGraph g_stGraph_x264Enc;
STaskGraph g_stGraph_H264Dec_PHONE;
// ##TASK_GRAPH_DECLARATION_TEMPLATE::END


// ##MODE_TRANSITION_TEMPLATE::START
STask *g_pastRelatedChildTasks_x264Enc_Default[] = {
	&g_astTasks_x264Enc[1],
	&g_astTasks_x264Enc[2],
	&g_astTasks_x264Enc[0],
	&g_astTasks_x264Enc[3],
	&g_astTasks_x264Enc[4],
};
		
SModeMap g_astModeMap_x264Enc[] = {
	{
		0,
		"Default",
		g_pastRelatedChildTasks_x264Enc_Default,
		5,
	},
};

SVariableIntMap g_astVariableIntMap_x264Enc[] = {
};


SModeTransitionMachine g_stModeTransition_x264Enc = {
	11,
	g_astModeMap_x264Enc, // mode list
	1, // number of modes
	g_astVariableIntMap_x264Enc, // Integer variable list
	0, // number of integer variables
	NULL, // mode transition function
	0, // Current mode index
	0, // Next mode index
	MODE_STATE_TRANSITING, // mode state (to decide source task execution)
};
STask *g_pastRelatedChildTasks_MP3Dec_Default[] = {
	&g_astTasks_MP3Dec[7],
	&g_astTasks_MP3Dec[6],
	&g_astTasks_MP3Dec[5],
	&g_astTasks_MP3Dec[0],
	&g_astTasks_MP3Dec[3],
	&g_astTasks_MP3Dec[2],
	&g_astTasks_MP3Dec[1],
	&g_astTasks_MP3Dec[4],
};
		
SModeMap g_astModeMap_MP3Dec[] = {
	{
		0,
		"Default",
		g_pastRelatedChildTasks_MP3Dec_Default,
		8,
	},
};

SVariableIntMap g_astVariableIntMap_MP3Dec[] = {
};


SModeTransitionMachine g_stModeTransition_MP3Dec = {
	17,
	g_astModeMap_MP3Dec, // mode list
	1, // number of modes
	g_astVariableIntMap_MP3Dec, // Integer variable list
	0, // number of integer variables
	NULL, // mode transition function
	0, // Current mode index
	0, // Next mode index
	MODE_STATE_TRANSITING, // mode state (to decide source task execution)
};
STask *g_pastRelatedChildTasks_H264Dec_VIDEO_P_Frame[] = {
	&g_astTasks_H264Dec_VIDEO[9],
	&g_astTasks_H264Dec_VIDEO[0],
	&g_astTasks_H264Dec_VIDEO[2],
	&g_astTasks_H264Dec_VIDEO[4],
	&g_astTasks_H264Dec_VIDEO[3],
	&g_astTasks_H264Dec_VIDEO[1],
	&g_astTasks_H264Dec_VIDEO[8],
};
STask *g_pastRelatedChildTasks_H264Dec_VIDEO_I_Frame[] = {
	&g_astTasks_H264Dec_VIDEO[9],
	&g_astTasks_H264Dec_VIDEO[5],
	&g_astTasks_H264Dec_VIDEO[0],
	&g_astTasks_H264Dec_VIDEO[2],
	&g_astTasks_H264Dec_VIDEO[3],
	&g_astTasks_H264Dec_VIDEO[4],
	&g_astTasks_H264Dec_VIDEO[7],
	&g_astTasks_H264Dec_VIDEO[6],
	&g_astTasks_H264Dec_VIDEO[1],
	&g_astTasks_H264Dec_VIDEO[8],
};
		
SModeMap g_astModeMap_H264Dec_VIDEO[] = {
	{
		1,
		"P_Frame",
		g_pastRelatedChildTasks_H264Dec_VIDEO_P_Frame,
		7,
	},
	{
		0,
		"I_Frame",
		g_pastRelatedChildTasks_H264Dec_VIDEO_I_Frame,
		10,
	},
};

SVariableIntMap g_astVariableIntMap_H264Dec_VIDEO[] = {
	{
		0,
		"FrameVar",
		0, 
	},
};

static uem_bool transitMode_H264Dec_VIDEO(SModeTransitionMachine *pstModeTransition) 
{
	uem_bool bModeChanged = FALSE;
	int FrameVar;
	int nCurrentModeId = pstModeTransition->astModeMap[pstModeTransition->nCurModeIndex].nModeId;
	int nNextModeId = nCurrentModeId;
	int nVarIndex = 0;
	
	nVarIndex = UKModeTransition_GetVariableIndexByName(pstModeTransition, "FrameVar");
	FrameVar = pstModeTransition->astVarIntMap[nVarIndex].nValue;
		
	if(nCurrentModeId == 0
	  && FrameVar == 1 )
	{
		nNextModeId = 1;
		bModeChanged = TRUE;
	}
	if(nCurrentModeId == 1
	  && FrameVar == 2 )
	{
		nNextModeId = 0;
		bModeChanged = TRUE;
	}
	if(bModeChanged == TRUE)
	{	// update only the mode is changed
		pstModeTransition->nNextModeIndex = UKModeTransition_GetModeIndexByModeId(pstModeTransition, nNextModeId);
		pstModeTransition->enModeState = MODE_STATE_TRANSITING;
	}
	
	return bModeChanged;
}

SModeTransitionMachine g_stModeTransition_H264Dec_VIDEO = {
	0,
	g_astModeMap_H264Dec_VIDEO, // mode list
	2, // number of modes
	g_astVariableIntMap_H264Dec_VIDEO, // Integer variable list
	1, // number of integer variables
	transitMode_H264Dec_VIDEO, // mode transition function
	0, // Current mode index
	0, // Next mode index
	MODE_STATE_TRANSITING, // mode state (to decide source task execution)
};
STask *g_pastRelatedChildTasks_H264Dec_PHONE_P_Frame[] = {
	&g_astTasks_H264Dec_PHONE[2],
	&g_astTasks_H264Dec_PHONE[6],
	&g_astTasks_H264Dec_PHONE[9],
	&g_astTasks_H264Dec_PHONE[8],
	&g_astTasks_H264Dec_PHONE[4],
	&g_astTasks_H264Dec_PHONE[1],
	&g_astTasks_H264Dec_PHONE[0],
};
STask *g_pastRelatedChildTasks_H264Dec_PHONE_I_Frame[] = {
	&g_astTasks_H264Dec_PHONE[2],
	&g_astTasks_H264Dec_PHONE[5],
	&g_astTasks_H264Dec_PHONE[3],
	&g_astTasks_H264Dec_PHONE[7],
	&g_astTasks_H264Dec_PHONE[6],
	&g_astTasks_H264Dec_PHONE[9],
	&g_astTasks_H264Dec_PHONE[8],
	&g_astTasks_H264Dec_PHONE[4],
	&g_astTasks_H264Dec_PHONE[1],
	&g_astTasks_H264Dec_PHONE[0],
};
		
SModeMap g_astModeMap_H264Dec_PHONE[] = {
	{
		1,
		"P_Frame",
		g_pastRelatedChildTasks_H264Dec_PHONE_P_Frame,
		7,
	},
	{
		0,
		"I_Frame",
		g_pastRelatedChildTasks_H264Dec_PHONE_I_Frame,
		10,
	},
};

SVariableIntMap g_astVariableIntMap_H264Dec_PHONE[] = {
	{
		0,
		"FrameVar",
		0, 
	},
};

static uem_bool transitMode_H264Dec_PHONE(SModeTransitionMachine *pstModeTransition) 
{
	uem_bool bModeChanged = FALSE;
	int FrameVar;
	int nCurrentModeId = pstModeTransition->astModeMap[pstModeTransition->nCurModeIndex].nModeId;
	int nNextModeId = nCurrentModeId;
	int nVarIndex = 0;
	
	nVarIndex = UKModeTransition_GetVariableIndexByName(pstModeTransition, "FrameVar");
	FrameVar = pstModeTransition->astVarIntMap[nVarIndex].nValue;
		
	if(nCurrentModeId == 0
	  && FrameVar == 1 )
	{
		nNextModeId = 1;
		bModeChanged = TRUE;
	}
	if(nCurrentModeId == 1
	  && FrameVar == 2 )
	{
		nNextModeId = 0;
		bModeChanged = TRUE;
	}
	if(bModeChanged == TRUE)
	{	// update only the mode is changed
		pstModeTransition->nNextModeIndex = UKModeTransition_GetModeIndexByModeId(pstModeTransition, nNextModeId);
		pstModeTransition->enModeState = MODE_STATE_TRANSITING;
	}
	
	return bModeChanged;
}

SModeTransitionMachine g_stModeTransition_H264Dec_PHONE = {
	29,
	g_astModeMap_H264Dec_PHONE, // mode list
	2, // number of modes
	g_astVariableIntMap_H264Dec_PHONE, // Integer variable list
	1, // number of integer variables
	transitMode_H264Dec_PHONE, // mode transition function
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
SAvailableChunk g_astAvailableInputChunk_channel_15[] = {
	{ 0, 0, NULL, NULL, },
};
SAvailableChunk g_astAvailableInputChunk_channel_16[] = {
	{ 0, 0, NULL, NULL, },
};
SAvailableChunk g_astAvailableInputChunk_channel_17[] = {
	{ 0, 0, NULL, NULL, },
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
SAvailableChunk g_astAvailableInputChunk_channel_35[] = {
	{ 0, 0, NULL, NULL, },
};
SAvailableChunk g_astAvailableInputChunk_channel_36[] = {
	{ 0, 0, NULL, NULL, },
};
SAvailableChunk g_astAvailableInputChunk_channel_37[] = {
	{ 0, 0, NULL, NULL, },
};
SAvailableChunk g_astAvailableInputChunk_channel_38[] = {
	{ 0, 0, NULL, NULL, },
};
SAvailableChunk g_astAvailableInputChunk_channel_39[] = {
	{ 0, 0, NULL, NULL, },
};
SAvailableChunk g_astAvailableInputChunk_channel_40[] = {
	{ 0, 0, NULL, NULL, },
};
SAvailableChunk g_astAvailableInputChunk_channel_41[] = {
	{ 0, 0, NULL, NULL, },
};
SAvailableChunk g_astAvailableInputChunk_channel_42[] = {
	{ 0, 0, NULL, NULL, },
};
SAvailableChunk g_astAvailableInputChunk_channel_43[] = {
	{ 0, 0, NULL, NULL, },
};
SAvailableChunk g_astAvailableInputChunk_channel_44[] = {
	{ 0, 0, NULL, NULL, },
};
SAvailableChunk g_astAvailableInputChunk_channel_45[] = {
	{ 0, 0, NULL, NULL, },
};
SAvailableChunk g_astAvailableInputChunk_channel_46[] = {
	{ 0, 0, NULL, NULL, },
};
SAvailableChunk g_astAvailableInputChunk_channel_47[] = {
	{ 0, 0, NULL, NULL, },
};
// ##AVAILABLE_CHUNK_LIST_TEMPLATE::END

// ##TASK_PARAMETER_TEMPLATE::START
STaskParameter g_astTaskParameter_Control[] = {
	{
		0,
		PARAMETER_TYPE_INT,
		"check",
		{ .nParam = 0, },
	},
};
// ##TASK_PARAMETER_TEMPLATE::END

// ##TASK_FUNCTION_LIST::START
STaskFunctions g_ast_x264Enc_Init_3_functions[] = {
	{
		x264Enc_Init_3_Init0, // Task init function
		x264Enc_Init_3_Go0, // Task go function
		x264Enc_Init_3_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_H264Dec_VIDEO_Decode_functions[] = {
	{
		H264Dec_VIDEO_Decode_Init0, // Task init function
		H264Dec_VIDEO_Decode_Go0, // Task go function
		H264Dec_VIDEO_Decode_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_G723Enc_functions[] = {
	{
		G723Enc_Init0, // Task init function
		G723Enc_Go0, // Task go function
		G723Enc_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_x264Enc_Deblock_2_functions[] = {
	{
		x264Enc_Deblock_2_Init0, // Task init function
		x264Enc_Deblock_2_Go0, // Task go function
		x264Enc_Deblock_2_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_H264Dec_PHONE_Decode_functions[] = {
	{
		H264Dec_PHONE_Decode_Init0, // Task init function
		H264Dec_PHONE_Decode_Go0, // Task go function
		H264Dec_PHONE_Decode_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_x264Enc_functions[] = {
};

STaskFunctions g_ast_MP3Dec_VLDStream_functions[] = {
	{
		MP3Dec_VLDStream_Init0, // Task init function
		MP3Dec_VLDStream_Go0, // Task go function
		MP3Dec_VLDStream_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_x264Enc_Encoder_1_functions[] = {
	{
		x264Enc_Encoder_1_Init0, // Task init function
		x264Enc_Encoder_1_Go0, // Task go function
		x264Enc_Encoder_1_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_Control_functions[] = {
	{
		Control_Init0, // Task init function
		Control_Go0, // Task go function
		Control_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_H264Dec_VIDEO_IntraPredY_functions[] = {
	{
		H264Dec_VIDEO_IntraPredY_Init0, // Task init function
		H264Dec_VIDEO_IntraPredY_Go0, // Task go function
		H264Dec_VIDEO_IntraPredY_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_MP3Dec_functions[] = {
};

STaskFunctions g_ast_H264Dec_VIDEO_IntraPredV_functions[] = {
	{
		H264Dec_VIDEO_IntraPredV_Init0, // Task init function
		H264Dec_VIDEO_IntraPredV_Go0, // Task go function
		H264Dec_VIDEO_IntraPredV_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_H264Dec_VIDEO_WriteFileH_functions[] = {
	{
		H264Dec_VIDEO_WriteFileH_Init0, // Task init function
		H264Dec_VIDEO_WriteFileH_Go0, // Task go function
		H264Dec_VIDEO_WriteFileH_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_MP3Dec_Antialias_functions[] = {
	{
		MP3Dec_Antialias_Init0, // Task init function
		MP3Dec_Antialias_Go0, // Task go function
		MP3Dec_Antialias_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_MP3Dec_Subband_functions[] = {
	{
		MP3Dec_Subband_Init0, // Task init function
		MP3Dec_Subband_Go0, // Task go function
		MP3Dec_Subband_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_H264Dec_PHONE_InterPredY_functions[] = {
	{
		H264Dec_PHONE_InterPredY_Init0, // Task init function
		H264Dec_PHONE_InterPredY_Go0, // Task go function
		H264Dec_PHONE_InterPredY_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_H264Dec_VIDEO_Deblock_functions[] = {
	{
		H264Dec_VIDEO_Deblock_Init0, // Task init function
		H264Dec_VIDEO_Deblock_Go0, // Task go function
		H264Dec_VIDEO_Deblock_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_x264Enc_ME_0_functions[] = {
	{
		x264Enc_ME_0_Init0, // Task init function
		x264Enc_ME_0_Go0, // Task go function
		x264Enc_ME_0_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_H264Dec_VIDEO_InterPredU_functions[] = {
	{
		H264Dec_VIDEO_InterPredU_Init0, // Task init function
		H264Dec_VIDEO_InterPredU_Go0, // Task go function
		H264Dec_VIDEO_InterPredU_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_H264Dec_VIDEO_IntraPredU_functions[] = {
	{
		H264Dec_VIDEO_IntraPredU_Init0, // Task init function
		H264Dec_VIDEO_IntraPredU_Go0, // Task go function
		H264Dec_VIDEO_IntraPredU_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_UserInput_functions[] = {
	{
		UserInput_Init0, // Task init function
		UserInput_Go0, // Task go function
		UserInput_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_H264Dec_VIDEO_functions[] = {
};

STaskFunctions g_ast_H264Dec_PHONE_Deblock_functions[] = {
	{
		H264Dec_PHONE_Deblock_Init0, // Task init function
		H264Dec_PHONE_Deblock_Go0, // Task go function
		H264Dec_PHONE_Deblock_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_H264Dec_VIDEO_InterPredV_functions[] = {
	{
		H264Dec_VIDEO_InterPredV_Init0, // Task init function
		H264Dec_VIDEO_InterPredV_Go0, // Task go function
		H264Dec_VIDEO_InterPredV_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_H264Dec_PHONE_ReadFileH_functions[] = {
	{
		H264Dec_PHONE_ReadFileH_Init0, // Task init function
		H264Dec_PHONE_ReadFileH_Go0, // Task go function
		H264Dec_PHONE_ReadFileH_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_H264Dec_VIDEO_InterPredY_functions[] = {
	{
		H264Dec_VIDEO_InterPredY_Init0, // Task init function
		H264Dec_VIDEO_InterPredY_Go0, // Task go function
		H264Dec_VIDEO_InterPredY_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_MP3Dec_WriteFileM_functions[] = {
	{
		MP3Dec_WriteFileM_Init0, // Task init function
		MP3Dec_WriteFileM_Go0, // Task go function
		MP3Dec_WriteFileM_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_H264Dec_PHONE_functions[] = {
};

STaskFunctions g_ast_H264Dec_PHONE_IntraPredV_functions[] = {
	{
		H264Dec_PHONE_IntraPredV_Init0, // Task init function
		H264Dec_PHONE_IntraPredV_Go0, // Task go function
		H264Dec_PHONE_IntraPredV_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_Interrupt_functions[] = {
	{
		Interrupt_Init0, // Task init function
		Interrupt_Go0, // Task go function
		Interrupt_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_H264Dec_PHONE_IntraPredU_functions[] = {
	{
		H264Dec_PHONE_IntraPredU_Init0, // Task init function
		H264Dec_PHONE_IntraPredU_Go0, // Task go function
		H264Dec_PHONE_IntraPredU_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_H264Dec_VIDEO_ReadFileH_functions[] = {
	{
		H264Dec_VIDEO_ReadFileH_Init0, // Task init function
		H264Dec_VIDEO_ReadFileH_Go0, // Task go function
		H264Dec_VIDEO_ReadFileH_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_H264Dec_PHONE_IntraPredY_functions[] = {
	{
		H264Dec_PHONE_IntraPredY_Init0, // Task init function
		H264Dec_PHONE_IntraPredY_Go0, // Task go function
		H264Dec_PHONE_IntraPredY_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_MP3Dec_Stereo_functions[] = {
	{
		MP3Dec_Stereo_Init0, // Task init function
		MP3Dec_Stereo_Go0, // Task go function
		MP3Dec_Stereo_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_MP3Dec_DeQ_functions[] = {
	{
		MP3Dec_DeQ_Init0, // Task init function
		MP3Dec_DeQ_Go0, // Task go function
		MP3Dec_DeQ_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_H264Dec_PHONE_WriteFileH_functions[] = {
	{
		H264Dec_PHONE_WriteFileH_Init0, // Task init function
		H264Dec_PHONE_WriteFileH_Go0, // Task go function
		H264Dec_PHONE_WriteFileH_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_H264Dec_PHONE_InterPredU_functions[] = {
	{
		H264Dec_PHONE_InterPredU_Init0, // Task init function
		H264Dec_PHONE_InterPredU_Go0, // Task go function
		H264Dec_PHONE_InterPredU_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_H264Dec_PHONE_InterPredV_functions[] = {
	{
		H264Dec_PHONE_InterPredV_Init0, // Task init function
		H264Dec_PHONE_InterPredV_Go0, // Task go function
		H264Dec_PHONE_InterPredV_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_MP3Dec_Reorder_functions[] = {
	{
		MP3Dec_Reorder_Init0, // Task init function
		MP3Dec_Reorder_Go0, // Task go function
		MP3Dec_Reorder_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_x264Enc_VLC_4_functions[] = {
	{
		x264Enc_VLC_4_Init0, // Task init function
		x264Enc_VLC_4_Go0, // Task go function
		x264Enc_VLC_4_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_G723Dec_functions[] = {
	{
		G723Dec_Init0, // Task init function
		G723Dec_Go0, // Task go function
		G723Dec_Wrapup0, // Task wrapup function
	},
};

STaskFunctions g_ast_MP3Dec_Hybrid_functions[] = {
	{
		MP3Dec_Hybrid_Init0, // Task init function
		MP3Dec_Hybrid_Go0, // Task go function
		MP3Dec_Hybrid_Wrapup0, // Task wrapup function
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
			2, // Task ID
			"inFrame", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_H264Dec_VIDEO_Decode_inFrame, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			1, // Task ID
			"outFrame", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_H264Dec_VIDEO_ReadFileH_outFrame, // Array of sample rate list
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
			3, // Task ID
			"inMB_Y", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_H264Dec_VIDEO_InterPredY_inMB_Y, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			2, // Task ID
			"outMB_interY", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_H264Dec_VIDEO_Decode_outMB_interY, // Array of sample rate list
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
			4, // Task ID
			"inMB_U", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_H264Dec_VIDEO_InterPredU_inMB_U, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			2, // Task ID
			"outMB_interU", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_H264Dec_VIDEO_Decode_outMB_interU, // Array of sample rate list
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
			5, // Task ID
			"inMB_V", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_H264Dec_VIDEO_InterPredV_inMB_V, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			2, // Task ID
			"outMB_interV", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_H264Dec_VIDEO_Decode_outMB_interV, // Array of sample rate list
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
			6, // Task ID
			"inFrame_interY", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_H264Dec_VIDEO_Deblock_inFrame_interY, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			3, // Task ID
			"outFrame_Y", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_H264Dec_VIDEO_InterPredY_outFrame_Y, // Array of sample rate list
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
			6, // Task ID
			"inFrame_interU", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_H264Dec_VIDEO_Deblock_inFrame_interU, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			4, // Task ID
			"outFrame_U", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_H264Dec_VIDEO_InterPredU_outFrame_U, // Array of sample rate list
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
			6, // Task ID
			"inFrame_interV", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_H264Dec_VIDEO_Deblock_inFrame_interV, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			5, // Task ID
			"outFrame_V", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_H264Dec_VIDEO_InterPredV_outFrame_V, // Array of sample rate list
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
			7, // Task ID
			"inFrame", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_H264Dec_VIDEO_WriteFileH_inFrame, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			6, // Task ID
			"outFrame", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_H264Dec_VIDEO_Deblock_outFrame, // Array of sample rate list
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
			3, // Task ID
			"inPrevFrame", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_H264Dec_VIDEO_InterPredY_inPrevFrame, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			6, // Task ID
			"outRef_Y", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_H264Dec_VIDEO_Deblock_outRef_Y, // Array of sample rate list
			1, // Array element number of sample rate list
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
		25344, // Initial data length 
	},
	{
		9, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_GENERAL, // Channel type
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
			4, // Task ID
			"inPrevFrame", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_H264Dec_VIDEO_InterPredU_inPrevFrame, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			6, // Task ID
			"outRef_U", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_H264Dec_VIDEO_Deblock_outRef_U, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
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
		1,
		NULL, // Chunk list head
		NULL, // Chunk list tail
		6336, // Initial data length 
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
			5, // Task ID
			"inPrevFrame", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_H264Dec_VIDEO_InterPredV_inPrevFrame, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			6, // Task ID
			"outRef_V", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_H264Dec_VIDEO_Deblock_outRef_V, // Array of sample rate list
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
		1,
		NULL, // Chunk list head
		NULL, // Chunk list tail
		6336, // Initial data length 
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
			8, // Task ID
			"inMB_Y", // Port name
			PORT_SAMPLE_RATE_MULTIPLE, // Port sample rate type
			g_astPortSampleRate_H264Dec_VIDEO_IntraPredY_inMB_Y, // Array of sample rate list
			2, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			2, // Task ID
			"outMB_intraY", // Port name
			PORT_SAMPLE_RATE_MULTIPLE, // Port sample rate type
			g_astPortSampleRate_H264Dec_VIDEO_Decode_outMB_intraY, // Array of sample rate list
			2, // Array element number of sample rate list
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
		1,
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
			9, // Task ID
			"inMB_U", // Port name
			PORT_SAMPLE_RATE_MULTIPLE, // Port sample rate type
			g_astPortSampleRate_H264Dec_VIDEO_IntraPredU_inMB_U, // Array of sample rate list
			2, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			2, // Task ID
			"outMB_intraU", // Port name
			PORT_SAMPLE_RATE_MULTIPLE, // Port sample rate type
			g_astPortSampleRate_H264Dec_VIDEO_Decode_outMB_intraU, // Array of sample rate list
			2, // Array element number of sample rate list
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
		1,
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
			10, // Task ID
			"inMB_V", // Port name
			PORT_SAMPLE_RATE_MULTIPLE, // Port sample rate type
			g_astPortSampleRate_H264Dec_VIDEO_IntraPredV_inMB_V, // Array of sample rate list
			2, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			2, // Task ID
			"outMB_intraV", // Port name
			PORT_SAMPLE_RATE_MULTIPLE, // Port sample rate type
			g_astPortSampleRate_H264Dec_VIDEO_Decode_outMB_intraV, // Array of sample rate list
			2, // Array element number of sample rate list
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
		1,
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
			6, // Task ID
			"inFrame_intraY", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_H264Dec_VIDEO_Deblock_inFrame_intraY, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			8, // Task ID
			"outFrame_Y", // Port name
			PORT_SAMPLE_RATE_MULTIPLE, // Port sample rate type
			g_astPortSampleRate_H264Dec_VIDEO_IntraPredY_outFrame_Y, // Array of sample rate list
			2, // Array element number of sample rate list
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
		1,
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
			6, // Task ID
			"inFrame_intraU", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_H264Dec_VIDEO_Deblock_inFrame_intraU, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			9, // Task ID
			"outFrame_U", // Port name
			PORT_SAMPLE_RATE_MULTIPLE, // Port sample rate type
			g_astPortSampleRate_H264Dec_VIDEO_IntraPredU_outFrame_U, // Array of sample rate list
			2, // Array element number of sample rate list
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
		1,
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
			6, // Task ID
			"inFrame_intraV", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_H264Dec_VIDEO_Deblock_inFrame_intraV, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			10, // Task ID
			"outFrame_V", // Port name
			PORT_SAMPLE_RATE_MULTIPLE, // Port sample rate type
			g_astPortSampleRate_H264Dec_VIDEO_IntraPredV_outFrame_V, // Array of sample rate list
			2, // Array element number of sample rate list
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
		1,
		NULL, // Chunk list head
		NULL, // Chunk list tail
		0, // Initial data length 
	},
	{
		17, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_GENERAL, // Channel type
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
			12, // Task ID
			"p101", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_x264Enc_ME_0_p101, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			11324, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			15, // Task ID
			"p71", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_x264Enc_Init_3_p71, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
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
		1,
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
			13, // Task ID
			"p52", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_x264Enc_Encoder_1_p52, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			12372, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			12, // Task ID
			"p105", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_x264Enc_ME_0_p105, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			12372, // Sample size
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
		1,
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
			16, // Task ID
			"p14", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_x264Enc_VLC_4_p14, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			32, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			13, // Task ID
			"p43", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_x264Enc_Encoder_1_p43, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			64, // Sample size
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
		1,
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
			14, // Task ID
			"p24", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_x264Enc_Deblock_2_p24, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			64, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			13, // Task ID
			"p28", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_x264Enc_Encoder_1_p28, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			32, // Sample size
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
		1,
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
			12, // Task ID
			"p102", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_x264Enc_ME_0_p102, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			396, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			14, // Task ID
			"p42", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_x264Enc_Deblock_2_p42, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			11324, // Sample size
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
		1,
		NULL, // Chunk list head
		NULL, // Chunk list tail
		11324, // Initial data length 
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
			19, // Task ID
			"input", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_MP3Dec_DeQ_input, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			18, // Task ID
			"output", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_MP3Dec_VLDStream_output, // Array of sample rate list
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
		1,
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
			23, // Task ID
			"input", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_MP3Dec_Hybrid_input, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			22, // Task ID
			"output", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_MP3Dec_Antialias_output, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
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
		1,
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
			24, // Task ID
			"input", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_MP3Dec_Subband_input, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			23, // Task ID
			"output", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_MP3Dec_Hybrid_output, // Array of sample rate list
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
		1,
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
			25, // Task ID
			"input", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_MP3Dec_WriteFileM_input, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			24, // Task ID
			"output", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_MP3Dec_Subband_output, // Array of sample rate list
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
		1,
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
			21, // Task ID
			"input", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_MP3Dec_Reorder_input, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			19, // Task ID
			"output", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_MP3Dec_DeQ_output, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
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
		1,
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
			20, // Task ID
			"input", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_MP3Dec_Stereo_input, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			21, // Task ID
			"output", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_MP3Dec_Reorder_output, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
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
		1,
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
			22, // Task ID
			"input", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_MP3Dec_Antialias_input, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			20, // Task ID
			"output", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_MP3Dec_Stereo_output, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
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
		1,
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
			31, // Task ID
			"inFrame", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_H264Dec_PHONE_Decode_inFrame, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			30, // Task ID
			"outFrame", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_H264Dec_PHONE_ReadFileH_outFrame, // Array of sample rate list
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
		1,
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
			32, // Task ID
			"inMB_Y", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_H264Dec_PHONE_InterPredY_inMB_Y, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			31, // Task ID
			"outMB_interY", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_H264Dec_PHONE_Decode_outMB_interY, // Array of sample rate list
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
		1,
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
			33, // Task ID
			"inMB_U", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_H264Dec_PHONE_InterPredU_inMB_U, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			31, // Task ID
			"outMB_interU", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_H264Dec_PHONE_Decode_outMB_interU, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
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
		1,
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
			34, // Task ID
			"inMB_V", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_H264Dec_PHONE_InterPredV_inMB_V, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			31, // Task ID
			"outMB_interV", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_H264Dec_PHONE_Decode_outMB_interV, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
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
		1,
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
			35, // Task ID
			"inFrame_interY", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_H264Dec_PHONE_Deblock_inFrame_interY, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			32, // Task ID
			"outFrame_Y", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_H264Dec_PHONE_InterPredY_outFrame_Y, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
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
		1,
		NULL, // Chunk list head
		NULL, // Chunk list tail
		0, // Initial data length 
	},
	{
		34, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_GENERAL, // Channel type
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
			35, // Task ID
			"inFrame_interU", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_H264Dec_PHONE_Deblock_inFrame_interU, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			33, // Task ID
			"outFrame_U", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_H264Dec_PHONE_InterPredU_outFrame_U, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
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
		1,
		NULL, // Chunk list head
		NULL, // Chunk list tail
		0, // Initial data length 
	},
	{
		35, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_GENERAL, // Channel type
		s_pChannel_35_buffer, // Channel buffer pointer
		CHANNEL_35_SIZE, // Channel size
		s_pChannel_35_buffer, // Channel data start
		s_pChannel_35_buffer, // Channel data end
		0, // Channel data length
		0, // Read reference count
		0, // Write reference count
		FALSE, // Read exit setting
		FALSE, // Write exit setting
		NULL, // Mutex
		NULL, // Read available notice event
		NULL, // Write available notice event
		{
			35, // Task ID
			"inFrame_interV", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_H264Dec_PHONE_Deblock_inFrame_interV, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			34, // Task ID
			"outFrame_V", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_H264Dec_PHONE_InterPredV_outFrame_V, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
		{
			g_astChunk_channel_35_in, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Input chunk information
		{
			g_astChunk_channel_35_out, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Output chunk information
		CHUNK_NUM_NOT_INITIALIZED, // Written output chunk number
		g_astAvailableInputChunk_channel_35, // Available chunk list
		1,
		NULL, // Chunk list head
		NULL, // Chunk list tail
		0, // Initial data length 
	},
	{
		36, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_GENERAL, // Channel type
		s_pChannel_36_buffer, // Channel buffer pointer
		CHANNEL_36_SIZE, // Channel size
		s_pChannel_36_buffer, // Channel data start
		s_pChannel_36_buffer, // Channel data end
		0, // Channel data length
		0, // Read reference count
		0, // Write reference count
		FALSE, // Read exit setting
		FALSE, // Write exit setting
		NULL, // Mutex
		NULL, // Read available notice event
		NULL, // Write available notice event
		{
			36, // Task ID
			"inFrame", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_H264Dec_PHONE_WriteFileH_inFrame, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			35, // Task ID
			"outFrame", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_H264Dec_PHONE_Deblock_outFrame, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
		{
			g_astChunk_channel_36_in, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Input chunk information
		{
			g_astChunk_channel_36_out, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Output chunk information
		CHUNK_NUM_NOT_INITIALIZED, // Written output chunk number
		g_astAvailableInputChunk_channel_36, // Available chunk list
		1,
		NULL, // Chunk list head
		NULL, // Chunk list tail
		0, // Initial data length 
	},
	{
		37, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_GENERAL, // Channel type
		s_pChannel_37_buffer, // Channel buffer pointer
		CHANNEL_37_SIZE, // Channel size
		s_pChannel_37_buffer, // Channel data start
		s_pChannel_37_buffer, // Channel data end
		0, // Channel data length
		0, // Read reference count
		0, // Write reference count
		FALSE, // Read exit setting
		FALSE, // Write exit setting
		NULL, // Mutex
		NULL, // Read available notice event
		NULL, // Write available notice event
		{
			32, // Task ID
			"inPrevFrame", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_H264Dec_PHONE_InterPredY_inPrevFrame, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			35, // Task ID
			"outRef_Y", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_H264Dec_PHONE_Deblock_outRef_Y, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
		{
			g_astChunk_channel_37_in, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Input chunk information
		{
			g_astChunk_channel_37_out, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Output chunk information
		CHUNK_NUM_NOT_INITIALIZED, // Written output chunk number
		g_astAvailableInputChunk_channel_37, // Available chunk list
		1,
		NULL, // Chunk list head
		NULL, // Chunk list tail
		25344, // Initial data length 
	},
	{
		38, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_GENERAL, // Channel type
		s_pChannel_38_buffer, // Channel buffer pointer
		CHANNEL_38_SIZE, // Channel size
		s_pChannel_38_buffer, // Channel data start
		s_pChannel_38_buffer, // Channel data end
		0, // Channel data length
		0, // Read reference count
		0, // Write reference count
		FALSE, // Read exit setting
		FALSE, // Write exit setting
		NULL, // Mutex
		NULL, // Read available notice event
		NULL, // Write available notice event
		{
			33, // Task ID
			"inPrevFrame", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_H264Dec_PHONE_InterPredU_inPrevFrame, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			35, // Task ID
			"outRef_U", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_H264Dec_PHONE_Deblock_outRef_U, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
		{
			g_astChunk_channel_38_in, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Input chunk information
		{
			g_astChunk_channel_38_out, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Output chunk information
		CHUNK_NUM_NOT_INITIALIZED, // Written output chunk number
		g_astAvailableInputChunk_channel_38, // Available chunk list
		1,
		NULL, // Chunk list head
		NULL, // Chunk list tail
		6336, // Initial data length 
	},
	{
		39, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_GENERAL, // Channel type
		s_pChannel_39_buffer, // Channel buffer pointer
		CHANNEL_39_SIZE, // Channel size
		s_pChannel_39_buffer, // Channel data start
		s_pChannel_39_buffer, // Channel data end
		0, // Channel data length
		0, // Read reference count
		0, // Write reference count
		FALSE, // Read exit setting
		FALSE, // Write exit setting
		NULL, // Mutex
		NULL, // Read available notice event
		NULL, // Write available notice event
		{
			34, // Task ID
			"inPrevFrame", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_H264Dec_PHONE_InterPredV_inPrevFrame, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			35, // Task ID
			"outRef_V", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_H264Dec_PHONE_Deblock_outRef_V, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
		{
			g_astChunk_channel_39_in, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Input chunk information
		{
			g_astChunk_channel_39_out, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Output chunk information
		CHUNK_NUM_NOT_INITIALIZED, // Written output chunk number
		g_astAvailableInputChunk_channel_39, // Available chunk list
		1,
		NULL, // Chunk list head
		NULL, // Chunk list tail
		6336, // Initial data length 
	},
	{
		40, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_GENERAL, // Channel type
		s_pChannel_40_buffer, // Channel buffer pointer
		CHANNEL_40_SIZE, // Channel size
		s_pChannel_40_buffer, // Channel data start
		s_pChannel_40_buffer, // Channel data end
		0, // Channel data length
		0, // Read reference count
		0, // Write reference count
		FALSE, // Read exit setting
		FALSE, // Write exit setting
		NULL, // Mutex
		NULL, // Read available notice event
		NULL, // Write available notice event
		{
			37, // Task ID
			"inMB_Y", // Port name
			PORT_SAMPLE_RATE_MULTIPLE, // Port sample rate type
			g_astPortSampleRate_H264Dec_PHONE_IntraPredY_inMB_Y, // Array of sample rate list
			2, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			31, // Task ID
			"outMB_intraY", // Port name
			PORT_SAMPLE_RATE_MULTIPLE, // Port sample rate type
			g_astPortSampleRate_H264Dec_PHONE_Decode_outMB_intraY, // Array of sample rate list
			2, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
		{
			g_astChunk_channel_40_in, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Input chunk information
		{
			g_astChunk_channel_40_out, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Output chunk information
		CHUNK_NUM_NOT_INITIALIZED, // Written output chunk number
		g_astAvailableInputChunk_channel_40, // Available chunk list
		1,
		NULL, // Chunk list head
		NULL, // Chunk list tail
		0, // Initial data length 
	},
	{
		41, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_GENERAL, // Channel type
		s_pChannel_41_buffer, // Channel buffer pointer
		CHANNEL_41_SIZE, // Channel size
		s_pChannel_41_buffer, // Channel data start
		s_pChannel_41_buffer, // Channel data end
		0, // Channel data length
		0, // Read reference count
		0, // Write reference count
		FALSE, // Read exit setting
		FALSE, // Write exit setting
		NULL, // Mutex
		NULL, // Read available notice event
		NULL, // Write available notice event
		{
			38, // Task ID
			"inMB_U", // Port name
			PORT_SAMPLE_RATE_MULTIPLE, // Port sample rate type
			g_astPortSampleRate_H264Dec_PHONE_IntraPredU_inMB_U, // Array of sample rate list
			2, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			31, // Task ID
			"outMB_intraU", // Port name
			PORT_SAMPLE_RATE_MULTIPLE, // Port sample rate type
			g_astPortSampleRate_H264Dec_PHONE_Decode_outMB_intraU, // Array of sample rate list
			2, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
		{
			g_astChunk_channel_41_in, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Input chunk information
		{
			g_astChunk_channel_41_out, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Output chunk information
		CHUNK_NUM_NOT_INITIALIZED, // Written output chunk number
		g_astAvailableInputChunk_channel_41, // Available chunk list
		1,
		NULL, // Chunk list head
		NULL, // Chunk list tail
		0, // Initial data length 
	},
	{
		42, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_GENERAL, // Channel type
		s_pChannel_42_buffer, // Channel buffer pointer
		CHANNEL_42_SIZE, // Channel size
		s_pChannel_42_buffer, // Channel data start
		s_pChannel_42_buffer, // Channel data end
		0, // Channel data length
		0, // Read reference count
		0, // Write reference count
		FALSE, // Read exit setting
		FALSE, // Write exit setting
		NULL, // Mutex
		NULL, // Read available notice event
		NULL, // Write available notice event
		{
			39, // Task ID
			"inMB_V", // Port name
			PORT_SAMPLE_RATE_MULTIPLE, // Port sample rate type
			g_astPortSampleRate_H264Dec_PHONE_IntraPredV_inMB_V, // Array of sample rate list
			2, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			31, // Task ID
			"outMB_intraV", // Port name
			PORT_SAMPLE_RATE_MULTIPLE, // Port sample rate type
			g_astPortSampleRate_H264Dec_PHONE_Decode_outMB_intraV, // Array of sample rate list
			2, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
		{
			g_astChunk_channel_42_in, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Input chunk information
		{
			g_astChunk_channel_42_out, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Output chunk information
		CHUNK_NUM_NOT_INITIALIZED, // Written output chunk number
		g_astAvailableInputChunk_channel_42, // Available chunk list
		1,
		NULL, // Chunk list head
		NULL, // Chunk list tail
		0, // Initial data length 
	},
	{
		43, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_GENERAL, // Channel type
		s_pChannel_43_buffer, // Channel buffer pointer
		CHANNEL_43_SIZE, // Channel size
		s_pChannel_43_buffer, // Channel data start
		s_pChannel_43_buffer, // Channel data end
		0, // Channel data length
		0, // Read reference count
		0, // Write reference count
		FALSE, // Read exit setting
		FALSE, // Write exit setting
		NULL, // Mutex
		NULL, // Read available notice event
		NULL, // Write available notice event
		{
			35, // Task ID
			"inFrame_intraY", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_H264Dec_PHONE_Deblock_inFrame_intraY, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			37, // Task ID
			"outFrame_Y", // Port name
			PORT_SAMPLE_RATE_MULTIPLE, // Port sample rate type
			g_astPortSampleRate_H264Dec_PHONE_IntraPredY_outFrame_Y, // Array of sample rate list
			2, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
		{
			g_astChunk_channel_43_in, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Input chunk information
		{
			g_astChunk_channel_43_out, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Output chunk information
		CHUNK_NUM_NOT_INITIALIZED, // Written output chunk number
		g_astAvailableInputChunk_channel_43, // Available chunk list
		1,
		NULL, // Chunk list head
		NULL, // Chunk list tail
		0, // Initial data length 
	},
	{
		44, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_GENERAL, // Channel type
		s_pChannel_44_buffer, // Channel buffer pointer
		CHANNEL_44_SIZE, // Channel size
		s_pChannel_44_buffer, // Channel data start
		s_pChannel_44_buffer, // Channel data end
		0, // Channel data length
		0, // Read reference count
		0, // Write reference count
		FALSE, // Read exit setting
		FALSE, // Write exit setting
		NULL, // Mutex
		NULL, // Read available notice event
		NULL, // Write available notice event
		{
			35, // Task ID
			"inFrame_intraU", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_H264Dec_PHONE_Deblock_inFrame_intraU, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			38, // Task ID
			"outFrame_U", // Port name
			PORT_SAMPLE_RATE_MULTIPLE, // Port sample rate type
			g_astPortSampleRate_H264Dec_PHONE_IntraPredU_outFrame_U, // Array of sample rate list
			2, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
		{
			g_astChunk_channel_44_in, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Input chunk information
		{
			g_astChunk_channel_44_out, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Output chunk information
		CHUNK_NUM_NOT_INITIALIZED, // Written output chunk number
		g_astAvailableInputChunk_channel_44, // Available chunk list
		1,
		NULL, // Chunk list head
		NULL, // Chunk list tail
		0, // Initial data length 
	},
	{
		45, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_GENERAL, // Channel type
		s_pChannel_45_buffer, // Channel buffer pointer
		CHANNEL_45_SIZE, // Channel size
		s_pChannel_45_buffer, // Channel data start
		s_pChannel_45_buffer, // Channel data end
		0, // Channel data length
		0, // Read reference count
		0, // Write reference count
		FALSE, // Read exit setting
		FALSE, // Write exit setting
		NULL, // Mutex
		NULL, // Read available notice event
		NULL, // Write available notice event
		{
			35, // Task ID
			"inFrame_intraV", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_H264Dec_PHONE_Deblock_inFrame_intraV, // Array of sample rate list
			1, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			39, // Task ID
			"outFrame_V", // Port name
			PORT_SAMPLE_RATE_MULTIPLE, // Port sample rate type
			g_astPortSampleRate_H264Dec_PHONE_IntraPredV_outFrame_V, // Array of sample rate list
			2, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
		{
			g_astChunk_channel_45_in, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Input chunk information
		{
			g_astChunk_channel_45_out, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Output chunk information
		CHUNK_NUM_NOT_INITIALIZED, // Written output chunk number
		g_astAvailableInputChunk_channel_45, // Available chunk list
		1,
		NULL, // Chunk list head
		NULL, // Chunk list tail
		0, // Initial data length 
	},
	{
		46, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_GENERAL, // Channel type
		s_pChannel_46_buffer, // Channel buffer pointer
		CHANNEL_46_SIZE, // Channel size
		s_pChannel_46_buffer, // Channel data start
		s_pChannel_46_buffer, // Channel data end
		0, // Channel data length
		0, // Read reference count
		0, // Write reference count
		FALSE, // Read exit setting
		FALSE, // Write exit setting
		NULL, // Mutex
		NULL, // Read available notice event
		NULL, // Write available notice event
		{
			41, // Task ID
			"in_UserInput", // Port name
			PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
			g_astPortSampleRate_Control_in_UserInput, // Array of sample rate list
			0, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			28, // Task ID
			"out", // Port name
			PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
			g_astPortSampleRate_UserInput_out, // Array of sample rate list
			0, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
		{
			g_astChunk_channel_46_in, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Input chunk information
		{
			g_astChunk_channel_46_out, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Output chunk information
		CHUNK_NUM_NOT_INITIALIZED, // Written output chunk number
		g_astAvailableInputChunk_channel_46, // Available chunk list
		1,
		NULL, // Chunk list head
		NULL, // Chunk list tail
		0, // Initial data length 
	},
	{
		47, // Channel ID
		-1, // Next channel index (which is used for single port is connecting to multiple channels)
		COMMUNICATION_TYPE_SHARED_MEMORY, // Channel communication type
		CHANNEL_TYPE_GENERAL, // Channel type
		s_pChannel_47_buffer, // Channel buffer pointer
		CHANNEL_47_SIZE, // Channel size
		s_pChannel_47_buffer, // Channel data start
		s_pChannel_47_buffer, // Channel data end
		0, // Channel data length
		0, // Read reference count
		0, // Write reference count
		FALSE, // Read exit setting
		FALSE, // Write exit setting
		NULL, // Mutex
		NULL, // Read available notice event
		NULL, // Write available notice event
		{
			41, // Task ID
			"in_Interrupt", // Port name
			PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
			g_astPortSampleRate_Control_in_Interrupt, // Array of sample rate list
			0, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			40, // Task ID
			"out", // Port name
			PORT_SAMPLE_RATE_VARIABLE, // Port sample rate type
			g_astPortSampleRate_Interrupt_out, // Array of sample rate list
			0, // Array element number of sample rate list
			0, //Selected sample rate index
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
		{
			g_astChunk_channel_47_in, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Input chunk information
		{
			g_astChunk_channel_47_out, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Output chunk information
		CHUNK_NUM_NOT_INITIALIZED, // Written output chunk number
		g_astAvailableInputChunk_channel_47, // Available chunk list
		1,
		NULL, // Chunk list head
		NULL, // Chunk list tail
		0, // Initial data length 
	},
};
// ##CHANNEL_LIST_TEMPLATE::END


// ##TASK_LIST_TEMPLATE::START
STask g_astTasks_top[] = {
	{ 	29, // Task ID
		"H264Dec_PHONE", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_H264Dec_PHONE_functions, // Task function array
		0, // Task function array number
		RUN_CONDITION_TIME_DRIVEN, // Run condition
		1, // Run rate
		1, // Period
		TIME_METRIC_MILLISEC, // Period metric
		&g_stGraph_H264Dec_PHONE, // Subgraph
		&g_stGraph_top, // Parent task graph
		&g_stModeTransition_H264Dec_PHONE, // MTM information
		NULL, // Loop information
		NULL, // Task parameter information
		0, // Task parameter number
		TRUE, // Statically scheduled or not
		0,	  // Throughput constraint
		NULL, // Mutex
		NULL, // Conditional variable
	},
	{ 	0, // Task ID
		"H264Dec_VIDEO", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_H264Dec_VIDEO_functions, // Task function array
		0, // Task function array number
		RUN_CONDITION_TIME_DRIVEN, // Run condition
		1, // Run rate
		1, // Period
		TIME_METRIC_MILLISEC, // Period metric
		&g_stGraph_H264Dec_VIDEO, // Subgraph
		&g_stGraph_top, // Parent task graph
		&g_stModeTransition_H264Dec_VIDEO, // MTM information
		NULL, // Loop information
		NULL, // Task parameter information
		0, // Task parameter number
		TRUE, // Statically scheduled or not
		0,	  // Throughput constraint
		NULL, // Mutex
		NULL, // Conditional variable
	},
	{ 	17, // Task ID
		"MP3Dec", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_MP3Dec_functions, // Task function array
		0, // Task function array number
		RUN_CONDITION_TIME_DRIVEN, // Run condition
		1, // Run rate
		1, // Period
		TIME_METRIC_MILLISEC, // Period metric
		&g_stGraph_MP3Dec, // Subgraph
		&g_stGraph_top, // Parent task graph
		&g_stModeTransition_MP3Dec, // MTM information
		NULL, // Loop information
		NULL, // Task parameter information
		0, // Task parameter number
		TRUE, // Statically scheduled or not
		0,	  // Throughput constraint
		NULL, // Mutex
		NULL, // Conditional variable
	},
	{ 	11, // Task ID
		"x264Enc", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_x264Enc_functions, // Task function array
		0, // Task function array number
		RUN_CONDITION_TIME_DRIVEN, // Run condition
		1, // Run rate
		100, // Period
		TIME_METRIC_MICROSEC, // Period metric
		&g_stGraph_x264Enc, // Subgraph
		&g_stGraph_top, // Parent task graph
		&g_stModeTransition_x264Enc, // MTM information
		NULL, // Loop information
		NULL, // Task parameter information
		0, // Task parameter number
		TRUE, // Statically scheduled or not
		0,	  // Throughput constraint
		NULL, // Mutex
		NULL, // Conditional variable
	},
	{ 	26, // Task ID
		"G723Dec", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_G723Dec_functions, // Task function array
		1, // Task function array number
		RUN_CONDITION_TIME_DRIVEN, // Run condition
		1, // Run rate
		4455000, // Period
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
	{ 	27, // Task ID
		"G723Enc", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_G723Enc_functions, // Task function array
		1, // Task function array number
		RUN_CONDITION_TIME_DRIVEN, // Run condition
		1, // Run rate
		4455000, // Period
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
	{ 	28, // Task ID
		"UserInput", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_UserInput_functions, // Task function array
		1, // Task function array number
		RUN_CONDITION_TIME_DRIVEN, // Run condition
		1, // Run rate
		4455000, // Period
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
	{ 	40, // Task ID
		"Interrupt", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_Interrupt_functions, // Task function array
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
	{ 	41, // Task ID
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
		g_astTaskParameter_Control, // Task parameter information
		1, // Task parameter number
		FALSE, // Statically scheduled or not
		0,	  // Throughput constraint
		NULL, // Mutex
		NULL, // Conditional variable
	},
};

STask g_astTasks_MP3Dec[] = {
	{ 	22, // Task ID
		"MP3Dec_Antialias", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_MP3Dec_Antialias_functions, // Task function array
		1, // Task function array number
		RUN_CONDITION_DATA_DRIVEN, // Run condition
		1, // Run rate
		1, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // Subgraph
		&g_stGraph_MP3Dec, // Parent task graph
		NULL, // MTM information
		NULL, // Loop information
		NULL, // Task parameter information
		0, // Task parameter number
		TRUE, // Statically scheduled or not
		0,	  // Throughput constraint
		NULL, // Mutex
		NULL, // Conditional variable
	},
	{ 	24, // Task ID
		"MP3Dec_Subband", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_MP3Dec_Subband_functions, // Task function array
		1, // Task function array number
		RUN_CONDITION_DATA_DRIVEN, // Run condition
		1, // Run rate
		1, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // Subgraph
		&g_stGraph_MP3Dec, // Parent task graph
		NULL, // MTM information
		NULL, // Loop information
		NULL, // Task parameter information
		0, // Task parameter number
		TRUE, // Statically scheduled or not
		0,	  // Throughput constraint
		NULL, // Mutex
		NULL, // Conditional variable
	},
	{ 	19, // Task ID
		"MP3Dec_DeQ", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_MP3Dec_DeQ_functions, // Task function array
		1, // Task function array number
		RUN_CONDITION_DATA_DRIVEN, // Run condition
		1, // Run rate
		1, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // Subgraph
		&g_stGraph_MP3Dec, // Parent task graph
		NULL, // MTM information
		NULL, // Loop information
		NULL, // Task parameter information
		0, // Task parameter number
		TRUE, // Statically scheduled or not
		0,	  // Throughput constraint
		NULL, // Mutex
		NULL, // Conditional variable
	},
	{ 	20, // Task ID
		"MP3Dec_Stereo", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_MP3Dec_Stereo_functions, // Task function array
		1, // Task function array number
		RUN_CONDITION_DATA_DRIVEN, // Run condition
		1, // Run rate
		1, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // Subgraph
		&g_stGraph_MP3Dec, // Parent task graph
		NULL, // MTM information
		NULL, // Loop information
		NULL, // Task parameter information
		0, // Task parameter number
		TRUE, // Statically scheduled or not
		0,	  // Throughput constraint
		NULL, // Mutex
		NULL, // Conditional variable
	},
	{ 	23, // Task ID
		"MP3Dec_Hybrid", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_MP3Dec_Hybrid_functions, // Task function array
		1, // Task function array number
		RUN_CONDITION_DATA_DRIVEN, // Run condition
		1, // Run rate
		1, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // Subgraph
		&g_stGraph_MP3Dec, // Parent task graph
		NULL, // MTM information
		NULL, // Loop information
		NULL, // Task parameter information
		0, // Task parameter number
		TRUE, // Statically scheduled or not
		0,	  // Throughput constraint
		NULL, // Mutex
		NULL, // Conditional variable
	},
	{ 	25, // Task ID
		"MP3Dec_WriteFileM", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_MP3Dec_WriteFileM_functions, // Task function array
		1, // Task function array number
		RUN_CONDITION_DATA_DRIVEN, // Run condition
		1, // Run rate
		1, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // Subgraph
		&g_stGraph_MP3Dec, // Parent task graph
		NULL, // MTM information
		NULL, // Loop information
		NULL, // Task parameter information
		0, // Task parameter number
		TRUE, // Statically scheduled or not
		0,	  // Throughput constraint
		NULL, // Mutex
		NULL, // Conditional variable
	},
	{ 	18, // Task ID
		"MP3Dec_VLDStream", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_MP3Dec_VLDStream_functions, // Task function array
		1, // Task function array number
		RUN_CONDITION_TIME_DRIVEN, // Run condition
		1, // Run rate
		1, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // Subgraph
		&g_stGraph_MP3Dec, // Parent task graph
		NULL, // MTM information
		NULL, // Loop information
		NULL, // Task parameter information
		0, // Task parameter number
		TRUE, // Statically scheduled or not
		0,	  // Throughput constraint
		NULL, // Mutex
		NULL, // Conditional variable
	},
	{ 	21, // Task ID
		"MP3Dec_Reorder", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_MP3Dec_Reorder_functions, // Task function array
		1, // Task function array number
		RUN_CONDITION_DATA_DRIVEN, // Run condition
		1, // Run rate
		1, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // Subgraph
		&g_stGraph_MP3Dec, // Parent task graph
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

STask g_astTasks_H264Dec_VIDEO[] = {
	{ 	1, // Task ID
		"H264Dec_VIDEO_ReadFileH", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_H264Dec_VIDEO_ReadFileH_functions, // Task function array
		1, // Task function array number
		RUN_CONDITION_TIME_DRIVEN, // Run condition
		1, // Run rate
		1, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // Subgraph
		&g_stGraph_H264Dec_VIDEO, // Parent task graph
		NULL, // MTM information
		NULL, // Loop information
		NULL, // Task parameter information
		0, // Task parameter number
		TRUE, // Statically scheduled or not
		0,	  // Throughput constraint
		NULL, // Mutex
		NULL, // Conditional variable
	},
	{ 	2, // Task ID
		"H264Dec_VIDEO_Decode", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_H264Dec_VIDEO_Decode_functions, // Task function array
		1, // Task function array number
		RUN_CONDITION_DATA_DRIVEN, // Run condition
		1, // Run rate
		1, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // Subgraph
		&g_stGraph_H264Dec_VIDEO, // Parent task graph
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
		"H264Dec_VIDEO_InterPredY", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_H264Dec_VIDEO_InterPredY_functions, // Task function array
		1, // Task function array number
		RUN_CONDITION_DATA_DRIVEN, // Run condition
		1, // Run rate
		1, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // Subgraph
		&g_stGraph_H264Dec_VIDEO, // Parent task graph
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
		"H264Dec_VIDEO_InterPredV", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_H264Dec_VIDEO_InterPredV_functions, // Task function array
		1, // Task function array number
		RUN_CONDITION_DATA_DRIVEN, // Run condition
		1, // Run rate
		1, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // Subgraph
		&g_stGraph_H264Dec_VIDEO, // Parent task graph
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
		"H264Dec_VIDEO_InterPredU", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_H264Dec_VIDEO_InterPredU_functions, // Task function array
		1, // Task function array number
		RUN_CONDITION_DATA_DRIVEN, // Run condition
		1, // Run rate
		1, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // Subgraph
		&g_stGraph_H264Dec_VIDEO, // Parent task graph
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
		"H264Dec_VIDEO_IntraPredU", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_H264Dec_VIDEO_IntraPredU_functions, // Task function array
		1, // Task function array number
		RUN_CONDITION_DATA_DRIVEN, // Run condition
		1, // Run rate
		1, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // Subgraph
		&g_stGraph_H264Dec_VIDEO, // Parent task graph
		NULL, // MTM information
		NULL, // Loop information
		NULL, // Task parameter information
		0, // Task parameter number
		TRUE, // Statically scheduled or not
		0,	  // Throughput constraint
		NULL, // Mutex
		NULL, // Conditional variable
	},
	{ 	8, // Task ID
		"H264Dec_VIDEO_IntraPredY", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_H264Dec_VIDEO_IntraPredY_functions, // Task function array
		1, // Task function array number
		RUN_CONDITION_DATA_DRIVEN, // Run condition
		1, // Run rate
		1, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // Subgraph
		&g_stGraph_H264Dec_VIDEO, // Parent task graph
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
		"H264Dec_VIDEO_IntraPredV", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_H264Dec_VIDEO_IntraPredV_functions, // Task function array
		1, // Task function array number
		RUN_CONDITION_DATA_DRIVEN, // Run condition
		1, // Run rate
		1, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // Subgraph
		&g_stGraph_H264Dec_VIDEO, // Parent task graph
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
		"H264Dec_VIDEO_Deblock", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_H264Dec_VIDEO_Deblock_functions, // Task function array
		1, // Task function array number
		RUN_CONDITION_DATA_DRIVEN, // Run condition
		1, // Run rate
		1, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // Subgraph
		&g_stGraph_H264Dec_VIDEO, // Parent task graph
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
		"H264Dec_VIDEO_WriteFileH", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_H264Dec_VIDEO_WriteFileH_functions, // Task function array
		1, // Task function array number
		RUN_CONDITION_DATA_DRIVEN, // Run condition
		1, // Run rate
		1, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // Subgraph
		&g_stGraph_H264Dec_VIDEO, // Parent task graph
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

STask g_astTasks_x264Enc[] = {
	{ 	16, // Task ID
		"x264Enc_VLC_4", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_x264Enc_VLC_4_functions, // Task function array
		1, // Task function array number
		RUN_CONDITION_TIME_DRIVEN, // Run condition
		1, // Run rate
		10, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // Subgraph
		&g_stGraph_x264Enc, // Parent task graph
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
		"x264Enc_ME_0", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_x264Enc_ME_0_functions, // Task function array
		1, // Task function array number
		RUN_CONDITION_TIME_DRIVEN, // Run condition
		1, // Run rate
		10, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // Subgraph
		&g_stGraph_x264Enc, // Parent task graph
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
		"x264Enc_Encoder_1", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_x264Enc_Encoder_1_functions, // Task function array
		1, // Task function array number
		RUN_CONDITION_TIME_DRIVEN, // Run condition
		1, // Run rate
		10, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // Subgraph
		&g_stGraph_x264Enc, // Parent task graph
		NULL, // MTM information
		NULL, // Loop information
		NULL, // Task parameter information
		0, // Task parameter number
		TRUE, // Statically scheduled or not
		0,	  // Throughput constraint
		NULL, // Mutex
		NULL, // Conditional variable
	},
	{ 	14, // Task ID
		"x264Enc_Deblock_2", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_x264Enc_Deblock_2_functions, // Task function array
		1, // Task function array number
		RUN_CONDITION_TIME_DRIVEN, // Run condition
		1, // Run rate
		10, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // Subgraph
		&g_stGraph_x264Enc, // Parent task graph
		NULL, // MTM information
		NULL, // Loop information
		NULL, // Task parameter information
		0, // Task parameter number
		TRUE, // Statically scheduled or not
		0,	  // Throughput constraint
		NULL, // Mutex
		NULL, // Conditional variable
	},
	{ 	15, // Task ID
		"x264Enc_Init_3", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_x264Enc_Init_3_functions, // Task function array
		1, // Task function array number
		RUN_CONDITION_TIME_DRIVEN, // Run condition
		1, // Run rate
		10, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // Subgraph
		&g_stGraph_x264Enc, // Parent task graph
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

STask g_astTasks_H264Dec_PHONE[] = {
	{ 	30, // Task ID
		"H264Dec_PHONE_ReadFileH", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_H264Dec_PHONE_ReadFileH_functions, // Task function array
		1, // Task function array number
		RUN_CONDITION_TIME_DRIVEN, // Run condition
		1, // Run rate
		1, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // Subgraph
		&g_stGraph_H264Dec_PHONE, // Parent task graph
		NULL, // MTM information
		NULL, // Loop information
		NULL, // Task parameter information
		0, // Task parameter number
		TRUE, // Statically scheduled or not
		0,	  // Throughput constraint
		NULL, // Mutex
		NULL, // Conditional variable
	},
	{ 	31, // Task ID
		"H264Dec_PHONE_Decode", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_H264Dec_PHONE_Decode_functions, // Task function array
		1, // Task function array number
		RUN_CONDITION_DATA_DRIVEN, // Run condition
		1, // Run rate
		1, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // Subgraph
		&g_stGraph_H264Dec_PHONE, // Parent task graph
		NULL, // MTM information
		NULL, // Loop information
		NULL, // Task parameter information
		0, // Task parameter number
		TRUE, // Statically scheduled or not
		0,	  // Throughput constraint
		NULL, // Mutex
		NULL, // Conditional variable
	},
	{ 	32, // Task ID
		"H264Dec_PHONE_InterPredY", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_H264Dec_PHONE_InterPredY_functions, // Task function array
		1, // Task function array number
		RUN_CONDITION_DATA_DRIVEN, // Run condition
		1, // Run rate
		1, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // Subgraph
		&g_stGraph_H264Dec_PHONE, // Parent task graph
		NULL, // MTM information
		NULL, // Loop information
		NULL, // Task parameter information
		0, // Task parameter number
		TRUE, // Statically scheduled or not
		0,	  // Throughput constraint
		NULL, // Mutex
		NULL, // Conditional variable
	},
	{ 	37, // Task ID
		"H264Dec_PHONE_IntraPredY", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_H264Dec_PHONE_IntraPredY_functions, // Task function array
		1, // Task function array number
		RUN_CONDITION_DATA_DRIVEN, // Run condition
		1, // Run rate
		1, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // Subgraph
		&g_stGraph_H264Dec_PHONE, // Parent task graph
		NULL, // MTM information
		NULL, // Loop information
		NULL, // Task parameter information
		0, // Task parameter number
		TRUE, // Statically scheduled or not
		0,	  // Throughput constraint
		NULL, // Mutex
		NULL, // Conditional variable
	},
	{ 	33, // Task ID
		"H264Dec_PHONE_InterPredU", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_H264Dec_PHONE_InterPredU_functions, // Task function array
		1, // Task function array number
		RUN_CONDITION_DATA_DRIVEN, // Run condition
		1, // Run rate
		1, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // Subgraph
		&g_stGraph_H264Dec_PHONE, // Parent task graph
		NULL, // MTM information
		NULL, // Loop information
		NULL, // Task parameter information
		0, // Task parameter number
		TRUE, // Statically scheduled or not
		0,	  // Throughput constraint
		NULL, // Mutex
		NULL, // Conditional variable
	},
	{ 	38, // Task ID
		"H264Dec_PHONE_IntraPredU", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_H264Dec_PHONE_IntraPredU_functions, // Task function array
		1, // Task function array number
		RUN_CONDITION_DATA_DRIVEN, // Run condition
		1, // Run rate
		1, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // Subgraph
		&g_stGraph_H264Dec_PHONE, // Parent task graph
		NULL, // MTM information
		NULL, // Loop information
		NULL, // Task parameter information
		0, // Task parameter number
		TRUE, // Statically scheduled or not
		0,	  // Throughput constraint
		NULL, // Mutex
		NULL, // Conditional variable
	},
	{ 	34, // Task ID
		"H264Dec_PHONE_InterPredV", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_H264Dec_PHONE_InterPredV_functions, // Task function array
		1, // Task function array number
		RUN_CONDITION_DATA_DRIVEN, // Run condition
		1, // Run rate
		1, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // Subgraph
		&g_stGraph_H264Dec_PHONE, // Parent task graph
		NULL, // MTM information
		NULL, // Loop information
		NULL, // Task parameter information
		0, // Task parameter number
		TRUE, // Statically scheduled or not
		0,	  // Throughput constraint
		NULL, // Mutex
		NULL, // Conditional variable
	},
	{ 	39, // Task ID
		"H264Dec_PHONE_IntraPredV", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_H264Dec_PHONE_IntraPredV_functions, // Task function array
		1, // Task function array number
		RUN_CONDITION_DATA_DRIVEN, // Run condition
		1, // Run rate
		1, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // Subgraph
		&g_stGraph_H264Dec_PHONE, // Parent task graph
		NULL, // MTM information
		NULL, // Loop information
		NULL, // Task parameter information
		0, // Task parameter number
		TRUE, // Statically scheduled or not
		0,	  // Throughput constraint
		NULL, // Mutex
		NULL, // Conditional variable
	},
	{ 	35, // Task ID
		"H264Dec_PHONE_Deblock", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_H264Dec_PHONE_Deblock_functions, // Task function array
		1, // Task function array number
		RUN_CONDITION_DATA_DRIVEN, // Run condition
		1, // Run rate
		1, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // Subgraph
		&g_stGraph_H264Dec_PHONE, // Parent task graph
		NULL, // MTM information
		NULL, // Loop information
		NULL, // Task parameter information
		0, // Task parameter number
		TRUE, // Statically scheduled or not
		0,	  // Throughput constraint
		NULL, // Mutex
		NULL, // Conditional variable
	},
	{ 	36, // Task ID
		"H264Dec_PHONE_WriteFileH", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		g_ast_H264Dec_PHONE_WriteFileH_functions, // Task function array
		1, // Task function array number
		RUN_CONDITION_DATA_DRIVEN, // Run condition
		1, // Run rate
		1, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // Subgraph
		&g_stGraph_H264Dec_PHONE, // Parent task graph
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
STaskGraph g_stGraph_top = {
		GRAPH_TYPE_PROCESS_NETWORK, // TODO: Task graph type (not used now)
		g_astTasks_top, // current task graph's task list
		9, // number of tasks
		NULL, // parent task
};

STaskGraph g_stGraph_MP3Dec = {
		GRAPH_TYPE_PROCESS_NETWORK, // TODO: Task graph type (not used now)
		g_astTasks_MP3Dec, // current task graph's task list
		8, // number of tasks
		&g_astTasks_top[2], // parent task
};

STaskGraph g_stGraph_H264Dec_VIDEO = {
		GRAPH_TYPE_PROCESS_NETWORK, // TODO: Task graph type (not used now)
		g_astTasks_H264Dec_VIDEO, // current task graph's task list
		10, // number of tasks
		&g_astTasks_top[1], // parent task
};

STaskGraph g_stGraph_x264Enc = {
		GRAPH_TYPE_PROCESS_NETWORK, // TODO: Task graph type (not used now)
		g_astTasks_x264Enc, // current task graph's task list
		5, // number of tasks
		&g_astTasks_top[3], // parent task
};

STaskGraph g_stGraph_H264Dec_PHONE = {
		GRAPH_TYPE_PROCESS_NETWORK, // TODO: Task graph type (not used now)
		g_astTasks_H264Dec_PHONE, // current task graph's task list
		10, // number of tasks
		&g_astTasks_top[0], // parent task
};

// ##TASK_GRAPH_TEMPLATE::END

// ##TASK_ID_TO_TASK_MAP_TEMPLATE::START
STaskIdToTaskMap g_astTaskIdToTask[] = {
	{ 	15, // Task ID
		"x264Enc_Init_3", // Task name
		&g_astTasks_x264Enc[4], // Task structure pointer
	},
	{ 	2, // Task ID
		"H264Dec_VIDEO_Decode", // Task name
		&g_astTasks_H264Dec_VIDEO[1], // Task structure pointer
	},
	{ 	27, // Task ID
		"G723Enc", // Task name
		&g_astTasks_top[5], // Task structure pointer
	},
	{ 	14, // Task ID
		"x264Enc_Deblock_2", // Task name
		&g_astTasks_x264Enc[3], // Task structure pointer
	},
	{ 	31, // Task ID
		"H264Dec_PHONE_Decode", // Task name
		&g_astTasks_H264Dec_PHONE[1], // Task structure pointer
	},
	{ 	11, // Task ID
		"x264Enc", // Task name
		&g_astTasks_top[3], // Task structure pointer
	},
	{ 	18, // Task ID
		"MP3Dec_VLDStream", // Task name
		&g_astTasks_MP3Dec[6], // Task structure pointer
	},
	{ 	13, // Task ID
		"x264Enc_Encoder_1", // Task name
		&g_astTasks_x264Enc[2], // Task structure pointer
	},
	{ 	41, // Task ID
		"Control", // Task name
		&g_astTasks_top[8], // Task structure pointer
	},
	{ 	8, // Task ID
		"H264Dec_VIDEO_IntraPredY", // Task name
		&g_astTasks_H264Dec_VIDEO[6], // Task structure pointer
	},
	{ 	17, // Task ID
		"MP3Dec", // Task name
		&g_astTasks_top[2], // Task structure pointer
	},
	{ 	10, // Task ID
		"H264Dec_VIDEO_IntraPredV", // Task name
		&g_astTasks_H264Dec_VIDEO[7], // Task structure pointer
	},
	{ 	7, // Task ID
		"H264Dec_VIDEO_WriteFileH", // Task name
		&g_astTasks_H264Dec_VIDEO[9], // Task structure pointer
	},
	{ 	22, // Task ID
		"MP3Dec_Antialias", // Task name
		&g_astTasks_MP3Dec[0], // Task structure pointer
	},
	{ 	24, // Task ID
		"MP3Dec_Subband", // Task name
		&g_astTasks_MP3Dec[1], // Task structure pointer
	},
	{ 	32, // Task ID
		"H264Dec_PHONE_InterPredY", // Task name
		&g_astTasks_H264Dec_PHONE[2], // Task structure pointer
	},
	{ 	6, // Task ID
		"H264Dec_VIDEO_Deblock", // Task name
		&g_astTasks_H264Dec_VIDEO[8], // Task structure pointer
	},
	{ 	12, // Task ID
		"x264Enc_ME_0", // Task name
		&g_astTasks_x264Enc[1], // Task structure pointer
	},
	{ 	4, // Task ID
		"H264Dec_VIDEO_InterPredU", // Task name
		&g_astTasks_H264Dec_VIDEO[4], // Task structure pointer
	},
	{ 	9, // Task ID
		"H264Dec_VIDEO_IntraPredU", // Task name
		&g_astTasks_H264Dec_VIDEO[5], // Task structure pointer
	},
	{ 	28, // Task ID
		"UserInput", // Task name
		&g_astTasks_top[6], // Task structure pointer
	},
	{ 	0, // Task ID
		"H264Dec_VIDEO", // Task name
		&g_astTasks_top[1], // Task structure pointer
	},
	{ 	35, // Task ID
		"H264Dec_PHONE_Deblock", // Task name
		&g_astTasks_H264Dec_PHONE[8], // Task structure pointer
	},
	{ 	5, // Task ID
		"H264Dec_VIDEO_InterPredV", // Task name
		&g_astTasks_H264Dec_VIDEO[3], // Task structure pointer
	},
	{ 	30, // Task ID
		"H264Dec_PHONE_ReadFileH", // Task name
		&g_astTasks_H264Dec_PHONE[0], // Task structure pointer
	},
	{ 	3, // Task ID
		"H264Dec_VIDEO_InterPredY", // Task name
		&g_astTasks_H264Dec_VIDEO[2], // Task structure pointer
	},
	{ 	25, // Task ID
		"MP3Dec_WriteFileM", // Task name
		&g_astTasks_MP3Dec[5], // Task structure pointer
	},
	{ 	29, // Task ID
		"H264Dec_PHONE", // Task name
		&g_astTasks_top[0], // Task structure pointer
	},
	{ 	39, // Task ID
		"H264Dec_PHONE_IntraPredV", // Task name
		&g_astTasks_H264Dec_PHONE[7], // Task structure pointer
	},
	{ 	40, // Task ID
		"Interrupt", // Task name
		&g_astTasks_top[7], // Task structure pointer
	},
	{ 	38, // Task ID
		"H264Dec_PHONE_IntraPredU", // Task name
		&g_astTasks_H264Dec_PHONE[5], // Task structure pointer
	},
	{ 	1, // Task ID
		"H264Dec_VIDEO_ReadFileH", // Task name
		&g_astTasks_H264Dec_VIDEO[0], // Task structure pointer
	},
	{ 	37, // Task ID
		"H264Dec_PHONE_IntraPredY", // Task name
		&g_astTasks_H264Dec_PHONE[3], // Task structure pointer
	},
	{ 	20, // Task ID
		"MP3Dec_Stereo", // Task name
		&g_astTasks_MP3Dec[3], // Task structure pointer
	},
	{ 	19, // Task ID
		"MP3Dec_DeQ", // Task name
		&g_astTasks_MP3Dec[2], // Task structure pointer
	},
	{ 	36, // Task ID
		"H264Dec_PHONE_WriteFileH", // Task name
		&g_astTasks_H264Dec_PHONE[9], // Task structure pointer
	},
	{ 	33, // Task ID
		"H264Dec_PHONE_InterPredU", // Task name
		&g_astTasks_H264Dec_PHONE[4], // Task structure pointer
	},
	{ 	34, // Task ID
		"H264Dec_PHONE_InterPredV", // Task name
		&g_astTasks_H264Dec_PHONE[6], // Task structure pointer
	},
	{ 	21, // Task ID
		"MP3Dec_Reorder", // Task name
		&g_astTasks_MP3Dec[7], // Task structure pointer
	},
	{ 	16, // Task ID
		"x264Enc_VLC_4", // Task name
		&g_astTasks_x264Enc[0], // Task structure pointer
	},
	{ 	26, // Task ID
		"G723Dec", // Task name
		&g_astTasks_top[4], // Task structure pointer
	},
	{ 	23, // Task ID
		"MP3Dec_Hybrid", // Task name
		&g_astTasks_MP3Dec[4], // Task structure pointer
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
void MP3Dec_0_0_0_200000_Go(int nTaskId) 
{
	MP3Dec_Antialias_Go0(22);//
	MP3Dec_Antialias_Go0(22);//
	MP3Dec_Subband_Go0(24);//
}

void MP3Dec_0_0_0_300000_Go(int nTaskId) 
{
	MP3Dec_VLDStream_Go0(18);//
	MP3Dec_Reorder_Go0(21);//
	MP3Dec_Reorder_Go0(21);//
}

void MP3Dec_0_0_1_200000_Go(int nTaskId) 
{
	MP3Dec_DeQ_Go0(19);//
	MP3Dec_DeQ_Go0(19);//
	MP3Dec_Stereo_Go0(20);//
}

void MP3Dec_0_0_1_300000_Go(int nTaskId) 
{
	MP3Dec_Stereo_Go0(20);//
	MP3Dec_Antialias_Go0(22);//
	MP3Dec_Antialias_Go0(22);//
}

void MP3Dec_0_0_2_200000_Go(int nTaskId) 
{
	MP3Dec_Hybrid_Go0(23);//
	MP3Dec_Hybrid_Go0(23);//
	MP3Dec_WriteFileM_Go0(25);//
}

void MP3Dec_0_0_2_300000_Go(int nTaskId) 
{
	MP3Dec_DeQ_Go0(19);//
	MP3Dec_DeQ_Go0(19);//
}

void MP3Dec_0_0_3_200000_Go(int nTaskId) 
{
	MP3Dec_VLDStream_Go0(18);//
	MP3Dec_Reorder_Go0(21);//
	MP3Dec_Reorder_Go0(21);//
}

void MP3Dec_0_0_3_300000_Go(int nTaskId) 
{
	MP3Dec_Hybrid_Go0(23);//
	MP3Dec_Hybrid_Go0(23);//
	MP3Dec_Subband_Go0(24);//
	MP3Dec_WriteFileM_Go0(25);//
}

void H264Dec_VIDEO_0_0_0_3000_Go(int nTaskId) 
{
	{
		uem_bool bTransition = FALSE;
		EModeState enModeState = MODE_STATE_TRANSITING;
		uem_result result;
		STask *pstTask = NULL;
		
		enModeState = UKModeTransition_GetModeState(nTaskId);
	
		if(enModeState == MODE_STATE_TRANSITING)
		{
			H264Dec_VIDEO_ReadFileH_Go0(1);//printf("H264Dec_VIDEO_ReadFileH_Go0 called-- (Line: %d)\n", __LINE__);
		}
		result = UKTask_GetTaskFromTaskId(nTaskId, &pstTask);
		if(result == ERR_UEM_NOERROR)
		{
			result = UCThreadMutex_Lock(pstTask->hMutex);
			if(result == ERR_UEM_NOERROR){
				bTransition = transitMode_H264Dec_VIDEO(g_astTasks_top[1].pstMTMInfo);
				UCThreadMutex_Unlock(pstTask->hMutex);
			}
			
			if(enModeState == MODE_STATE_TRANSITING) return; // exit when the transition is changed.
		}
	}
	{
		EInternalTaskState enState = INTERNAL_STATE_STOP;
		UKTask_GetTaskState("H264Dec_VIDEO", &enState);
		if(enState == INTERNAL_STATE_STOP || enState == INTERNAL_STATE_END) return; 
	}
	H264Dec_VIDEO_Decode_Go0(2);//printf("H264Dec_VIDEO_Decode_Go0 called (Line: %d)\n", __LINE__);
	H264Dec_VIDEO_InterPredY_Go0(3);//printf("H264Dec_VIDEO_InterPredY_Go0 called (Line: %d)\n", __LINE__);
	H264Dec_VIDEO_InterPredV_Go0(5);//printf("H264Dec_VIDEO_InterPredV_Go0 called (Line: %d)\n", __LINE__);
}

void H264Dec_VIDEO_0_0_1_3000_Go(int nTaskId) 
{
	H264Dec_VIDEO_InterPredU_Go0(4);//printf("H264Dec_VIDEO_InterPredU_Go0 called (Line: %d)\n", __LINE__);
	H264Dec_VIDEO_IntraPredU_Go0(9);//printf("H264Dec_VIDEO_IntraPredU_Go0 called (Line: %d)\n", __LINE__);
}

void H264Dec_VIDEO_0_0_2_3000_Go(int nTaskId) 
{
	H264Dec_VIDEO_IntraPredY_Go0(8);//printf("H264Dec_VIDEO_IntraPredY_Go0 called (Line: %d)\n", __LINE__);
	H264Dec_VIDEO_IntraPredV_Go0(10);//printf("H264Dec_VIDEO_IntraPredV_Go0 called (Line: %d)\n", __LINE__);
	H264Dec_VIDEO_Deblock_Go0(6);//printf("H264Dec_VIDEO_Deblock_Go0 called (Line: %d)\n", __LINE__);
	H264Dec_VIDEO_WriteFileH_Go0(7);//printf("H264Dec_VIDEO_WriteFileH_Go0 called (Line: %d)\n", __LINE__);
}

void H264Dec_VIDEO_1_0_0_3000_Go(int nTaskId) 
{
	{
		uem_bool bTransition = FALSE;
		EModeState enModeState = MODE_STATE_TRANSITING;
		uem_result result;
		STask *pstTask = NULL;
		
		enModeState = UKModeTransition_GetModeState(nTaskId);
	
		if(enModeState == MODE_STATE_TRANSITING)
		{
			H264Dec_VIDEO_ReadFileH_Go0(1);//printf("H264Dec_VIDEO_ReadFileH_Go0 called-- (Line: %d)\n", __LINE__);
		}
		result = UKTask_GetTaskFromTaskId(nTaskId, &pstTask);
		if(result == ERR_UEM_NOERROR)
		{
			result = UCThreadMutex_Lock(pstTask->hMutex);
			if(result == ERR_UEM_NOERROR){
				bTransition = transitMode_H264Dec_VIDEO(g_astTasks_top[1].pstMTMInfo);
				UCThreadMutex_Unlock(pstTask->hMutex);
			}
			
			if(enModeState == MODE_STATE_TRANSITING) return; // exit when the transition is changed.
		}
	}
	{
		EInternalTaskState enState = INTERNAL_STATE_STOP;
		UKTask_GetTaskState("H264Dec_VIDEO", &enState);
		if(enState == INTERNAL_STATE_STOP || enState == INTERNAL_STATE_END) return; 
	}
	H264Dec_VIDEO_Decode_Go0(2);//printf("H264Dec_VIDEO_Decode_Go0 called (Line: %d)\n", __LINE__);
	H264Dec_VIDEO_InterPredY_Go0(3);//printf("H264Dec_VIDEO_InterPredY_Go0 called (Line: %d)\n", __LINE__);
}

void H264Dec_VIDEO_1_0_1_3000_Go(int nTaskId) 
{
	H264Dec_VIDEO_InterPredU_Go0(4);//printf("H264Dec_VIDEO_InterPredU_Go0 called (Line: %d)\n", __LINE__);
}

void H264Dec_VIDEO_1_0_2_3000_Go(int nTaskId) 
{
	H264Dec_VIDEO_InterPredV_Go0(5);//printf("H264Dec_VIDEO_InterPredV_Go0 called (Line: %d)\n", __LINE__);
	H264Dec_VIDEO_Deblock_Go0(6);//printf("H264Dec_VIDEO_Deblock_Go0 called (Line: %d)\n", __LINE__);
	H264Dec_VIDEO_WriteFileH_Go0(7);//printf("H264Dec_VIDEO_WriteFileH_Go0 called (Line: %d)\n", __LINE__);
}

void x264Enc_0_0_0_110000_Go(int nTaskId) 
{
	x264Enc_VLC_4_Go0(16);//
}

void x264Enc_0_0_2_110000_Go(int nTaskId) 
{
	x264Enc_ME_0_Go0(12);//
	x264Enc_Encoder_1_Go0(13);//
	x264Enc_Deblock_2_Go0(14);//
}

void x264Enc_0_0_3_110000_Go(int nTaskId) 
{
	x264Enc_Init_3_Go0(15);//
}

void H264Dec_PHONE_0_0_1_2500_Go(int nTaskId) 
{
	{
		uem_bool bTransition = FALSE;
		EModeState enModeState = MODE_STATE_TRANSITING;
		uem_result result;
		STask *pstTask = NULL;
		
		enModeState = UKModeTransition_GetModeState(nTaskId);
	
		if(enModeState == MODE_STATE_TRANSITING)
		{
			H264Dec_PHONE_ReadFileH_Go0(30);//printf("H264Dec_PHONE_ReadFileH_Go0 called-- (Line: %d)\n", __LINE__);
		}
		result = UKTask_GetTaskFromTaskId(nTaskId, &pstTask);
		if(result == ERR_UEM_NOERROR)
		{
			result = UCThreadMutex_Lock(pstTask->hMutex);
			if(result == ERR_UEM_NOERROR){
				bTransition = transitMode_H264Dec_PHONE(g_astTasks_top[0].pstMTMInfo);
				UCThreadMutex_Unlock(pstTask->hMutex);
			}
			
			if(enModeState == MODE_STATE_TRANSITING) return; // exit when the transition is changed.
		}
	}
	{
		EInternalTaskState enState = INTERNAL_STATE_STOP;
		UKTask_GetTaskState("H264Dec_PHONE", &enState);
		if(enState == INTERNAL_STATE_STOP || enState == INTERNAL_STATE_END) return; 
	}
	H264Dec_PHONE_Decode_Go0(31);//printf("H264Dec_PHONE_Decode_Go0 called (Line: %d)\n", __LINE__);
	H264Dec_PHONE_InterPredY_Go0(32);//printf("H264Dec_PHONE_InterPredY_Go0 called (Line: %d)\n", __LINE__);
	H264Dec_PHONE_IntraPredY_Go0(37);//printf("H264Dec_PHONE_IntraPredY_Go0 called (Line: %d)\n", __LINE__);
}

void H264Dec_PHONE_0_0_2_2500_Go(int nTaskId) 
{
	H264Dec_PHONE_InterPredU_Go0(33);//printf("H264Dec_PHONE_InterPredU_Go0 called (Line: %d)\n", __LINE__);
	H264Dec_PHONE_IntraPredU_Go0(38);//printf("H264Dec_PHONE_IntraPredU_Go0 called (Line: %d)\n", __LINE__);
}

void H264Dec_PHONE_0_0_3_2500_Go(int nTaskId) 
{
	H264Dec_PHONE_InterPredV_Go0(34);//printf("H264Dec_PHONE_InterPredV_Go0 called (Line: %d)\n", __LINE__);
	H264Dec_PHONE_IntraPredV_Go0(39);//printf("H264Dec_PHONE_IntraPredV_Go0 called (Line: %d)\n", __LINE__);
	H264Dec_PHONE_Deblock_Go0(35);//printf("H264Dec_PHONE_Deblock_Go0 called (Line: %d)\n", __LINE__);
	H264Dec_PHONE_WriteFileH_Go0(36);//printf("H264Dec_PHONE_WriteFileH_Go0 called (Line: %d)\n", __LINE__);
}

void H264Dec_PHONE_1_0_1_2500_Go(int nTaskId) 
{
	{
		uem_bool bTransition = FALSE;
		EModeState enModeState = MODE_STATE_TRANSITING;
		uem_result result;
		STask *pstTask = NULL;
		
		enModeState = UKModeTransition_GetModeState(nTaskId);
	
		if(enModeState == MODE_STATE_TRANSITING)
		{
			H264Dec_PHONE_ReadFileH_Go0(30);//printf("H264Dec_PHONE_ReadFileH_Go0 called-- (Line: %d)\n", __LINE__);
		}
		result = UKTask_GetTaskFromTaskId(nTaskId, &pstTask);
		if(result == ERR_UEM_NOERROR)
		{
			result = UCThreadMutex_Lock(pstTask->hMutex);
			if(result == ERR_UEM_NOERROR){
				bTransition = transitMode_H264Dec_PHONE(g_astTasks_top[0].pstMTMInfo);
				UCThreadMutex_Unlock(pstTask->hMutex);
			}
			
			if(enModeState == MODE_STATE_TRANSITING) return; // exit when the transition is changed.
		}
	}
	{
		EInternalTaskState enState = INTERNAL_STATE_STOP;
		UKTask_GetTaskState("H264Dec_PHONE", &enState);
		if(enState == INTERNAL_STATE_STOP || enState == INTERNAL_STATE_END) return; 
	}
	H264Dec_PHONE_InterPredU_Go0(33);//printf("H264Dec_PHONE_InterPredU_Go0 called (Line: %d)\n", __LINE__);
}

void H264Dec_PHONE_1_0_2_2500_Go(int nTaskId) 
{
	H264Dec_PHONE_InterPredY_Go0(32);//printf("H264Dec_PHONE_InterPredY_Go0 called (Line: %d)\n", __LINE__);
}

void H264Dec_PHONE_1_0_3_2500_Go(int nTaskId) 
{
	H264Dec_PHONE_Decode_Go0(31);//printf("H264Dec_PHONE_Decode_Go0 called (Line: %d)\n", __LINE__);
	H264Dec_PHONE_InterPredV_Go0(34);//printf("H264Dec_PHONE_InterPredV_Go0 called (Line: %d)\n", __LINE__);
	H264Dec_PHONE_Deblock_Go0(35);//printf("H264Dec_PHONE_Deblock_Go0 called (Line: %d)\n", __LINE__);
	H264Dec_PHONE_WriteFileH_Go0(36);//printf("H264Dec_PHONE_WriteFileH_Go0 called (Line: %d)\n", __LINE__);
}

// ##SCHEDULED_COMPOSITE_TASK_FUNCTION_IMPLEMENTATION::END

// ##SCHEDULED_COMPOSITE_TASKS_TEMPLATE::START
SScheduleList g_astScheduleList_MP3Dec_0_0_0[] = {
	{
		MP3Dec_0_0_0_200000_Go, // Composite GO function
		200000, // Throughput constraint
		FALSE,
	},
	{
		MP3Dec_0_0_0_300000_Go, // Composite GO function
		300000, // Throughput constraint
		FALSE,
	},
};
SScheduleList g_astScheduleList_MP3Dec_0_0_1[] = {
	{
		MP3Dec_0_0_1_200000_Go, // Composite GO function
		200000, // Throughput constraint
		FALSE,
	},
	{
		MP3Dec_0_0_1_300000_Go, // Composite GO function
		300000, // Throughput constraint
		FALSE,
	},
};
SScheduleList g_astScheduleList_MP3Dec_0_0_2[] = {
	{
		MP3Dec_0_0_2_200000_Go, // Composite GO function
		200000, // Throughput constraint
		FALSE,
	},
	{
		MP3Dec_0_0_2_300000_Go, // Composite GO function
		300000, // Throughput constraint
		FALSE,
	},
};
SScheduleList g_astScheduleList_MP3Dec_0_0_3[] = {
	{
		MP3Dec_0_0_3_200000_Go, // Composite GO function
		200000, // Throughput constraint
		FALSE,
	},
	{
		MP3Dec_0_0_3_300000_Go, // Composite GO function
		300000, // Throughput constraint
		FALSE,
	},
};
SScheduleList g_astScheduleList_H264Dec_VIDEO_0_0_0[] = {
	{
		H264Dec_VIDEO_0_0_0_3000_Go, // Composite GO function
		3000, // Throughput constraint
		TRUE,
	},
};
SScheduleList g_astScheduleList_H264Dec_VIDEO_0_0_1[] = {
	{
		H264Dec_VIDEO_0_0_1_3000_Go, // Composite GO function
		3000, // Throughput constraint
		FALSE,
	},
};
SScheduleList g_astScheduleList_H264Dec_VIDEO_0_0_2[] = {
	{
		H264Dec_VIDEO_0_0_2_3000_Go, // Composite GO function
		3000, // Throughput constraint
		FALSE,
	},
};
SScheduleList g_astScheduleList_H264Dec_VIDEO_1_0_0[] = {
	{
		H264Dec_VIDEO_1_0_0_3000_Go, // Composite GO function
		3000, // Throughput constraint
		TRUE,
	},
};
SScheduleList g_astScheduleList_H264Dec_VIDEO_1_0_1[] = {
	{
		H264Dec_VIDEO_1_0_1_3000_Go, // Composite GO function
		3000, // Throughput constraint
		FALSE,
	},
};
SScheduleList g_astScheduleList_H264Dec_VIDEO_1_0_2[] = {
	{
		H264Dec_VIDEO_1_0_2_3000_Go, // Composite GO function
		3000, // Throughput constraint
		FALSE,
	},
};
SScheduleList g_astScheduleList_x264Enc_0_0_0[] = {
	{
		x264Enc_0_0_0_110000_Go, // Composite GO function
		110000, // Throughput constraint
		FALSE,
	},
};
SScheduleList g_astScheduleList_x264Enc_0_0_2[] = {
	{
		x264Enc_0_0_2_110000_Go, // Composite GO function
		110000, // Throughput constraint
		FALSE,
	},
};
SScheduleList g_astScheduleList_x264Enc_0_0_3[] = {
	{
		x264Enc_0_0_3_110000_Go, // Composite GO function
		110000, // Throughput constraint
		FALSE,
	},
};
SScheduleList g_astScheduleList_H264Dec_PHONE_0_0_1[] = {
	{
		H264Dec_PHONE_0_0_1_2500_Go, // Composite GO function
		2500, // Throughput constraint
		TRUE,
	},
};
SScheduleList g_astScheduleList_H264Dec_PHONE_0_0_2[] = {
	{
		H264Dec_PHONE_0_0_2_2500_Go, // Composite GO function
		2500, // Throughput constraint
		FALSE,
	},
};
SScheduleList g_astScheduleList_H264Dec_PHONE_0_0_3[] = {
	{
		H264Dec_PHONE_0_0_3_2500_Go, // Composite GO function
		2500, // Throughput constraint
		FALSE,
	},
};
SScheduleList g_astScheduleList_H264Dec_PHONE_1_0_1[] = {
	{
		H264Dec_PHONE_1_0_1_2500_Go, // Composite GO function
		2500, // Throughput constraint
		TRUE,
	},
};
SScheduleList g_astScheduleList_H264Dec_PHONE_1_0_2[] = {
	{
		H264Dec_PHONE_1_0_2_2500_Go, // Composite GO function
		2500, // Throughput constraint
		FALSE,
	},
};
SScheduleList g_astScheduleList_H264Dec_PHONE_1_0_3[] = {
	{
		H264Dec_PHONE_1_0_3_2500_Go, // Composite GO function
		2500, // Throughput constraint
		FALSE,
	},
};
// ##SCHEDULED_COMPOSITE_TASKS_TEMPLATE::END



SScheduledTasks g_astScheduledTaskList[] = {
	{	&g_astTasks_top[2], // Parent Task ID
		0, // Mode transition mode ID
		g_astScheduleList_MP3Dec_0_0_0, // schedule list per throughput constraint
		2, // The number of schedules in the schedule list
		0, // Schedule Index (Default to set 0)
		0, // Mode Sequence ID 
	},
	{	&g_astTasks_top[2], // Parent Task ID
		0, // Mode transition mode ID
		g_astScheduleList_MP3Dec_0_0_1, // schedule list per throughput constraint
		2, // The number of schedules in the schedule list
		0, // Schedule Index (Default to set 0)
		1, // Mode Sequence ID 
	},
	{	&g_astTasks_top[2], // Parent Task ID
		0, // Mode transition mode ID
		g_astScheduleList_MP3Dec_0_0_2, // schedule list per throughput constraint
		2, // The number of schedules in the schedule list
		0, // Schedule Index (Default to set 0)
		2, // Mode Sequence ID 
	},
	{	&g_astTasks_top[2], // Parent Task ID
		0, // Mode transition mode ID
		g_astScheduleList_MP3Dec_0_0_3, // schedule list per throughput constraint
		2, // The number of schedules in the schedule list
		0, // Schedule Index (Default to set 0)
		3, // Mode Sequence ID 
	},
	{	&g_astTasks_top[1], // Parent Task ID
		0, // Mode transition mode ID
		g_astScheduleList_H264Dec_VIDEO_0_0_0, // schedule list per throughput constraint
		1, // The number of schedules in the schedule list
		0, // Schedule Index (Default to set 0)
		0, // Mode Sequence ID 
	},
	{	&g_astTasks_top[1], // Parent Task ID
		0, // Mode transition mode ID
		g_astScheduleList_H264Dec_VIDEO_0_0_1, // schedule list per throughput constraint
		1, // The number of schedules in the schedule list
		0, // Schedule Index (Default to set 0)
		1, // Mode Sequence ID 
	},
	{	&g_astTasks_top[1], // Parent Task ID
		0, // Mode transition mode ID
		g_astScheduleList_H264Dec_VIDEO_0_0_2, // schedule list per throughput constraint
		1, // The number of schedules in the schedule list
		0, // Schedule Index (Default to set 0)
		2, // Mode Sequence ID 
	},
	{	&g_astTasks_top[1], // Parent Task ID
		1, // Mode transition mode ID
		g_astScheduleList_H264Dec_VIDEO_1_0_0, // schedule list per throughput constraint
		1, // The number of schedules in the schedule list
		0, // Schedule Index (Default to set 0)
		0, // Mode Sequence ID 
	},
	{	&g_astTasks_top[1], // Parent Task ID
		1, // Mode transition mode ID
		g_astScheduleList_H264Dec_VIDEO_1_0_1, // schedule list per throughput constraint
		1, // The number of schedules in the schedule list
		0, // Schedule Index (Default to set 0)
		1, // Mode Sequence ID 
	},
	{	&g_astTasks_top[1], // Parent Task ID
		1, // Mode transition mode ID
		g_astScheduleList_H264Dec_VIDEO_1_0_2, // schedule list per throughput constraint
		1, // The number of schedules in the schedule list
		0, // Schedule Index (Default to set 0)
		2, // Mode Sequence ID 
	},
	{	&g_astTasks_top[3], // Parent Task ID
		0, // Mode transition mode ID
		g_astScheduleList_x264Enc_0_0_0, // schedule list per throughput constraint
		1, // The number of schedules in the schedule list
		0, // Schedule Index (Default to set 0)
		0, // Mode Sequence ID 
	},
	{	&g_astTasks_top[3], // Parent Task ID
		0, // Mode transition mode ID
		g_astScheduleList_x264Enc_0_0_2, // schedule list per throughput constraint
		1, // The number of schedules in the schedule list
		0, // Schedule Index (Default to set 0)
		1, // Mode Sequence ID 
	},
	{	&g_astTasks_top[3], // Parent Task ID
		0, // Mode transition mode ID
		g_astScheduleList_x264Enc_0_0_3, // schedule list per throughput constraint
		1, // The number of schedules in the schedule list
		0, // Schedule Index (Default to set 0)
		2, // Mode Sequence ID 
	},
	{	&g_astTasks_top[0], // Parent Task ID
		0, // Mode transition mode ID
		g_astScheduleList_H264Dec_PHONE_0_0_1, // schedule list per throughput constraint
		1, // The number of schedules in the schedule list
		0, // Schedule Index (Default to set 0)
		0, // Mode Sequence ID 
	},
	{	&g_astTasks_top[0], // Parent Task ID
		0, // Mode transition mode ID
		g_astScheduleList_H264Dec_PHONE_0_0_2, // schedule list per throughput constraint
		1, // The number of schedules in the schedule list
		0, // Schedule Index (Default to set 0)
		1, // Mode Sequence ID 
	},
	{	&g_astTasks_top[0], // Parent Task ID
		0, // Mode transition mode ID
		g_astScheduleList_H264Dec_PHONE_0_0_3, // schedule list per throughput constraint
		1, // The number of schedules in the schedule list
		0, // Schedule Index (Default to set 0)
		2, // Mode Sequence ID 
	},
	{	&g_astTasks_top[0], // Parent Task ID
		1, // Mode transition mode ID
		g_astScheduleList_H264Dec_PHONE_1_0_1, // schedule list per throughput constraint
		1, // The number of schedules in the schedule list
		0, // Schedule Index (Default to set 0)
		0, // Mode Sequence ID 
	},
	{	&g_astTasks_top[0], // Parent Task ID
		1, // Mode transition mode ID
		g_astScheduleList_H264Dec_PHONE_1_0_2, // schedule list per throughput constraint
		1, // The number of schedules in the schedule list
		0, // Schedule Index (Default to set 0)
		1, // Mode Sequence ID 
	},
	{	&g_astTasks_top[0], // Parent Task ID
		1, // Mode transition mode ID
		g_astScheduleList_H264Dec_PHONE_1_0_3, // schedule list per throughput constraint
		1, // The number of schedules in the schedule list
		0, // Schedule Index (Default to set 0)
		2, // Mode Sequence ID 
	},
};


// ##MAPPING_SCHEDULING_INFO_TEMPLATE::START

SMappedGeneralTaskInfo g_astGeneralTaskMappingInfo[] = {
	{	TASK_TYPE_CONTROL, // Task type
		&g_astTasks_top[8], // Task ID or composite task information
		0, // Processor ID
		1, // Processor local ID
	},
	{	TASK_TYPE_COMPUTATIONAL, // Task type
		&g_astTasks_top[5], // Task ID or composite task information
		0, // Processor ID
		0, // Processor local ID
	},
	{	TASK_TYPE_COMPUTATIONAL, // Task type
		&g_astTasks_top[6], // Task ID or composite task information
		0, // Processor ID
		1, // Processor local ID
	},
	{	TASK_TYPE_COMPUTATIONAL, // Task type
		&g_astTasks_top[4], // Task ID or composite task information
		0, // Processor ID
		3, // Processor local ID
	},
	{	TASK_TYPE_COMPUTATIONAL, // Task type
		&g_astTasks_top[7], // Task ID or composite task information
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
		1, // Processor local ID		
	},
	{
		&g_astScheduledTaskList[2],
		0, // Processor ID
		2, // Processor local ID		
	},
	{
		&g_astScheduledTaskList[3],
		0, // Processor ID
		3, // Processor local ID		
	},
	{
		&g_astScheduledTaskList[4],
		0, // Processor ID
		0, // Processor local ID		
	},
	{
		&g_astScheduledTaskList[5],
		0, // Processor ID
		1, // Processor local ID		
	},
	{
		&g_astScheduledTaskList[6],
		0, // Processor ID
		2, // Processor local ID		
	},
	{
		&g_astScheduledTaskList[7],
		0, // Processor ID
		0, // Processor local ID		
	},
	{
		&g_astScheduledTaskList[8],
		0, // Processor ID
		1, // Processor local ID		
	},
	{
		&g_astScheduledTaskList[9],
		0, // Processor ID
		2, // Processor local ID		
	},
	{
		&g_astScheduledTaskList[10],
		0, // Processor ID
		0, // Processor local ID		
	},
	{
		&g_astScheduledTaskList[11],
		0, // Processor ID
		2, // Processor local ID		
	},
	{
		&g_astScheduledTaskList[12],
		0, // Processor ID
		3, // Processor local ID		
	},
	{
		&g_astScheduledTaskList[13],
		0, // Processor ID
		1, // Processor local ID		
	},
	{
		&g_astScheduledTaskList[14],
		0, // Processor ID
		2, // Processor local ID		
	},
	{
		&g_astScheduledTaskList[15],
		0, // Processor ID
		3, // Processor local ID		
	},
	{
		&g_astScheduledTaskList[16],
		0, // Processor ID
		1, // Processor local ID		
	},
	{
		&g_astScheduledTaskList[17],
		0, // Processor ID
		2, // Processor local ID		
	},
	{
		&g_astScheduledTaskList[18],
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
};

// ##LIBRARY_INFO_TEMPLATE::END


int g_nChannelNum = ARRAYLEN(g_astChannels);
int g_nNumOfTasks_top = ARRAYLEN(g_astTasks_top);
int g_nTaskIdToTaskNum = ARRAYLEN(g_astTaskIdToTask);
int g_nProcessorInfoNum = ARRAYLEN(g_astProcessorInfo);
int g_nLibraryInfoNum = 0;

