/*
*********************************************************************************************************
*                                     MICIRUM BOARD SUPPORT PACKAGE
*
*                              (c) Copyright 2010; Micrium, Inc.; Weston, FL
*
*                   All rights reserved.  Protected by international copyright laws.
*                   Knowledge of the source code may not be used to write a similar
*                   product.  This file may only be used in accordance with a license
*                   and should not be redistributed in any way.
*********************************************************************************************************
*/

/*
*********************************************************************************************************
*
*                                        BOARD SUPPORT PACKAGE
*
*                             LUMINARY MICRO LM3S9B92 on the LM3S9B92-EVALBOT
*
* Filename      : bsp_wav.h
* Version       : V1.00
* Programmer(s) : EMO
*********************************************************************************************************
*/

#ifndef __BSP_WAV_H__
#define __BSP_WAV_H__
/*
*********************************************************************************************************
*                                               MODULE
*
* Note(s) : (1) This header file is protected from multiple pre-processor inclusion through use of the
*               BSP present pre-processor macro definition.
*********************************************************************************************************
*/

/*
*********************************************************************************************************
*                                               EXTERNS
*********************************************************************************************************
*/



/*
*********************************************************************************************************
*                                            INCLUDE FILES
*********************************************************************************************************
*/

#include  <includes.h>

/*
*********************************************************************************************************
*                                               DEFINES
*********************************************************************************************************
*/


/*
*********************************************************************************************************
*                                             DATA TYPES
*********************************************************************************************************
*/

// The wav file header information.
typedef struct
{
    // Sample rate in bytes per second.
    CPU_INT32U ulSampleRate;

    // The average byte rate for the wav file.
    CPU_INT32U ulAvgByteRate;
    
    // The size of the wav data in the file.
    CPU_INT32U ulDataSize;

    // The number of bits per sample.
    CPU_INT16U usBitsPerSample;

    // The wav file format.
    CPU_INT16U usFormat;

    // The number of audio channels.
    CPU_INT16U usNumChannels;
}
tWaveHeader;


typedef enum
{
    WAVE_OK = 0,
    WAVE_INVALID_RIFF,
    WAVE_INVALID_CHUNK,
    WAVE_INVALID_FORMAT
}
tWaveReturnCode;


typedef struct
{
    CPU_INT08S  pcFilename[16];
    CPU_INT32U  *pucFilePtr;
}
tWaveFileInfo;
    

/*
*********************************************************************************************************
*                                          GLOBAL VARIABLES
*********************************************************************************************************
*/

/*
*********************************************************************************************************
*                                               MACRO'S
*********************************************************************************************************
*/


/*
*********************************************************************************************************
*                                         FUNCTION PROTOTYPES
*********************************************************************************************************
*/

tWaveReturnCode BSP_WaveOpen(CPU_INT32U *pulAddress, tWaveHeader *pWaveHeader);
void            BSP_WavePlay(tWaveHeader *pWaveHeader);
void            BSP_WaveStop(void);
void            BSP_WaveDisplayTime(tWaveHeader *pWaveHeader, CPU_INT32U ulForceUpdate);
CPU_INT32U      BSP_WavePlaybackStatus(void);

/*
*********************************************************************************************************
*                                        CONFIGURATION ERRORS
*********************************************************************************************************
*/


/*
*********************************************************************************************************
*                                             MODULE END
*********************************************************************************************************
*/

#endif // __BSP_WAV_H__