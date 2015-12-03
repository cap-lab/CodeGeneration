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
* Filename      : bsp_sound.h
* Version       : V1.00
* Programmer(s) : EMO
*********************************************************************************************************
*/

#ifndef __BSP_SOUND_H__
#define __BSP_SOUND_H__
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

#include  <bsp.h>

/*
*********************************************************************************************************
*                                               DEFINES
*********************************************************************************************************
*/

// I2S Pin definitions.
#define I2S0_LRCTX_PERIPH       (SYSCTL_PERIPH_GPIOE)
#define I2S0_LRCTX_PORT         (GPIO_PORTE_BASE)
#define I2S0_LRCTX_PIN          (GPIO_PIN_4)

#define I2S0_SDATX_PERIPH       (SYSCTL_PERIPH_GPIOE)
#define I2S0_SDATX_PORT         (GPIO_PORTE_BASE)
#define I2S0_SDATX_PIN          (GPIO_PIN_5)

#define I2S0_SCLKTX_PERIPH      (SYSCTL_PERIPH_GPIOB)
#define I2S0_SCLKTX_PORT        (GPIO_PORTB_BASE)
#define I2S0_SCLKTX_PIN         (GPIO_PIN_6)

#define I2S0_MCLKTX_PERIPH      (SYSCTL_PERIPH_GPIOF)
#define I2S0_MCLKTX_PORT        (GPIO_PORTF_BASE)
#define I2S0_MCLKTX_PIN         (GPIO_PIN_1)


// The frequencies of the piano keys, for convenience when constructing a song.
// Note that the minimum frequency that can be produced is processor clock
// divided by 65536; at 50 MHz this equates to 763 Hz.  Lower audio frequencies
// are available if the processor clock is lowered, at the cost of lower
// processor performance (most noticable in the screen update speed).
#define SILENCE                 40000
#define A0                      28
#define AS0                     29
#define B0                      31
#define C1                      33
#define CS1                     35
#define D1                      37
#define DS1                     39
#define E1                      41
#define F1                      44
#define FS1                     46
#define G1                      49
#define GS1                     52
#define A1                      55
#define AS1                     58
#define B1                      62
#define C2                      65
#define CS2                     69
#define D2                      73
#define DS2                     78
#define E2                      82
#define F2                      87
#define FS2                     92
#define G2                      98
#define GS2                     104
#define A2                      110
#define AS2                     117
#define B2                      123
#define C3                      131
#define CS3                     139
#define D3                      147
#define DS3                     156
#define E3                      165
#define F3                      175
#define FS3                     185
#define G3                      196
#define GS3                     208
#define A3                      220
#define AS3                     233
#define B3                      247
#define C4                      262
#define CS4                     277
#define D4                      294
#define DS4                     311
#define E4                      330
#define F4                      349
#define FS4                     370
#define G4                      392
#define GS4                     415
#define A4                      440
#define AS4                     466
#define B4                      494
#define C5                      523
#define CS5                     554
#define D5                      587
#define DS5                     622
#define E5                      659
#define F5                      698
#define FS5                     740
#define G5                      784
#define GS5                     831
#define A5                      880
#define AS5                     932
#define B5                      988
#define C6                      1047
#define CS6                     1109
#define D6                      1175
#define DS6                     1245
#define E6                      1319
#define F6                      1397
#define FS6                     1480
#define G6                      1568
#define GS6                     1661
#define A6                      1760
#define AS6                     1865
#define B6                      1976
#define C7                      2093
#define CS7                     2217
#define D7                      2349
#define DS7                     2489
#define E7                      2637
#define F7                      2794
#define FS7                     2960
#define G7                      3136
#define GS7                     3322
#define A7                      3520
#define AS7                     3729
#define B7                      3951
#define C8                      4186


// Enables using uDMA to fill the I2S fifo.
#define BUFFER_EVENT_FREE       0x00000001
#define BUFFER_EVENT_FULL       0x00000002


typedef void (* tBufferCallback)(void *pvBuffer, CPU_INT32U ulEvent);

/*
*********************************************************************************************************
*                                             DATA TYPES
*********************************************************************************************************
*/

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

void        BSP_SoundInit(void);
void        BSP_SoundIntHandler(void);
void        BSP_SoundPlay(const CPU_INT16U *pusSong, CPU_INT32U ulLength);
void        BSP_SoundSetFormat(CPU_INT32U ulSampleRate, CPU_INT16U usBitsPerSample,
                               CPU_INT16U usChannels);
CPU_INT32U  BSP_SoundSampleRateGet(void);
CPU_INT32U  BSP_SoundBufferPlay(const void *pvData, CPU_INT32U ulLength,
                                tBufferCallback pfnCallback);
void        BSP_SoundVolumeSet(CPU_INT32U ulPercent);
CPU_INT08U  BSP_SoundVolumeGet(void);
void        BSP_SoundVolumeDown(CPU_INT32U ulPercent);
void        BSP_SoundVolumeUp(CPU_INT32U ulPercent);

void        BSP_SoundClassDEn(void);
void        BSP_SoundClassDDis(void);



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

#endif // __BSP_SOUND_H__