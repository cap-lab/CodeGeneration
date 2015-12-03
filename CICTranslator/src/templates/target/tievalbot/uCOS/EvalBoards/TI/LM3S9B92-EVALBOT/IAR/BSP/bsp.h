/*
*********************************************************************************************************
*                                     MICIRUM BOARD SUPPORT PACKAGE
*
*                              (c) Copyright 2009; Micrium, Inc.; Weston, FL
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
*                             TEXAS INSTRMENTS EVALBOT with the LM3S9B92
*
* Filename      : bsp.h
* Version       : V1.00
* Programmer(s) : EMO
*********************************************************************************************************
*/


/*
*********************************************************************************************************
*                                               MODULE
*
* Note(s) : (1) This header file is protected from multiple pre-processor inclusion through use of the
*               BSP present pre-processor macro definition.
*********************************************************************************************************
*/

#ifndef  BSP_PRESENT
#define  BSP_PRESENT


/*
*********************************************************************************************************
*                                               EXTERNS
*********************************************************************************************************
*/

#ifdef   BSP_MODULE
#define  BSP_EXT
#else
#define  BSP_EXT  extern
#endif


/*
*********************************************************************************************************
*                                            INCLUDE FILES
*********************************************************************************************************
*/

#include  <cpu.h>
#include  <cpu_core.h>

#include  <lib_def.h>
#include  <lib_mem.h>
#include  <lib_str.h>

#include  <app_cfg.h>

#include  <pin_map.h>

#include  <hw_memmap.h>
#include  <hw_types.h>
#include  <hw_i2c.h>
#include  <hw_ints.h>
#include  <hw_udma.h>
#include  <hw_sysctl.h>
#include  <hw_i2s.h>
#include  <hw_ethernet.h>

#include  <sysctl.h>
#include  <gpio.h>
#include  <interrupt.h>
#include  <pwm.h>
#include  <i2c.h>
#include  <timer.h>
#include  <i2s.h>
#include  <udma.h>
#include  <ethernet.h>

#include  <bsp_int.h>
#include  <bsp_io.h>
#include  <bsp_display.h>
#include  <bsp_dac.h>
#include  <bsp_sound.h>
#include  <bsp_wav.h>

/*
*********************************************************************************************************
*                                               DEFINES
*********************************************************************************************************
*/

typedef  enum {
    LEFT_SIDE = 0,
    RIGHT_SIDE
} tSide;

                                                                /* tSide must be defined before reading in the ...      */
                                                                /* ... motor and sensors header files.                  */
#include  <bsp_motor.h>
#include  <bsp_sensors.h>

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

BSP_EXT  CPU_INT32U  BSP_CPU_ClkFreq_MHz;

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

void        BSP_Init      (void);
CPU_INT32U  BSP_CPUClkFreq(void);

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

#endif                                                          /* End of module include.                               */
