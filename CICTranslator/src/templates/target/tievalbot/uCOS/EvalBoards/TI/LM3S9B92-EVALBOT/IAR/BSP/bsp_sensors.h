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
*                                      ON-BOARD SENSORS SERVICES
*
*                             TEXAS INSTRUMENTS LM3S9B90 on the EK-LM3S9B90
*
* Filename      : bsp_sensors.h
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

#ifndef  BSP_SENSORS_PRESENT
#define  BSP_SENSORS_PRESENT


/*
*********************************************************************************************************
*                                               EXTERNS
*********************************************************************************************************
*/

#ifdef   BSP_SENSORS_MODULE
#define  BSP_SENSORS_EXT
#else
#define  BSP_SENSORS_EXT  extern
#endif


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

#define LEFT_RIGHT_IR_LED_PORT          GPIO_PORTE_BASE
#define LEFT_RIGHT_IR_LED_PIN           GPIO_PIN_6

#define LEFT_IR_SENSOR_A_PORT           GPIO_PORTE_BASE
#define LEFT_IR_SENSOR_A_PIN            GPIO_PIN_3
#define LEFT_IR_SENSOR_B_PORT           GPIO_PORTG_BASE
#define LEFT_IR_SENSOR_B_PIN            GPIO_PIN_7   
#define LEFT_IR_SENSOR_A_INT            BSP_INT_ID_GPIOE
#define LEFT_IR_SENSOR_B_INT            BSP_INT_ID_GPIOG

#define RIGHT_IR_SENSOR_A_PORT          GPIO_PORTE_BASE
#define RIGHT_IR_SENSOR_A_PIN           GPIO_PIN_2
#define RIGHT_IR_SENSOR_B_PORT          GPIO_PORTC_BASE
#define RIGHT_IR_SENSOR_B_PIN           GPIO_PIN_7
#define RIGHT_IR_SENSOR_A_INT           BSP_INT_ID_GPIOE
#define RIGHT_IR_SENSOR_B_INT           BSP_INT_ID_GPIOC

#define NUM_WHEEL_EDGE_PER_REVOL        16

/*
*********************************************************************************************************
*                                             DATA TYPES
*********************************************************************************************************
*/

typedef enum
{
    SENSOR_A = 0,
    SENSOR_B
}
tSensor;

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

/*
*********************************************************************************************************
*                                         BUMP SENSOR SERVICES
*********************************************************************************************************
*/

void         BSP_BumpSensorsInit (void);
CPU_BOOLEAN  BSP_BumpSensorGetStatus(CPU_INT08U  bs);

/*
*********************************************************************************************************
*                                         WHEEL SENSOR CONTROL SERVICES
*********************************************************************************************************
*/

void         BSP_WheelSensorsInit(void);
void         BSP_WheelSensorEnable(void);
void         BSP_WheelSensorDisable(void);
void         BSP_WheelSensorIntEnable(tSide eSide, tSensor eSensor, CPU_FNCT_VOID isr);
void         BSP_WheelSensorIntDisable(tSide eSide, tSensor eSensor);

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