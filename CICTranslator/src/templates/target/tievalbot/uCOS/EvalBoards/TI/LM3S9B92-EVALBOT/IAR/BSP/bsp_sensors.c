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
*                                      ON-BOARD SENSOR SERVICES
*
*                             TEXAS INSTRUMENTS LM3S9B90 on the EK-LM3S9B90
*
* Filename      : bsp_sensors.c
* Version       : V1.00
* Programmer(s) : EMO
*********************************************************************************************************
*/


/*
*********************************************************************************************************
*                                            INCLUDE FILES
*********************************************************************************************************
*/

#define   BSP_SENSORS_MODULE
#include <bsp_sensors.h>


/*
*********************************************************************************************************
*                                            LOCAL DEFINES
*********************************************************************************************************
*/


/*
*********************************************************************************************************
*                                           LOCAL CONSTANTS
*********************************************************************************************************
*/


/*
*********************************************************************************************************
*                                          LOCAL DATA TYPES
*********************************************************************************************************
*/


/*
*********************************************************************************************************
*                                            LOCAL TABLES
*********************************************************************************************************
*/


/*
*********************************************************************************************************
*                                       LOCAL GLOBAL VARIABLES
*********************************************************************************************************
*/


/*
*********************************************************************************************************
*                                      LOCAL FUNCTION PROTOTYPES
*********************************************************************************************************
*/


/*
*********************************************************************************************************
*                                     LOCAL CONFIGURATION ERRORS
*********************************************************************************************************
*/

/*
*********************************************************************************************************
*********************************************************************************************************
*                                            BUMP SENSOR FUNCTIONS
*********************************************************************************************************
*********************************************************************************************************
*/

/*
*********************************************************************************************************
*                                            BSP_BumpSensorsInit()
*
* Description : Initialize the board's bump sensors.
*
* Argument(s) : none.
*
* Return(s)   : none.
*
* Caller(s)   : BSP_Init().
*
* Note(s)     : none.
*********************************************************************************************************
*/

void  BSP_BumpSensorsInit (void)
{
    GPIOPinTypeGPIOInput(GPIO_PORTE_BASE, GPIO_PIN_0 | GPIO_PIN_1);
    GPIOPadConfigSet(GPIO_PORTE_BASE, GPIO_PIN_0 | GPIO_PIN_1, GPIO_STRENGTH_2MA,
                         GPIO_PIN_TYPE_STD_WPU);
}


/*
*********************************************************************************************************
*                                         BSP_BumpSensorGetStatus()
*
* Description : Get the status of a bump sensors on the board.
*
* Argument(s) : bs      The ID of the bump sensors to probe
*
*                       1    probe the right bump sensor (BUMP_R)
*                       2    probe the left bump sensor (BUMP_L)
*
* Return(s)   : DEF_FALSE   if the bump sensor is open.
*               DEF_TRUE    if the bump sensor is closed.
*
* Caller(s)   : Application.
*
* Note(s)     : none.
*********************************************************************************************************
*/

CPU_BOOLEAN  BSP_BumpSensorGetStatus (CPU_INT08U bs)
{
    CPU_BOOLEAN  status;
    
    switch (bs) {
        case 1:
             status = GPIOPinRead(GPIO_PORTE_BASE, GPIO_PIN_0) ? DEF_TRUE : DEF_FALSE;
             break;
             
        case 2:
             status = GPIOPinRead(GPIO_PORTE_BASE, GPIO_PIN_1) ? DEF_TRUE : DEF_FALSE;
             break;

        default:
             break;
    }

    return (status);
}

/*
*********************************************************************************************************
*********************************************************************************************************
*                                        WHEEL SENSOR FUNCTIONS
*********************************************************************************************************
*********************************************************************************************************
*/

/*
*********************************************************************************************************
*                         BSP_WheelSensorsInit(void)
*
* Description : Initializes the infrared wheel sensors.
*
* Argument(s) : none.
*
* Return(s)   : none.
*
* Caller(s)   : BSP_Init().
*
* Note(s)     : none.
*********************************************************************************************************
*/
void  BSP_WheelSensorsInit(void)
{
    // Configure the sensor inputs
    GPIOPinTypeGPIOInput(LEFT_IR_SENSOR_A_PORT, LEFT_IR_SENSOR_A_PIN);
    GPIOPinTypeGPIOInput(LEFT_IR_SENSOR_B_PORT, LEFT_IR_SENSOR_B_PIN);
    GPIOPinTypeGPIOInput(RIGHT_IR_SENSOR_A_PORT, RIGHT_IR_SENSOR_A_PIN);
    GPIOPinTypeGPIOInput(RIGHT_IR_SENSOR_B_PORT, RIGHT_IR_SENSOR_B_PIN);
    
    // Configure the LED outputs.  Initially turn the LEDs off by setting the
    // pins high.
    GPIOPinTypeGPIOOutput(LEFT_RIGHT_IR_LED_PORT, LEFT_RIGHT_IR_LED_PIN);
    GPIOPadConfigSet(LEFT_RIGHT_IR_LED_PORT, LEFT_RIGHT_IR_LED_PIN, GPIO_STRENGTH_8MA, GPIO_PIN_TYPE_STD);

    GPIOPinWrite(LEFT_RIGHT_IR_LED_PORT, LEFT_RIGHT_IR_LED_PIN, LEFT_RIGHT_IR_LED_PIN);
                                  
    // Disable all of the pin interrupts
    GPIOPinIntDisable(LEFT_IR_SENSOR_A_PORT, LEFT_IR_SENSOR_A_PIN);
    GPIOPinIntDisable(LEFT_IR_SENSOR_B_PORT, LEFT_IR_SENSOR_B_PIN);
    GPIOPinIntDisable(RIGHT_IR_SENSOR_A_PORT, RIGHT_IR_SENSOR_A_PIN);
    GPIOPinIntDisable(RIGHT_IR_SENSOR_B_PORT, RIGHT_IR_SENSOR_B_PIN);
    
    GPIOIntTypeSet(LEFT_IR_SENSOR_A_PORT, LEFT_IR_SENSOR_A_PIN, GPIO_RISING_EDGE);
    GPIOIntTypeSet(LEFT_IR_SENSOR_B_PORT, LEFT_IR_SENSOR_B_PIN, GPIO_RISING_EDGE);
    GPIOIntTypeSet(RIGHT_IR_SENSOR_A_PORT, RIGHT_IR_SENSOR_A_PIN, GPIO_RISING_EDGE);
    GPIOIntTypeSet(RIGHT_IR_SENSOR_B_PORT, RIGHT_IR_SENSOR_B_PIN, GPIO_RISING_EDGE);

    
    // Enable the GPIO port interrupts for the inputs.
    // The interrupts for the individual pins still need to be enabled by
    // BSP_WheelSensorIntEnable().
    BSP_IntEn(LEFT_IR_SENSOR_A_INT);
    BSP_IntEn(LEFT_IR_SENSOR_B_INT);
    BSP_IntEn(RIGHT_IR_SENSOR_A_INT);
    BSP_IntEn(RIGHT_IR_SENSOR_B_INT);
}

/*
*********************************************************************************************************
*               BSP_WheelSensorEnable(void)
*
* Description : Enables the infrared wheel sensors by turning on the LEDs.
*
* Argument(s) : none.
*
* Return(s)   : none.
*
* Caller(s)   : Application.
*
* Note(s)     : none.
*********************************************************************************************************
*/
void  BSP_WheelSensorEnable(void)
{
    //turn off the LEDs by setting the pin high    
    GPIOPinWrite(LEFT_RIGHT_IR_LED_PORT, LEFT_RIGHT_IR_LED_PIN,
                         LEFT_RIGHT_IR_LED_PIN);
}

/*
*********************************************************************************************************
*               BSP_WheelSensorDisable(void)
*
* Description : Disables the infrared wheel sensors by turning off the LEDs.
*
* Argument(s) : none.
*
* Return(s)   : none.
*
* Caller(s)   : Application.
*
* Note(s)     : none.
*********************************************************************************************************
*/
void  BSP_WheelSensorDisable(void)
{
    //turn on the LEDs by setting the pin low 
    GPIOPinWrite(LEFT_RIGHT_IR_LED_PORT, LEFT_RIGHT_IR_LED_PIN, 0);
}

/*
*********************************************************************************************************
*              BSP_WheelSensorIntEnable(tSide eSide, tSensor eSensor,
*                                        CPU_FNCT_VOID isr)
*
* Description : Enables the interrupts for infrared wheel sensors.
*
* Argument(s) : none.
*
* Return(s)   : none.
*
* Caller(s)   : Application.
*
* Note(s)     : none.
*********************************************************************************************************
*/
void  BSP_WheelSensorIntEnable(tSide eSide, tSensor eSensor,
                                CPU_FNCT_VOID isr)
{
    // Enable the interrupt for the specified wheel and sensor
    if(eSide == LEFT_SIDE)
    {
        if(eSensor == SENSOR_A)
        {
            //note that the SENSOR_A on the RIGHT_SIDE is also on port E.  This
            //means that they must share an interrupt service routine.            
            GPIOPinIntClear(LEFT_IR_SENSOR_A_PORT, LEFT_IR_SENSOR_A_PIN);
            BSP_IntVectSet(LEFT_IR_SENSOR_A_INT, isr);
            GPIOPinIntEnable(LEFT_IR_SENSOR_A_PORT, LEFT_IR_SENSOR_A_PIN);
        }
        else
        {
            GPIOPinIntClear(LEFT_IR_SENSOR_B_PORT, LEFT_IR_SENSOR_B_PIN);
            BSP_IntVectSet(LEFT_IR_SENSOR_B_INT, isr);
            GPIOPinIntEnable(LEFT_IR_SENSOR_B_PORT, LEFT_IR_SENSOR_B_PIN);
        }
    }
    else
    {
        if(eSensor == SENSOR_A)
        {
            //note that the SENSOR_A on the LEFT_SIDE is also on port E.  This
            //means that they must share an interrupt service routine.
            GPIOPinIntClear(RIGHT_IR_SENSOR_A_PORT, RIGHT_IR_SENSOR_A_PIN);
            BSP_IntVectSet(RIGHT_IR_SENSOR_A_INT, isr);
            GPIOPinIntEnable(RIGHT_IR_SENSOR_A_PORT, RIGHT_IR_SENSOR_A_PIN);
       }
        else
        {
            GPIOPinIntClear(RIGHT_IR_SENSOR_B_PORT, RIGHT_IR_SENSOR_B_PIN);
            BSP_IntVectSet(RIGHT_IR_SENSOR_B_INT, isr);
            GPIOPinIntEnable(RIGHT_IR_SENSOR_B_PORT, RIGHT_IR_SENSOR_B_PIN);
        }
    }
}

/*
*********************************************************************************************************
*              BSP_WheelSensorIntDisable(tSide eSide, tSensor eSensor)
*
* Description : Disables the interrupts for infrared wheel sensors.
*
* Argument(s) : none.
*
* Return(s)   : none.
*
* Caller(s)   : Application.
*
* Note(s)     : none.
*********************************************************************************************************
*/
void  BSP_WheelSensorIntDisable(tSide eSide, tSensor eSensor)
{
    // Disable the interrupt for the specified wheel and sensor
    if(eSide == LEFT_SIDE)
    {
        if(eSensor == SENSOR_A)
        {
            GPIOPinIntDisable(LEFT_IR_SENSOR_A_PORT, LEFT_IR_SENSOR_A_PIN);
        }
        else
        {
            GPIOPinIntDisable(LEFT_IR_SENSOR_B_PORT, LEFT_IR_SENSOR_B_PIN);
        }
    }
    else
    {
        if(eSensor == SENSOR_A)
        {
            GPIOPinIntDisable(RIGHT_IR_SENSOR_A_PORT, RIGHT_IR_SENSOR_A_PIN);
        }
        else
        {
            GPIOPinIntDisable(RIGHT_IR_SENSOR_B_PORT, RIGHT_IR_SENSOR_B_PIN);
        }
    }
}