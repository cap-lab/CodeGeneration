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
*                                      BASIC BOARD I/O SERVICES
*
*                             TEXAS INSTRUMENTS LM3S9B90 on the EK-LM3S9B90
*
* Filename      : bsp_io.c
* Version       : V1.00
* Programmer(s) : EMO
*********************************************************************************************************
*/


/*
*********************************************************************************************************
*                                            INCLUDE FILES
*********************************************************************************************************
*/

#define   BSP_IO_MODULE
#include <bsp_io.h>


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
*                                            PB FUNCTIONS
*********************************************************************************************************
*********************************************************************************************************
*/

/*
*********************************************************************************************************
*                                            BSP_PushButtonsInit()
*
* Description : Initialize the board's push buttons.
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

void  BSP_PushButtonsInit (void)
{
    GPIOPinTypeGPIOInput(GPIO_PORTD_BASE, GPIO_PIN_6 | GPIO_PIN_7);
    GPIOPadConfigSet(GPIO_PORTD_BASE, GPIO_PIN_6 | GPIO_PIN_7, GPIO_STRENGTH_2MA,
                         GPIO_PIN_TYPE_STD_WPU);
}

/*
*********************************************************************************************************
*                                         BSP_PushButtonGetStatus()
*
* Description : Get the status of a push button on the board.
*
* Argument(s) : pb      The ID of the push button to probe
*
*                       1    probe the USR push button number 1 (SW1)
*                       2    probe the USR push button number 2 (SW2)
*
* Return(s)   : DEF_FALSE   if the push button is pressed.
*               DEF_TRUE    if the push button is not pressed.
*
* Caller(s)   : Application.
*
* Note(s)     : none.
*********************************************************************************************************
*/

CPU_BOOLEAN  BSP_PushButtonGetStatus (CPU_INT08U pb)
{
    CPU_BOOLEAN  status;
    
    switch (pb) {
        case 1:
             status = GPIOPinRead(GPIO_PORTD_BASE, GPIO_PIN_6) ? DEF_TRUE : DEF_FALSE;
             break;
             
        case 2:
             status = GPIOPinRead(GPIO_PORTD_BASE, GPIO_PIN_7) ? DEF_TRUE : DEF_FALSE;
             break;

        default:
             break;
    }

    return (status);
}

/*
*********************************************************************************************************
*********************************************************************************************************
*                                            LED FUNCTIONS
*********************************************************************************************************
*********************************************************************************************************
*/

/*
*********************************************************************************************************
*                                           BSP_LEDsInit()
*
* Description : Initialize the I/O for the LEDs
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

void  BSP_LEDsInit (void)
{
    GPIOPinTypeGPIOOutput(GPIO_PORTF_BASE, GPIO_PIN_4 | GPIO_PIN_5);
    BSP_LED_Off(0);                                            /* Turn OFF all the LEDs.                               */
}


/*
*********************************************************************************************************
*                                            BSP_LED_On()
*
* Description : Turn ON any or all the LEDs on the board.
*
* Argument(s) : led     The ID of the LED to control:
*
*                       0    turn ON all LEDs on the board
*                       1    turn ON LED1
                        2    turn ON LED2
*
* Return(s)   : none.
*
* Caller(s)   : Application.
*
* Note(s)     : none.
*********************************************************************************************************
*/

void  BSP_LED_On (CPU_INT08U led)
{
    switch (led) {
        case 0:
            GPIOPinWrite(GPIO_PORTF_BASE, GPIO_PIN_4 | GPIO_PIN_5, GPIO_PIN_4 | GPIO_PIN_5);
            break;
        case 1:
             GPIOPinWrite(GPIO_PORTF_BASE, GPIO_PIN_4, GPIO_PIN_4);
             break;
        case 2:
             GPIOPinWrite(GPIO_PORTF_BASE, GPIO_PIN_5, GPIO_PIN_5);
             break;

        default:
             break;
    }
}


/*
*********************************************************************************************************
*                                            BSP_LED_Off()
*
* Description : Turn OFF any or all the LEDs on the board.
*
* Argument(s) : led     The ID of the LED to control:
*
*                       0    turn OFF all LEDs on the board
*                       1    turn OFF LED
*
* Return(s)   : none.
*
* Caller(s)   : Application.
*
* Note(s)     : none.
*********************************************************************************************************
*/

void  BSP_LED_Off (CPU_INT08U led)
{
    switch (led) {
        case 0:
            GPIOPinWrite(GPIO_PORTF_BASE, GPIO_PIN_4 | GPIO_PIN_5, 0);
            break;
        case 1:
             GPIOPinWrite(GPIO_PORTF_BASE, GPIO_PIN_4, 0);
             break;
        case 2:
             GPIOPinWrite(GPIO_PORTF_BASE, GPIO_PIN_5, 0);
             break;

        default:
             break;
    }
}


/*
*********************************************************************************************************
*                                          BSP_LED_Toggle()
*
* Description : TOGGLE any or all the LEDs on the board.
*
* Argument(s) : led     The ID of the LED to control:
*
*                       0    TOGGLE all LEDs on the board
*                       1    TOGGLE LED
*
* Return(s)   : none.
*
* Caller(s)   : Application.
*
* Note(s)     : none.
*********************************************************************************************************
*/

void  BSP_LED_Toggle (CPU_INT08U led)
{
    switch (led) {
        case 0:
             GPIOPinWrite(GPIO_PORTF_BASE, GPIO_PIN_4 | GPIO_PIN_5, 
                          ~GPIOPinRead(GPIO_PORTF_BASE, GPIO_PIN_4 | GPIO_PIN_5));
             break;
        case 1:
             GPIOPinWrite(GPIO_PORTF_BASE, GPIO_PIN_4, 
                          ~GPIOPinRead(GPIO_PORTF_BASE, GPIO_PIN_4));
             break;
        case 2:
             GPIOPinWrite(GPIO_PORTF_BASE, GPIO_PIN_5, 
                          ~GPIOPinRead(GPIO_PORTF_BASE, GPIO_PIN_5));
             break;

        default:
             break;
    }
}