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
*                                       MOTOR CONTROL SERVICES
*
*                             TEXAS INSTRUMENTS LM3S9B90 on the EK-LM3S9B90
*
* Filename      : bsp_motor.c
* Version       : V1.00
* Programmer(s) : EMO
*********************************************************************************************************
*/


/*
*********************************************************************************************************
*                                            INCLUDE FILES
*********************************************************************************************************
*/

#define   BSP_MOTOR_MODULE

#include <bsp_motor.h>


/*
*********************************************************************************************************
*                                            LOCAL DEFINES
*********************************************************************************************************
*/

#define PWM_CLOCK  (SysCtlClockGet() / 16000)

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
*                                            MOTOR CONTROL FUNCTIONS
*********************************************************************************************************
*********************************************************************************************************
*/

/*
*********************************************************************************************************
*                                            BSP_MotorsInit()
*
* Description : Initialize the motor control logic.
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

void  BSP_MotorsInit (void)
{
    SysCtlPWMClockSet(SYSCTL_PWMDIV_1);
    
    // set up the pin muxing for these pins to be PWM pins
    GPIOPinConfigure(GPIO_PD0_PWM0);
    GPIOPinConfigure(GPIO_PH0_PWM2);
        
    // configure the PWM0 generator
    PWMGenConfigure(PWM_BASE, PWM_GEN_0,
                    PWM_GEN_MODE_UP_DOWN | PWM_GEN_MODE_NO_SYNC);
    PWMGenPeriodSet(PWM_BASE, PWM_GEN_0, PWM_CLOCK);
    // configure the PWM1 generator
    PWMGenConfigure(PWM_BASE, PWM_GEN_1,
                    PWM_GEN_MODE_UP_DOWN | PWM_GEN_MODE_NO_SYNC);
    PWMGenPeriodSet(PWM_BASE, PWM_GEN_1, PWM_CLOCK);
    
   
    // configure the pulse widths for each PWM signal to initially 0%
    PWMPulseWidthSet(PWM_BASE, PWM_OUT_0, 0);
    PWMPulseWidthSet(PWM_BASE, PWM_OUT_2, 0);
    
    // Initially disable the the PWM0 and PWM2 output signals.
    PWMOutputState(PWM_BASE, PWM_OUT_0_BIT | PWM_OUT_2_BIT, false);

    //
    // Enable the PWM generators.
    //
    PWMGenEnable(PWM_BASE, PWM_GEN_0);
    PWMGenEnable(PWM_BASE, PWM_GEN_1);
    
    // Set the pins connected to the motor driver fault signal to input with
    // pull ups
    GPIOPinTypeGPIOInput(GPIO_PORTD_BASE, GPIO_PIN_3);
    GPIOPadConfigSet(GPIO_PORTD_BASE, GPIO_PIN_3, GPIO_STRENGTH_2MA, GPIO_PIN_TYPE_STD_WPU);
    
    // Enable slow decay mode
    GPIOPinTypeGPIOOutput(GPIO_PORTD_BASE, GPIO_PIN_2);
    GPIOPinWrite(GPIO_PORTD_BASE, GPIO_PIN_2, GPIO_PIN_2);

    // initially configure the direction control and enable pins as GPIO and set low
    GPIOPinTypeGPIOOutput(GPIO_PORTD_BASE, GPIO_PIN_0 | GPIO_PIN_1);
    GPIOPinTypeGPIOOutput(GPIO_PORTH_BASE, GPIO_PIN_0 | GPIO_PIN_1);
    GPIOPinWrite(GPIO_PORTD_BASE, GPIO_PIN_0 | GPIO_PIN_1, 0);
    GPIOPinWrite(GPIO_PORTH_BASE, GPIO_PIN_0 | GPIO_PIN_1, 0);
    
    // Enable the 12V boost
    GPIOPinTypeGPIOOutput(GPIO_PORTD_BASE, GPIO_PIN_5);
    GPIOPinWrite(GPIO_PORTD_BASE, GPIO_PIN_5, GPIO_PIN_5);
}

/*
*********************************************************************************************************
*               BSP_MotorDir(CPU_INT08U ucMotor, tDirection bForward)
*
* Description : Configures the DMOS Motor Driver to drive the motor in the 
*               required direction.
*
* Argument(s) : bForward   The direction to drive the motor.
*
* Return(s)   : none.
*
* Caller(s)   : Application().
*
* Note(s)     : none.
*********************************************************************************************************
*/

void  BSP_MotorDir (tSide ucMotor, tDirection eDirection)
{
    if(ucMotor == LEFT_SIDE)
    {
        if(eDirection == FORWARD)
        {
            GPIOPinWrite(GPIO_PORTH_BASE, GPIO_PIN_1 , 0);
        }
        
        else
        {
            GPIOPinWrite(GPIO_PORTH_BASE, GPIO_PIN_1 , GPIO_PIN_1);
        }
    }
    else
    {
        if(eDirection == FORWARD)
        {
            GPIOPinWrite(GPIO_PORTD_BASE, GPIO_PIN_1 , GPIO_PIN_1);
        }
        else
        {
            GPIOPinWrite(GPIO_PORTD_BASE, GPIO_PIN_1 , 0);
        }
    }
}

/*
*********************************************************************************************************
*                                            BSP_MotorRun()
*
* Description : Starts the motor.
*
* Argument(s) : none.
*
* Return(s)   : none.
*
* Caller(s)   : Application().
*
* Note(s)     : none.
*********************************************************************************************************
*/

void  BSP_MotorRun (tSide ucMotor)
{
    if(ucMotor == LEFT_SIDE)
    {
        // Configure the pin to be controlled by the PWM module
        GPIOPinTypePWM(GPIO_PORTH_BASE, GPIO_PIN_0);
    }
    else
    {
        // Configure the pin to be controlled by the PWM module
        GPIOPinTypePWM(GPIO_PORTD_BASE, GPIO_PIN_0);        
    }
}

/*
*********************************************************************************************************
*                                            BSP_MotorStop()
*
* Description : Stops the motor.
*
* Argument(s) : none.
*
* Return(s)   : none.
*
* Caller(s)   : Application().
*
* Note(s)     : none.
*********************************************************************************************************
*/

void  BSP_MotorStop (tSide ucMotor)
{
    if(ucMotor == LEFT_SIDE)
    {
        // Configure the pin to be a software controlled GP Output
        GPIOPinTypeGPIOOutput(GPIO_PORTH_BASE, GPIO_PIN_0);
        // Set the pin low
        GPIOPinWrite(GPIO_PORTH_BASE, GPIO_PIN_0, 0);
    }
    else
    {
        // Configure the pin to be a software controlled GP Output
        GPIOPinTypeGPIOOutput(GPIO_PORTD_BASE, GPIO_PIN_0);
        // Set the pin low
        GPIOPinWrite(GPIO_PORTD_BASE, GPIO_PIN_0, 0);
    }
}

/*
*********************************************************************************************************
*                                            BSP_MotorSpeed()
*
* Description : Sets the motor to be driven at the requested speed.
*
* Argument(s) : usPercent   Percent of the maximum speed to drive the motor in
*                           8.8 fixed point format.
*
* Return(s)   : none.
*
* Caller(s)   : Application().
*
* Note(s)     : none.
*********************************************************************************************************
*/

void  BSP_MotorSpeed(tSide ucMotor, CPU_INT16U usPercent)
{
    /*// Set the PWM duty cycle to match that of the requested speed
    // Is the requested percentage is greater than 95%?
    if(usPercent > (95 << 8))
    {
        // Just set the signal high for 100% duty cycle for speeds this high.
        // There is no need to incur the switching losses for negligible
        // the speed differences in this range.  Additionally, this serves as
        // a workaround for the errata titled "PWM generation is incorrect with
        // extreme duty cycles"
        
        if(ucMotor == LEFT_SIDE)
        {
            // first, disable the PWM output
            PWMOutputState(PWM_BASE, PWM_OUT_2_BIT, false);
            // then, invert to the output so that it is high
            PWMOutputInvert(PWM_BASE, PWM_OUT_2_BIT, true);
        }
        else
        {
            // first, disable the PWM output
            PWMOutputState(PWM_BASE, PWM_OUT_0_BIT, false);
            // then, invert to the output so that it is high
            PWMOutputInvert(PWM_BASE, PWM_OUT_0_BIT, true);        
        }
    }
    else
    {*/
    
        if(ucMotor == LEFT_SIDE)
        {   
            // first, enable the PWM output in case it was disabled by the
            // previously requested speed being greater than 95%
            PWMOutputState(PWM_BASE, PWM_OUT_2_BIT, true);
            // make sure that output is not inverted
            PWMOutputInvert(PWM_BASE, PWM_OUT_2_BIT, false);
            //set the pulse width to the requested value
            //divide by two since we are using 6V motors with 12V power rail
            //PWMPulseWidthSet(PWM_BASE, PWM_OUT_2, ((PWM_CLOCK * usPercent) / 
            //                                      (100 << 8)) / 2);
            PWMPulseWidthSet(PWM_BASE, PWM_OUT_2, ((PWM_CLOCK * usPercent) / 
                                                    (100 << 8)));
        }
        else
        {
            // first, enable the PWM output in case it was disabled by the
            // previously requested speed being greater than 95%
            PWMOutputState(PWM_BASE, PWM_OUT_0_BIT, true);
            // make sure that output is not inverted
            PWMOutputInvert(PWM_BASE, PWM_OUT_0_BIT, false);
            //set the pulse width to the requested value
            //divide by two since we are using 6V motors with 12V power rail
            //PWMPulseWidthSet(PWM_BASE, PWM_OUT_0, ((PWM_CLOCK * usPercent) /
            //                                      (100 << 8)) / 2);
            PWMPulseWidthSet(PWM_BASE, PWM_OUT_0, ((PWM_CLOCK * usPercent) /
                                                  (100 << 8)));
        }
    //}
}
