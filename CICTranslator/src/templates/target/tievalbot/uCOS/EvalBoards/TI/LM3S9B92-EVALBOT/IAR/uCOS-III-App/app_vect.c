/*
*********************************************************************************************************
*                                              EXAMPLE CODE
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
*                                         EXCEPTION VECTORS
*
*                             LUMINARY MICRO LM3S9B90 on the EK-LM3S9B90
*
* Filename      : app_vect.c
* Version       : V1.02
* Programmer(s) : BAN
*********************************************************************************************************
*/

#include <includes.h>


/*
*********************************************************************************************************
*                                            LOCAL DEFINES
*********************************************************************************************************
*/


/*
*********************************************************************************************************
*                                          LOCAL DATA TYPES
*********************************************************************************************************
*/

typedef  union {
    CPU_FNCT_VOID   Fnct;
    void           *Ptr;
} APP_INTVECT_ELEM;



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

#pragma language=extended
#pragma segment="CSTACK"

static  void  App_NMI_ISR        (void);

static  void  App_Fault_ISR      (void);

static  void  App_BusFault_ISR   (void);

static  void  App_UsageFault_ISR (void);

static  void  App_MemFault_ISR   (void);

static  void  App_Spurious_ISR   (void);

extern  void  __iar_program_start(void);


/*
*********************************************************************************************************
*                                     LOCAL CONFIGURATION ERRORS
*********************************************************************************************************
*/


/*
*********************************************************************************************************
*                                  EXCEPTION / INTERRUPT VECTOR TABLE
*
* Note(s) : (1) The Cortex-M3 may have up to 256 external interrupts, which are the final entries in the
*               vector table.  The LM3Sxxxx has 54 external interrupt vectors.
*********************************************************************************************************
*/

__root  const  APP_INTVECT_ELEM  __vector_table[] @ ".intvec" = {
    { .Ptr = (void *)__sfe( "CSTACK" )},                        /*  0, SP start value.                                  */
    __iar_program_start,                                        /*  1, PC start value.                                  */
    App_NMI_ISR,                                                /*  2, NMI.                                             */
    App_Fault_ISR,                                              /*  3, Hard Fault.                                      */
    App_MemFault_ISR,                                           /*  4, Memory Management.                               */
    App_BusFault_ISR,                                           /*  5, Bus Fault.                                       */
    App_UsageFault_ISR,                                         /*  6, Usage Fault.                                     */
    App_Spurious_ISR,                                           /*  7, Reserved.                                        */
    App_Spurious_ISR,                                           /*  8, Reserved.                                        */
    App_Spurious_ISR,                                           /*  9, Reserved.                                        */
    App_Spurious_ISR,                                           /* 10, Reserved.                                        */
    App_Spurious_ISR,                                           /* 11, SVCall.                                          */
    App_Spurious_ISR,                                           /* 12, Debug Monitor.                                   */
    App_Spurious_ISR,                                           /* 13, Reserved.                                        */
    OS_CPU_PendSVHandler,                                       /* 14, PendSV Handler.                                  */
    OS_CPU_SysTickHandler,                                      /* 15, uC/OS-II Tick ISR Handler.                       */

    BSP_IntHandlerGPIOA,
    BSP_IntHandlerGPIOB,
    BSP_IntHandlerGPIOC,
    BSP_IntHandlerGPIOD,
    BSP_IntHandlerGPIOE,
    BSP_IntHandlerUART0,
    BSP_IntHandlerUART1,
    BSP_IntHandlerSSI0,
    BSP_IntHandlerI2C0,
    App_Spurious_ISR,
    App_Spurious_ISR,
    App_Spurious_ISR,
    App_Spurious_ISR,
    App_Spurious_ISR,
    BSP_IntHandlerADC0,
    BSP_IntHandlerADC1,

    BSP_IntHandlerADC2,
    BSP_IntHandlerADC3,
    BSP_IntHandlerWATCHDOG,
    BSP_IntHandlerTIMER0A,
    BSP_IntHandlerTIMER0B,
    BSP_IntHandlerTIMER1A,
    BSP_IntHandlerTIMER1B,
    BSP_IntHandlerTIMER2A,
    BSP_IntHandlerTIMER2B,
    BSP_IntHandlerCOMP0,
    BSP_IntHandlerCOMP1,
    BSP_IntHandlerCOMP2,
    BSP_IntHandlerSYSCTL,
    BSP_IntHandlerFLASH,
    BSP_IntHandlerGPIOF,
    BSP_IntHandlerGPIOG,

    BSP_IntHandlerGPIOH,
    BSP_IntHandlerUART2,
    BSP_IntHandlerSSI1,
    BSP_IntHandlerTIMER3A,
    BSP_IntHandlerTIMER3B,
    BSP_IntHandlerI2C1,
    App_Spurious_ISR,
    BSP_IntHandlerCAN0,
    BSP_IntHandlerCAN1,
    App_Spurious_ISR,
    BSP_IntHandlerETH,
    BSP_IntHandlerHIBERNATE,
    BSP_IntHandlerUSB,
    App_Spurious_ISR,
    BSP_IntHandlerUDMA_SW,
    BSP_IntHandlerUDMA_ERR,
    BSP_IntHandlerADC1_0,
    BSP_IntHandlerADC1_1,
    BSP_IntHandlerADC1_2,
    BSP_IntHandlerADC1_3,
    BSP_IntHandlerI2S0,
    BSP_IntHandlerEPI,
    BSP_IntHandlerGPIOJ
};


/*
*********************************************************************************************************
*                                            App_NMI_ISR()
*
* Description : Handle Non-Maskable Interrupt (NMI).
*
* Argument(s) : none.
*
* Return(s)   : none.
*
* Caller(s)   : This is an ISR.
*
* Note(s)     : (1) Since the NMI is not being used, this serves merely as a catch for a spurious
*                   exception.
*********************************************************************************************************
*/

static  void  App_NMI_ISR (void)
{
    while (DEF_TRUE) {
        ;
    }
}


/*
*********************************************************************************************************
*                                             App_Fault_ISR()
*
* Description : Handle hard fault.
*
* Argument(s) : none.
*
* Return(s)   : none.
*
* Caller(s)   : This is an ISR.
*
* Note(s)     : none.
*********************************************************************************************************
*/

static  void  App_Fault_ISR (void)
{
    while (DEF_TRUE) {
        ;
    }
}


/*
*********************************************************************************************************
*                                           App_BusFault_ISR()
*
* Description : Handle bus fault.
*
* Argument(s) : none.
*
* Return(s)   : none.
*
* Caller(s)   : This is an ISR.
*
* Note(s)     : none.
*********************************************************************************************************
*/

static  void  App_BusFault_ISR (void)
{
    while (DEF_TRUE) {
        ;
    }
}


/*
*********************************************************************************************************
*                                          App_UsageFault_ISR()
*
* Description : Handle usage fault.
*
* Argument(s) : none.
*
* Return(s)   : none.
*
* Caller(s)   : This is an ISR.
*
* Note(s)     : none.
*********************************************************************************************************
*/

static  void  App_UsageFault_ISR (void)
{
    while (DEF_TRUE) {
        ;
    }
}


/*
*********************************************************************************************************
*                                           App_MemFault_ISR()
*
* Description : Handle memory fault.
*
* Argument(s) : none.
*
* Return(s)   : none.
*
* Caller(s)   : This is an ISR.
*
* Note(s)     : none.
*********************************************************************************************************
*/

static  void  App_MemFault_ISR (void)
{
    while (DEF_TRUE) {
        ;
    }
}


/*
*********************************************************************************************************
*                                           App_Spurious_ISR()
*
* Description : Handle spurious interrupt.
*
* Argument(s) : none.
*
* Return(s)   : none.
*
* Caller(s)   : This is an ISR.
*
* Note(s)     : none.
*********************************************************************************************************
*/

static  void  App_Spurious_ISR (void)
{
    while (DEF_TRUE) {
        ;
    }
}
