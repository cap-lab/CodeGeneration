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
*                                         INTERRUPT SERVICES
*
*                             LUMINARY MICRO LM3S9B90 on the EK-LM3S9B90
*
* Filename      : bsp_int.c
* Version       : V1.02
* Programmer(s) : BAN
*********************************************************************************************************
*/


/*
*********************************************************************************************************
*                                            INCLUDE FILES
*********************************************************************************************************
*/

#define   BSP_INT_MODULE
#include <bsp.h>


/*
*********************************************************************************************************
*                                            LOCAL DEFINES
*********************************************************************************************************
*/

#define  BSP_INT_SRC_NBR                                 70u


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

static  CPU_FNCT_VOID  BSP_IntVectTbl[BSP_INT_SRC_NBR];


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

static  void  BSP_IntHandler     (CPU_DATA  int_id);

static  void  BSP_IntHandlerDummy(void);


/*
*********************************************************************************************************
*                                    EXTERNAL FUNCTION PROTOTYPES
*********************************************************************************************************
*/

void          OSIntEnter         (void);

void          OSIntExit          (void);


/*
*********************************************************************************************************
*                                     LOCAL CONFIGURATION ERRORS
*********************************************************************************************************
*/


/*
*********************************************************************************************************
*                                            BSP_IntClr()
*
* Description : Clear interrupt.
*
* Argument(s) : int_id      Interrupt to clear.
*
* Return(s)   : none.
*
* Caller(s)   : Application.
*
* Note(s)     : (1) An interrupt does not need to be cleared within the interrupt controller.
*********************************************************************************************************
*/

void  BSP_IntClr (CPU_DATA  int_id)
{

}


/*
*********************************************************************************************************
*                                            BSP_IntDis()
*
* Description : Disable interrupt.
*
* Argument(s) : int_id      Interrupt to disable.
*
* Return(s)   : none.
*
* Caller(s)   : Application.
*
* Note(s)     : none.
*********************************************************************************************************
*/

void  BSP_IntDis (CPU_DATA  int_id)
{
    if (int_id < BSP_INT_SRC_NBR) {
        CPU_IntSrcDis((CPU_INT08U)int_id + 16u);
    }
}


/*
*********************************************************************************************************
*                                           BSP_IntDisAll()
*
* Description : Disable ALL interrupts.
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

void  BSP_IntDisAll (void)
{
    CPU_IntDis();
}


/*
*********************************************************************************************************
*                                             BSP_IntEn()
*
* Description : Enable interrupt.
*
* Argument(s) : int_id      Interrupt to enable.
*
* Return(s)   : none.
*
* Caller(s)   : Application.
*
* Note(s)     : none.
*********************************************************************************************************
*/

void  BSP_IntEn (CPU_DATA  int_id)
{
    if (int_id < BSP_INT_SRC_NBR) {
        CPU_IntSrcEn((CPU_INT08U)int_id + 16u);
    }
}


/*
*********************************************************************************************************
*                                          BSP_IntVectSet()
*
* Description : Assign ISR handler.
*
* Argument(s) : int_id      Interrupt for which vector will be set.
*
*               isr         Handler to assign
*
* Return(s)   : none.
*
* Caller(s)   : Application.
*
* Note(s)     : none.
*********************************************************************************************************
*/

void  BSP_IntVectSet (CPU_DATA       int_id,
                      CPU_FNCT_VOID  isr)
{
    CPU_SR_ALLOC();


    if (int_id < BSP_INT_SRC_NBR) {
        CPU_CRITICAL_ENTER();
        BSP_IntVectTbl[int_id] = isr;
        CPU_CRITICAL_EXIT();
    }
}


/*
*********************************************************************************************************
*                                          BSP_IntPrioSet()
*
* Description : Assign ISR priority.
*
* Argument(s) : int_id      Interrupt for which vector will be set.
*
*               prio        Priority to assign
*
* Return(s)   : none.
*
* Caller(s)   : Application.
*
* Note(s)     : none.
*********************************************************************************************************
*/

void  BSP_IntPrioSet (CPU_DATA    int_id,
                      CPU_INT08U  prio)
{
    CPU_SR_ALLOC();


    if (int_id < BSP_INT_SRC_NBR) {
        CPU_CRITICAL_ENTER();
        CPU_IntSrcPrioSet((CPU_INT08U)int_id + 16u, prio);
        CPU_CRITICAL_EXIT();
    }
}


/*
*********************************************************************************************************
*********************************************************************************************************
*                                         INTERNAL FUNCTIONS
*********************************************************************************************************
*********************************************************************************************************
*/

/*
*********************************************************************************************************
*                                            BSP_IntInit()
*
* Description : Initialize interrupts:
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

void  BSP_IntInit (void)
{
    CPU_DATA  int_id;


    for (int_id = 0u; int_id < BSP_INT_SRC_NBR; int_id++) {
        BSP_IntVectSet(int_id, BSP_IntHandlerDummy);
    }
}


/*
*********************************************************************************************************
*                                        BSP_IntHandler####()
*
* Description : Handle an interrupt.
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

void  BSP_IntHandlerGPIOA    (void)  { BSP_IntHandler(BSP_INT_ID_GPIOA);     }
void  BSP_IntHandlerGPIOB    (void)  { BSP_IntHandler(BSP_INT_ID_GPIOB);     }
void  BSP_IntHandlerGPIOC    (void)  { BSP_IntHandler(BSP_INT_ID_GPIOC);     }
void  BSP_IntHandlerGPIOD    (void)  { BSP_IntHandler(BSP_INT_ID_GPIOD);     }
void  BSP_IntHandlerGPIOE    (void)  { BSP_IntHandler(BSP_INT_ID_GPIOE);     }
void  BSP_IntHandlerUART0    (void)  { BSP_IntHandler(BSP_INT_ID_UART0);     }
void  BSP_IntHandlerUART1    (void)  { BSP_IntHandler(BSP_INT_ID_UART1);     }
void  BSP_IntHandlerSSI0     (void)  { BSP_IntHandler(BSP_INT_ID_SSI0);      }
void  BSP_IntHandlerI2C0     (void)  { BSP_IntHandler(BSP_INT_ID_I2C0);      }
void  BSP_IntHandlerADC0     (void)  { BSP_IntHandler(BSP_INT_ID_ADC0);      }
void  BSP_IntHandlerADC1     (void)  { BSP_IntHandler(BSP_INT_ID_ADC1);      }

void  BSP_IntHandlerADC2     (void)  { BSP_IntHandler(BSP_INT_ID_ADC2);      }
void  BSP_IntHandlerADC3     (void)  { BSP_IntHandler(BSP_INT_ID_ADC3);      }
void  BSP_IntHandlerWATCHDOG (void)  { BSP_IntHandler(BSP_INT_ID_WATCHDOG);  }
void  BSP_IntHandlerTIMER0A  (void)  { BSP_IntHandler(BSP_INT_ID_TIMER0A);   }
void  BSP_IntHandlerTIMER0B  (void)  { BSP_IntHandler(BSP_INT_ID_TIMER0B);   }
void  BSP_IntHandlerTIMER1A  (void)  { BSP_IntHandler(BSP_INT_ID_TIMER1A);   }
void  BSP_IntHandlerTIMER1B  (void)  { BSP_IntHandler(BSP_INT_ID_TIMER1B);   }
void  BSP_IntHandlerTIMER2A  (void)  { BSP_IntHandler(BSP_INT_ID_TIMER2A);   }
void  BSP_IntHandlerTIMER2B  (void)  { BSP_IntHandler(BSP_INT_ID_TIMER2B);   }
void  BSP_IntHandlerCOMP0    (void)  { BSP_IntHandler(BSP_INT_ID_COMP0);     }
void  BSP_IntHandlerCOMP1    (void)  { BSP_IntHandler(BSP_INT_ID_COMP1);     }
void  BSP_IntHandlerCOMP2    (void)  { BSP_IntHandler(BSP_INT_ID_COMP2);     }
void  BSP_IntHandlerSYSCTL   (void)  { BSP_IntHandler(BSP_INT_ID_SYSCTL);    }
void  BSP_IntHandlerFLASH    (void)  { BSP_IntHandler(BSP_INT_ID_FLASH);     }
void  BSP_IntHandlerGPIOF    (void)  { BSP_IntHandler(BSP_INT_ID_GPIOF);     }
void  BSP_IntHandlerGPIOG    (void)  { BSP_IntHandler(BSP_INT_ID_GPIOG);     }

void  BSP_IntHandlerGPIOH    (void)  { BSP_IntHandler(BSP_INT_ID_GPIOH);     }
void  BSP_IntHandlerUART2    (void)  { BSP_IntHandler(BSP_INT_ID_UART2);     }
void  BSP_IntHandlerSSI1     (void)  { BSP_IntHandler(BSP_INT_ID_SSI1);      }
void  BSP_IntHandlerTIMER3A  (void)  { BSP_IntHandler(BSP_INT_ID_TIMER3A);   }
void  BSP_IntHandlerTIMER3B  (void)  { BSP_IntHandler(BSP_INT_ID_TIMER3B);   }
void  BSP_IntHandlerI2C1     (void)  { BSP_IntHandler(BSP_INT_ID_I2C1);      }
void  BSP_IntHandlerCAN0     (void)  { BSP_IntHandler(BSP_INT_ID_CAN0);      }
void  BSP_IntHandlerCAN1     (void)  { BSP_IntHandler(BSP_INT_ID_CAN1);      }
void  BSP_IntHandlerETH      (void)  { BSP_IntHandler(BSP_INT_ID_ETH);       }
void  BSP_IntHandlerHIBERNATE(void)  { BSP_IntHandler(BSP_INT_ID_HIBERNATE); }
void  BSP_IntHandlerUSB      (void)  { BSP_IntHandler(BSP_INT_ID_USB);       }
void  BSP_IntHandlerUDMA_SW  (void)  { BSP_IntHandler(BSP_INT_ID_UDMA_SW);   }
void  BSP_IntHandlerUDMA_ERR (void)  { BSP_IntHandler(BSP_INT_ID_UDMA_ERR);  }

void  BSP_IntHandlerADC1_0   (void)  { BSP_IntHandler(BSP_INT_ID_ADC1_0);    }
void  BSP_IntHandlerADC1_1   (void)  { BSP_IntHandler(BSP_INT_ID_ADC1_1);    }
void  BSP_IntHandlerADC1_2   (void)  { BSP_IntHandler(BSP_INT_ID_ADC1_2);    }
void  BSP_IntHandlerADC1_3   (void)  { BSP_IntHandler(BSP_INT_ID_ADC1_3);    }
void  BSP_IntHandlerI2S0     (void)  { BSP_IntHandler(BSP_INT_ID_I2S0);      }
void  BSP_IntHandlerEPI      (void)  { BSP_IntHandler(BSP_INT_ID_EPI);       }
void  BSP_IntHandlerGPIOJ    (void)  { BSP_IntHandler(BSP_INT_ID_GPIOJ);     }


/*
*********************************************************************************************************
*********************************************************************************************************
*                                           LOCAL FUNCTIONS
*********************************************************************************************************
*********************************************************************************************************
*/

/*
*********************************************************************************************************
*                                          BSP_IntHandler()
*
* Description : Central interrupt handler.
*
* Argument(s) : int_id      Interrupt that will be handled.
*
* Return(s)   : none.
*
* Caller(s)   : ISR handlers.
*
* Note(s)     : none.
*********************************************************************************************************
*/

static  void  BSP_IntHandler (CPU_DATA  int_id)
{
    CPU_FNCT_VOID  isr;


    OSIntEnter();                                               /* Tell OS that we are starting an ISR.                 */

    if (int_id < BSP_INT_SRC_NBR) {
        isr = BSP_IntVectTbl[int_id];
        if (isr != (CPU_FNCT_VOID)0) {
            isr();
        }
    }

    OSIntExit();                                                /* Tell OS that we are leaving the ISR.                 */
}


/*
*********************************************************************************************************
*                                        BSP_IntHandlerDummy()
*
* Description : Dummy interrupt handler.
*
* Argument(s) : none.
*
* Return(s)   : none.
*
* Caller(s)   : BSP_IntHandler().
*
* Note(s)     : none.
*********************************************************************************************************
*/

static  void  BSP_IntHandlerDummy (void)
{

}
