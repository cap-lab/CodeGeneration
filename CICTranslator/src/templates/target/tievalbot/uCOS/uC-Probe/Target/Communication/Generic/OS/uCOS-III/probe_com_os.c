/*
*********************************************************************************************************
*                                       uC/Probe Communication
*
*                         (c) Copyright 2007-2008; Micrium, Inc.; Weston, FL
*
*               All rights reserved.  Protected by international copyright laws.
*               Knowledge of the source code may NOT be used to develop a similar product.
*               Please help us continue to provide the Embedded community with the finest
*               software available.  Your honesty is greatly appreciated.
*********************************************************************************************************
*/

/*
*********************************************************************************************************
*
*                                       COMMUNICATION: GENERIC
*                                       Micrium uC/OS-III PORT
*
* Filename      : probe_com_os.c
* Version       : V2.30
* Programmer(s) : BAN
*********************************************************************************************************
* Note(s)       : (1) This file is the uC/OS-III layer for the uC/Probe Generic Communication Module.
*
*                 (2) Assumes uC/OS-III V3.00+ is included in the project build.
*********************************************************************************************************
*/


/*
*********************************************************************************************************
*                                            INCLUDE FILES
*********************************************************************************************************
*/

#include  <probe_com.h>
#include  <os.h>


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

#if (PROBE_COM_CFG_TERMINAL_REQ_EN == DEF_ENABLED)
static  OS_SEM  ProbeCom_OS_TerminalSem;
#endif


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

#if     (PROBE_COM_CFG_TERMINAL_REQ_EN == DEF_ENABLED)
#if     (OS_CFG_SEM_EN < 1)
#error  "OS_CFG_SEM_EN          illegally #define'd in 'os_cfg.h'   "
#error  "                       [MUST be  > 0]                      "
#endif
#endif


/*
*********************************************************************************************************
*                                         ProbeCom_OS_Init()
*
* Description : Create a semaphore for terminal output completion notification.
*
* Argument(s) : none.
*
* Return(s)   : DEF_OK,   if OS initialization successful.
*               DEF_FAIL, otherwise.
*
* Caller(s)   : ProbeCom_Init().
*
* Note(s)     : none.
*********************************************************************************************************
*/

#if (PROBE_COM_CFG_TERMINAL_REQ_EN == DEF_ENABLED)
CPU_BOOLEAN  ProbeCom_OS_Init (void)
{
    OS_ERR  os_err;


    OSSemCreate((OS_SEM     *)&ProbeCom_OS_TerminalSem,
                (CPU_CHAR   *)"Probe Term",
                (OS_SEM_CTR  ) 0,
                (OS_ERR     *)&os_err);

    if (os_err != OS_ERR_NONE) {
        return (DEF_FAIL);
    }

    return (DEF_OK);
}
#endif


/*
*********************************************************************************************************
*                                    ProbeCom_OS_TerminalOutWait()
*
* Description : Wait for terminal output to complete.
*
* Argument(s) : none.
*
* Return(s)   : none.
*
* Caller(s)   : ProbeCom_TerminalOut().
*
* Note(s)     : none.
*********************************************************************************************************
*/

#if (PROBE_COM_CFG_TERMINAL_REQ_EN == DEF_ENABLED)
void  ProbeCom_OS_TerminalOutWait (void)
{
    OS_ERR  os_err;


    OSSemPend((OS_SEM  *)&ProbeCom_OS_TerminalSem,              /* Wait for output to complete ...                      */
              (OS_TICK  ) 0,                                    /* ... without timeout.                                 */
              (OS_OPT   ) OS_OPT_PEND_BLOCKING,
              (CPU_TS  *) 0,
              (OS_ERR  *)&os_err);

   (void)&os_err;
}
#endif


/*
*********************************************************************************************************
*                                   ProbeCom_OS_TerminalOutSignal()
*
* Description : Signal terminal output completion.
*
* Argument(s) : none.
*
* Return(s)   : none.
*
* Caller(s)   : ProbeCom_CmdTerminalOut().
*
* Note(s)     : none.
*********************************************************************************************************
*/

#if (PROBE_COM_CFG_TERMINAL_REQ_EN == DEF_ENABLED)
void  ProbeCom_OS_TerminalOutSignal (void)
{
    OS_ERR  os_err;


    OSSemPost((OS_SEM *)&ProbeCom_OS_TerminalSem,               /* Output has completed.                                */
              (OS_OPT  )OS_OPT_POST_FIFO,
              (OS_ERR *)&os_err);

   (void)&os_err;
}
#endif
