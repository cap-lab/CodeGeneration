/*
*********************************************************************************************************
*                                               uC/OS-II
*                                         The Real-Time Kernel
*
*                         (c) Copyright 1992-2002, Jean J. Labrosse, Weston, FL
*                                          All Rights Reserved
*
*
*                                       80x86/80x88 Specific code
*                                          LARGE MEMORY MODEL
*
*                                          Borland C/C++ V4.51
*
* File         : OS_CPU_C.C
* By           : Jean J. Labrosse
*********************************************************************************************************
*/

#define  OS_CPU_GLOBALS
#include "includes.h"

/*
*********************************************************************************************************
*                                       OS INITIALIZATION HOOK
*                                            (BEGINNING)
*
* Description: This function is called by OSInit() at the beginning of OSInit().
*
* Arguments  : none
*
* Note(s)    : 1) Interrupts should be disabled during this call.
*********************************************************************************************************
*/
#if OS_CPU_HOOKS_EN > 0 && OS_VERSION > 203
void  OSInitHookBegin (void)
{
}
#endif

/*
*********************************************************************************************************
*                                       OS INITIALIZATION HOOK
*                                               (END)
*
* Description: This function is called by OSInit() at the end of OSInit().
*
* Arguments  : none
*
* Note(s)    : 1) Interrupts should be disabled during this call.
*********************************************************************************************************
*/
#if OS_CPU_HOOKS_EN > 0 && OS_VERSION > 203
void  OSInitHookEnd (void)
{
}
#endif

/*$PAGE*/
/*
*********************************************************************************************************
*                                          TASK CREATION HOOK
*
* Description: This function is called when a task is created.
*
* Arguments  : ptcb   is a pointer to the task control block of the task being created.
*
* Note(s)    : 1) Interrupts are disabled during this call.
*********************************************************************************************************
*/
#if OS_CPU_HOOKS_EN > 0 
void  OSTaskCreateHook (OS_TCB *ptcb)
{
    ptcb = ptcb;                       /* Prevent compiler warning                                     */
}
#endif


/*
*********************************************************************************************************
*                                           TASK DELETION HOOK
*
* Description: This function is called when a task is deleted.
*
* Arguments  : ptcb   is a pointer to the task control block of the task being deleted.
*
* Note(s)    : 1) Interrupts are disabled during this call.
*********************************************************************************************************
*/
#if OS_CPU_HOOKS_EN > 0 
void  OSTaskDelHook (OS_TCB *ptcb)
{
    ptcb = ptcb;                       /* Prevent compiler warning                                     */
}
#endif

/*
*********************************************************************************************************
*                                             IDLE TASK HOOK
*
* Description: This function is called by the idle task.  This hook has been added to allow you to do  
*              such things as STOP the CPU to conserve power.
*
* Arguments  : none
*
* Note(s)    : 1) Interrupts are enabled during this call.
*********************************************************************************************************
*/
#if OS_CPU_HOOKS_EN > 0 && OS_VERSION >= 251
void  OSTaskIdleHook (void)
{
}
#endif

/*
*********************************************************************************************************
*                                           STATISTIC TASK HOOK
*
* Description: This function is called every second by uC/OS-II's statistics task.  This allows your 
*              application to add functionality to the statistics task.
*
* Arguments  : none
*********************************************************************************************************
*/

#if OS_CPU_HOOKS_EN > 0 
void  OSTaskStatHook (void)
{
}
#endif

/*$PAGE*/
/*
*********************************************************************************************************
*                                        INITIALIZE A TASK'S STACK
*
* Description: This function is called by either OSTaskCreate() or OSTaskCreateExt() to initialize the
*              stack frame of the task being created.  This function is highly processor specific.
*
* Arguments  : task          is a pointer to the task code
*
*              pdata         is a pointer to a user supplied data area that will be passed to the task
*                            when the task first executes.
*
*              ptos          is a pointer to the top of stack.  It is assumed that 'ptos' points to
*                            a 'free' entry on the task stack.  If OS_STK_GROWTH is set to 1 then 
*                            'ptos' will contain the HIGHEST valid address of the stack.  Similarly, if
*                            OS_STK_GROWTH is set to 0, the 'ptos' will contains the LOWEST valid address
*                            of the stack.
*
*              opt           specifies options that can be used to alter the behavior of OSTaskStkInit().
*                            (see uCOS_II.H for OS_TASK_OPT_???).
*
* Returns    : Always returns the location of the new top-of-stack' once the processor registers have
*              been placed on the stack in the proper order.
*
* Note(s)    : Interrupts are enabled when your task starts executing. You can change this by setting the
*              PSW to 0x0002 instead.  In this case, interrupts would be disabled upon task startup.  The
*              application code would be responsible for enabling interrupts at the beginning of the task
*              code.  You will need to modify OSTaskIdle() and OSTaskStat() so that they enable 
*              interrupts.  Failure to do this will make your system crash!
*********************************************************************************************************
*/

// use full & decrement stack
// use instruction stmdb(desc before) & ldmia(inc after)

OS_STK  *OSTaskStkInit (void (*task)(void *pd), void *pdata, OS_STK *ptos, INT16U opt)
{
    INT32U *stk;

    opt    = opt;                           /* 'opt' is not used, prevent warning                      */
    stk    = (INT32U *)ptos;                /* Load stack pointer                                      */
    *(--stk) = (INT32U)task; //R15(PC)         /* Simulate call to function with argument */
    *(--stk) = (INT32U)0;    //R14(LR)
    *(--stk) = (INT32U)0;    //R12
    *(--stk) = (INT32U)0;    //R11
    *(--stk) = (INT32U)0;    //R10
    *(--stk) = (INT32U)0;    //R9
    *(--stk) = (INT32U)0;    //R8
    *(--stk) = (INT32U)0;    //R7
    *(--stk) = (INT32U)0;    //R6
    *(--stk) = (INT32U)0;    //R5
    *(--stk) = (INT32U)0;    //R4
    *(--stk) = (INT32U)0;    //R3
    *(--stk) = (INT32U)0;    //R2
    *(--stk) = (INT32U)0;    //R1
    *(--stk) = (INT32U)pdata;//R0
    *(--stk) = (INT32U)0;    //CPSR
    return ((OS_STK *)stk);
}

/*$PAGE*/
/*
*********************************************************************************************************
*                        INITIALIZE A TASK'S STACK FOR FLOATING POINT EMULATION
*
* Description: This function MUST be called BEFORE calling either OSTaskCreate() or OSTaskCreateExt() in
*              order to initialize the task's stack to allow the task to use the Borland floating-point 
*              emulation.  The returned pointer MUST be used in the task creation call.
*
*              Ex.:   OS_STK TaskStk[1000];
*
*
*                     void main (void)
*                     {
*                         OS_STK *ptos;
*                         OS_STK *pbos;
*                         INT32U  size;
*
*
*                         OSInit();
*                         .
*                         .
*                         ptos  = &TaskStk[999];
*                         pbos  = &TaskStk[0];
*                         psize = 1000;
*                         OSTaskStkInit_FPE_x86(&ptos, &pbos, &size);
*                         OSTaskCreate(Task, (void *)0, ptos, 10);
*                         .
*                         .
*                         OSStart();
*                     }
*
* Arguments  : pptos         is the pointer to the task's top-of-stack pointer which would be passed to 
*                            OSTaskCreate() or OSTaskCreateExt().
*
*              ppbos         is the pointer to the new bottom of stack pointer which would be passed to
*                            OSTaskCreateExt().
*
*              psize         is a pointer to the size of the stack (in number of stack elements).  You 
*                            MUST allocate sufficient stack space to leave at least 384 bytes for the 
*                            floating-point emulation.
*
* Returns    : The new size of the stack once memory is allocated to the floating-point emulation.
*
* Note(s)    : 1) _SS  is a Borland 'pseudo-register' and returns the contents of the Stack Segment (SS)
*              2) The pointer to the top-of-stack (pptos) will be modified so that it points to the new
*                 top-of-stack.
*              3) The pointer to the bottom-of-stack (ppbos) will be modified so that it points to the new
*                 bottom-of-stack.
*              4) The new size of the stack is adjusted to reflect the fact that memory was reserved on
*                 the stack for the floating-point emulation.
*********************************************************************************************************
*/

/*$PAGE*/
/*
*********************************************************************************************************
*                                           TASK SWITCH HOOK
*
* Description: This function is called when a task switch is performed.  This allows you to perform other
*              operations during a context switch.
*
* Arguments  : none
*
* Note(s)    : 1) Interrupts are disabled during this call.
*              2) It is assumed that the global pointer 'OSTCBHighRdy' points to the TCB of the task that
*                 will be 'switched in' (i.e. the highest priority task) and, 'OSTCBCur' points to the 
*                 task being switched out (i.e. the preempted task).
*********************************************************************************************************
*/
#if OS_CPU_HOOKS_EN > 0 
void  OSTaskSwHook (void)
{
}
#endif

/*
*********************************************************************************************************
*                                           OSTCBInit() HOOK
*
* Description: This function is called by OS_TCBInit() after setting up most of the TCB.
*
* Arguments  : ptcb    is a pointer to the TCB of the task being created.
*
* Note(s)    : 1) Interrupts may or may not be ENABLED during this call.
*********************************************************************************************************
*/
#if OS_CPU_HOOKS_EN > 0 && OS_VERSION > 203
void  OSTCBInitHook (OS_TCB *ptcb)
{
    ptcb = ptcb;                                           /* Prevent Compiler warning                 */
}
#endif


/*
*********************************************************************************************************
*                                               TICK HOOK
*
* Description: This function is called every tick.
*
* Arguments  : none
*
* Note(s)    : 1) Interrupts may or may not be ENABLED during this call.
*********************************************************************************************************
*/
#if OS_CPU_HOOKS_EN > 0 
void  OSTimeTickHook (void)
{
}
#endif

static int enter_critical_section_count[256] = { 0, };

void os_enter_critical(void)
{
	int cpu_sr = 0;

	__asm {
		MRS cpu_sr, cpsr
		ORR cpu_sr, cpu_sr, 0x80
		MSR cpsr_c, cpu_sr
	}
    ++(enter_critical_section_count[OSTCBCur->OSTCBPrio]);
}

void os_exit_critical(void)
{
	int cpu_sr = 0;

    --(enter_critical_section_count[OSTCBCur->OSTCBPrio]);

    if (enter_critical_section_count[OSTCBCur->OSTCBPrio] <= 0) {
        __asm {
            MRS cpu_sr, cpsr
            BIC cpu_sr, cpu_sr, 0x80
            MSR cpsr_c, cpu_sr
        }
    }
}

void os_enter_critical_isr(void)
{
	int cpu_sr = 0;

	__asm {
		MRS cpu_sr, cpsr
		ORR cpu_sr, cpu_sr, 0x80
		MSR cpsr_c, cpu_sr
	}
}

void os_exit_critical_isr(void)
{
	int cpu_sr = 0;

    __asm {
        MRS cpu_sr, cpsr
        BIC cpu_sr, cpu_sr, 0x80
        MSR cpsr_c, cpu_sr
    }
}
