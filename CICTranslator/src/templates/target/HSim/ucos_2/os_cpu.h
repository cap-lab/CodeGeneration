#ifdef  OS_CPU_GLOBALS
#define OS_CPU_EXT
#else
#define OS_CPU_EXT  extern
#endif


/*
*********************************************************************************************************
*                                              DATA TYPES
*                                         (Compiler Specific)
*********************************************************************************************************
*/

typedef unsigned char  BOOLEAN;
typedef unsigned char  INT8U;                    /* Unsigned  8 bit quantity                           */
typedef signed   char  INT8S;                    /* Signed    8 bit quantity                           */
typedef unsigned short INT16U;                   /* Unsigned 16 bit quantity                           */
typedef signed   short INT16S;                   /* Signed   16 bit quantity                           */
typedef unsigned long  INT32U;                   /* Unsigned 32 bit quantity                           */
typedef signed   long  INT32S;                   /* Signed   32 bit quantity                           */
typedef float          FP32;                     /* Single precision floating point                    */
typedef double         FP64;                     /* Double precision floating point                    */

typedef unsigned long   OS_STK;                   /* Each stack entry is 16-bit wide                    */
typedef unsigned int   OS_CPU_SR;                /* Define size of CPU status register (PSW = 16 bits) */

#define BYTE           INT8S                     /* Define data types for backward compatibility ...   */
#define UBYTE          INT8U                     /* ... to uC/OS V1.xx.  Not actually needed for ...   */
#define WORD           INT32S                    /* ... uC/OS-II.                                      */
#define UWORD          INT32U
#define LONG           INT32S
#define ULONG          INT32U



/* 
 * *********************************************************************************************************
 * *                              ARM v5
 * *
 * * Method #1:  Disable/Enable interrupts using simple instructions.  After critical section, interrupts
 * *             will be enabled even if they were disabled before entering the critical section.
 * *
 * * Method #2:  Disable/Enable interrupts by preserving the state of interrupts.  In other words, if 
 * *             interrupts were disabled before entering the critical section, they will be disabled when
 * *             leaving the critical section.
 * *
 * * Method #3:  Disable/Enable interrupts by preserving the state of interrupts.  Generally speaking you
 * *             would store the state of the interrupt disable flag in the local variable 'cpu_sr' and then
 * *             disable interrupts.  'cpu_sr' is allocated in all of uC/OS-II's functions that need to 
 * *             disable interrupts.  You would restore the interrupt disable state by copying back 'cpu_sr'
 * *             into the CPU's status register.
 * *********************************************************************************************************
 * */
#define  OS_CRITICAL_METHOD    1

#if      OS_CRITICAL_METHOD == 1
#define  OS_ENTER_CRITICAL()  os_enter_critical()
#define  OS_EXIT_CRITICAL()   os_exit_critical() 
#define  OS_ENTER_CRITICAL_ISR()  os_enter_critical_isr()
#define  OS_EXIT_CRITICAL_ISR()   os_exit_critical_isr() 
#endif

#if      OS_CRITICAL_METHOD == 2
//#define  OS_ENTER_CRITICAL()  __asm { \
	STMDB sp!, CPSR_c \
	BIC CPSR_c, #I_Bit:OR:F_Bit \
}                                                      /* Disable interrupts                        */
#define  OS_EXIT_CRITICAL()   __asm { LDMIA sp!, CPSR_c }                              /* Enable  interrupts                        */
#endif

#if      OS_CRITICAL_METHOD == 3
#define  OS_ENTER_CRITICAL()  os_enter_critical()
#define  OS_EXIT_CRITICAL()   os_exit_critical()    /* Enable  interrupts                        */
#endif

/*
 * *********************************************************************************************************
 * *                           Intel 80x86 (Real-Mode, Large Model) Miscellaneous
 * *********************************************************************************************************
 * */

#define  OS_STK_GROWTH        1                       /* Stack grows from HIGH to LOW memory on 80x86  */

extern void os_enter_critical(void);
extern void os_exit_critical(void);
extern void os_enter_critical_isr(void);
extern void os_exit_critical_isr(void);

extern void OSCtxSw(void);
#define  OS_TASK_SW()         OSCtxSw()

/*
 * *********************************************************************************************************
 * *                                            GLOBAL VARIABLES
 * *********************************************************************************************************
 * */

OS_CPU_EXT  INT8U  OSTickDOSCtr;       /* Counter used to invoke DOS's tick handler every 'n' ticks    */

/*
 * *********************************************************************************************************
 * *                                              PROTOTYPES
 * *********************************************************************************************************
 * */

#if OS_CRITICAL_METHOD == 3                      /* Allocate storage for CPU status register           */
OS_CPU_SR  OSCPUSaveSR(void);
void       OSCPURestoreSR(OS_CPU_SR cpu_sr);
#endif
