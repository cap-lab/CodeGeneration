;********************************************************************************************************
;                                               uC/OS-II
;                                         The Real-Time Kernel
;
;                          (c) Copyright 1992-2002, Jean J. Labrosse, Weston, FL
;                                          All Rights Reserved
;
;
;                                       80x86/80x88 Specific code
;                                          LARGE MEMORY MODEL
;
;                                           Borland C/C++ V4.51
;                                       (IBM/PC Compatible Target)
;
; File         : OS_CPU_A.ASM
; By           : Jean J. Labrosse
;********************************************************************************************************

		PRESERVE8
		AREA os_cpu_a, CODE, READONLY

;********************************************************************************************************
;                                    PUBLIC and EXTERNAL REFERENCES
;********************************************************************************************************

		ENTRY

		EXPORT OSTickISR
		EXPORT OSStartHighRdy
		EXPORT OSCtxSw
		EXPORT OSIntCtxSw
;		EXPORT os_enter_critical

		IMPORT  OSIntExit
		IMPORT  OSTimeTick
		IMPORT  OSTaskSwHook
		
		IMPORT  OSIntNesting
		IMPORT  OSPrioHighRdy
		IMPORT  OSPrioCur
		IMPORT  OSRunning
		IMPORT  OSTCBCur
		IMPORT  OSTCBHighRdy

;*********************************************************************************************************
;                                          START MULTITASKING
;                                       void OSStartHighRdy(void)
;
; The stack frame is assumed to look as follows:
;
; OSTCBHighRdy->OSTCBStkPtr --> DS                               (Low memory)
;                               ES
;                               DI
;                               SI
;                               BP
;                               SP
;                               BX
;                               DX
;                               CX
;                               AX
;                               OFFSET  of task code address
;                               SEGMENT of task code address
;                               Flags to load in PSW
;                               OFFSET  of task code address
;                               SEGMENT of task code address
;                               OFFSET  of 'pdata'
;                               SEGMENT of 'pdata'               (High memory)
;
; Note : OSStartHighRdy() MUST:
;           a) Call OSTaskSwHook() then,
;           b) Set OSRunning to TRUE,
;           c) Switch to the highest priority task.
;*********************************************************************************************************

OSStartHighRdy

; Run OSTaskSwHook
			LDR		r0, =OSRunning
			MOV		r1, #1
			STRB	r1, [r0]				; set OSRunning value to TRUE

			LDR		r0, =OSTCBHighRdy
			LDR 	r0, [r0]
			LDR		sp, [r0]				; load task's stack pointer

			LDMFD	sp!, {r0}				; load cpsr
			MSR		CPSR_xsf, r0
			LDMFD	sp!, {r0-r12, lr, pc}	; load registers 


;*********************************************************************************************************
;                                PERFORM A CONTEXT SWITCH (From task level)
;                                           void OSCtxSw(void)
;
; Note(s): 1) Upon entry,
;             OSTCBCur     points to the OS_TCB of the task to suspend
;             OSTCBHighRdy points to the OS_TCB of the task to resume
;
;          2) The stack frame of the task to suspend looks as follows:
;
;                 SP -> OFFSET  of task to suspend    (Low memory)
;                       SEGMENT of task to suspend
;                       PSW     of task to suspend    (High memory)
;
;          3) The stack frame of the task to resume looks as follows:
;
;                 OSTCBHighRdy->OSTCBStkPtr --> DS                               (Low memory)
;                                               ES
;                                               DI
;                                               SI
;                                               BP
;                                               SP
;                                               BX
;                                               DX
;                                               CX
;                                               AX
;                                               OFFSET  of task code address
;                                               SEGMENT of task code address
;                                               Flags to load in PSW             (High memory)
;*********************************************************************************************************

OSCtxSw
			STMFD	sp!, {lr}
			STMFD	sp!, {r0-r12, lr}
			MRS		r0, CPSR
			STMFD	sp!, {r0}				; push register set

			LDR		r0, =OSTCBCur
			LDR		r0, [r0]
			STR		sp, [r0]				; set TCB stack pointer

			BL		OSTaskSwHook

			LDR		r0, =OSTCBCur
			LDR		r1, =OSTCBHighRdy
			LDR		r2, [r1]
			STR		r2, [r0]				; save OSTCBHighRdy to OSTCBCur

			LDR		r0, =OSPrioCur
			LDR		r1, =OSPrioHighRdy
			LDRB	r3, [r1]
			STRB	r3, [r0]				; save OSPrioHighRdy to OSPrioCur

			LDR		sp, [r2]
			LDMFD	sp!, {r0}
			MSR		CPSR_xsf, r0
			LDMFD	sp!, {r0-r12, lr, pc}	; load the register set






;*********************************************************************************************************
;                                PERFORM A CONTEXT SWITCH (From an ISR)
;                                        void OSIntCtxSw(void)
;
; Note(s): 1) Upon entry,
;             OSTCBCur     points to the OS_TCB of the task to suspend
;             OSTCBHighRdy points to the OS_TCB of the task to resume
;
;          2) The stack frame of the task to suspend looks as follows:
;
;             OSTCBCur->OSTCBStkPtr ------>  DS                              (Low memory)
;                                            ES
;                                            DI
;                                            SI
;                                            BP
;                                            SP
;                                            BX
;                                            DX
;                                            CX
;                                            AX
;                                            OFFSET  of task code address
;                                            SEGMENT of task code address
;                                            Flags to load in PSW            (High memory)
;
;
;          3) The stack frame of the task to resume looks as follows:
;
;             OSTCBHighRdy->OSTCBStkPtr --> DS                               (Low memory)
;                                           ES
;                                           DI
;                                           SI
;                                           BP
;                                           SP
;                                           BX
;                                           DX
;                                           CX
;                                           AX
;                                           OFFSET  of task code address
;                                           SEGMENT of task code address
;                                           Flags to load in PSW             (High memory)
;*********************************************************************************************************

OSIntCtxSw 

			; save last stack pointer-4 (return address)
			ADD		sp, sp, #8
			LDR		r0, =OSTCBCur
			LDR		r0, [r0]
			STR		sp, [r0]

			BL		OSTaskSwHook

			; save OSTCBHighRdy to OSTCBCur
			LDR		r0, =OSTCBCur
			LDR		r1, =OSTCBHighRdy
			LDR		r2, [r1]
			STR		r2, [r0]

			; set OSPrioCur
			LDR		r0, =OSPrioCur
			LDR		r1, =OSPrioHighRdy
			LDRB	r3, [r1]
			STRB	r3, [r0]

			; load privious register set
			LDR		sp, [r2]
			LDMFD	sp!, {r0}
			MSR		CPSR_xsf, r0
			LDMFD	sp!, {r0-r12, lr, pc}

;*********************************************************************************************************
;                                            HANDLE TICK ISR
;
; Description: This function is called 199.99 times per second or, 11 times faster than the normal DOS
;              tick rate of 18.20648 Hz.  Thus every 11th time, the normal DOS tick handler is called.
;              This is called chaining.  10 times out of 11, however, the interrupt controller on the PC
;              must be cleared to allow for the next interrupt.
;
; Arguments  : none
;
; Returns    : none
;
; Note(s)    : The following C-like pseudo-code describe the operation being performed in the code below.
;
;              Save all registers on the current task's stack;
;              OSIntNesting++;
;              if (OSIntNesting == 1) {
;                 OSTCBCur->OSTCBStkPtr = SS:SP
;              }
;              OSTickDOSCtr--;
;              if (OSTickDOSCtr == 0) {
;                  OSTickDOSCtr = 11;
;                  INT 81H;               Chain into DOS every 54.925 mS
;                                         (Interrupt will be cleared by DOS)
;              } else {
;                  Send EOI to PIC;       Clear tick interrupt by sending an End-Of-Interrupt to the 8259
;                                         PIC (Priority Interrupt Controller)
;              }
;              OSTimeTick();              Notify uC/OS-II that a tick has occured
;              OSIntExit();               Notify uC/OS-II about end of ISR
;              Restore all registers that were save on the current task's stack;
;              Return from Interrupt;
;*********************************************************************************************************
;
OSTickISR
;
            LDR    r0, =OSIntNesting             ; load OSIntNesting address
            LDRB   r1, [r0]                      ; load OSIntNesting value
            ADD    r1, r1, #1
			STRB   r1, [r0]                      ; save OSIntNesting
;
            BL     OSTimeTick                    ; Process system tick
            BL     OSIntExit                     ; Notify uC/OS-II of ISR
;
            LDMFD  sp!, {r0}					; load CPSR
            MSR    CPSR_xsf, r0					; save CPSR
            LDMFD  sp!, {r0-r12, lr, pc}		; Reload register and return

;os_enter_critical
;			STMFD sp!, {r0}
;			MRS   r0, CPSR
;			ORR   r0, r0, #0x80
;			MSR   CPSR_c, r0
;			LDMFD sp!, {r0}
;			MOV   pc, lr
            END
