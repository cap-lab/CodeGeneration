;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;
; Copyright ARM Ltd 2007 All rights reserved.
;
; Where there is ROM fixed at 0x0 (build_b & build_c), these are hard-coded at 0x0.
; Where ROM/RAM remapping occurs (build_d), these are copied from ROM to RAM.
; The copying is done automatically by the C library code inside __main.
;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;



		PRESERVE8
        AREA Vect, CODE, READONLY

; These are example exception vectors and exception handlers

; *****************
; Exception Vectors
; *****************

; Note: LDR PC instructions are used here because branch (B) instructions
; could not simply be copied (the branch offsets would be wrong).  Also,
; a branch instruction might not reach if the ROM is at an address >32MB).


; definition
TIMER_BASE			EQU		0x0A800000
;TIMER_BASE			EQU		0x10A01000
TIMER_LOAD_OFF		EQU		0x000
TIMER_CONTROL_OFF	EQU		0x008
TIMER_CLEAR_OFF		EQU		0x00C
TIMER_LOAD			EQU		TIMER_BASE + TIMER_LOAD_OFF
TIMER_CONTROL		EQU		TIMER_BASE + TIMER_CONTROL_OFF
TIMER_CLEAR			EQU		TIMER_BASE + TIMER_CLEAR_OFF
TIMER_VALUE			EQU		0x7FFF

IRQCT_BASE			EQU		0x0A801000
;IRQCT_BASE			EQU		0x10A00000
IRQCT_ENABLESET		EQU		0x008
IRQCT_ENABLECLR		EQU		0x00C
IRQCT_SET			EQU		IRQCT_BASE + 0x08
IRQCT_CLEAR			EQU		IRQCT_BASE + 0x0C



IRQCT_TIMER_BIT		EQU		4
IRQCT_TIMER			EQU		0x10
IRQCT_IPC			EQU		0x40
IRQCT_ALL			EQU 	IRQCT_TIMER+IRQCT_IPC

; Program start


        ENTRY

        LDR     PC, Reset_Addr
        LDR     PC, Undefined_Addr
        LDR     PC, SWI_Addr
        LDR     PC, Prefetch_Addr
        LDR     PC, Abort_Addr
        NOP                             ; Reserved vector
        LDR     PC, IRQ_Addr
        LDR     PC, FIQ_Addr
        
;        IMPORT  IRQ_Handler             ; In sort.c
        IMPORT  Reset_Handler           ; In init.s
		IMPORT  OSTickISR
		IMPORT	interrupt_receive       ;In ipc_handler.c
		IMPORT  OSIntExit
		IMPORT  OSIntNesting
        
Reset_Addr      DCD     Reset_Handler
Undefined_Addr  DCD     Undefined_Handler
SWI_Addr        DCD     SWI_Handler
Prefetch_Addr   DCD     Prefetch_Handler
Abort_Addr      DCD     Abort_Handler
                DCD     0               ; Reserved vector
;IRQ_Addr        DCD     IRQ_Handler
IRQ_Addr        DCD     interrupt_receive
FIQ_Addr        DCD     FIQ_Handler


; ************************
; Exception Handlers
; ************************

angel_SWIreason_ReportException EQU 0x18
ADP_Stopped_ApplicationExit		EQU 0x20026 						
; The following dummy handlers do not do anything useful in this example.
; They are set up here for completeness.

Undefined_Handler
	SUBS    PC, R14, #4     ; return
	NOP
SWI_Handler
;    STMFD    sp!, {r10}
;	LDR      r10, [lr, #-4]
;	BIC      r10, r10, #0xff000000
;	CMP		 r10, #0x80
;	BEQ      SVC_mode
;	LDMFD    sp!, {r10}

	CMP      R0,#angel_SWIreason_ReportException
    BNE      NOTEXITSWI
	LDR		 R0,=ADP_Stopped_ApplicationExit			
    CMP      R1,R0
    BNE      NOTEXITSWI
	LDR	  	 PC, Reset_Addr					
NOTEXITSWI
	SUBS    PC, R14, #0     ; return
	NOP
;SVC_mode
;	LDMFD    sp!, {r10}
;	SUB      PC, R14, #0	; return

Prefetch_Handler
	SUBS    PC, R14, #4     ; return
	NOP
Abort_Handler
	SUBS    PC, R14, #4     ; return
	NOP

FIQ_Handler
	SUBS    PC, R14, #4     ; return
	NOP

; stack
;--------------------
; pc - task
; lr - task's LR
; r12~r0
; PSR
;--------------------

TimerIRQ
	LDR		r2, =TIMER_CLEAR
	STR		r0, [r2]		; timer clear

	LDR		r2, =IRQCT_CLEAR
	MOV		r0, #IRQCT_ALL
	STR		r0, [r2]		; interrupt clear

	LDR		r2, =IRQCT_SET
	STR		r0, [r2]		; enable timer interrupt

	MOV		r2, sp			; copy IRQ's sp
	ADD		sp, sp, #16		; recover IRQ's sp
	SUB		r3, lr, #4		; copy return address

	LDR		r0, =IRQ_2
	MOVS	pc, r0			; change mode to SVC
IRQ_2
	STMFD	sp!, {r3}		; save return address (task's PC)
	STMFD	sp!, {r4-r12,lr}; --save SVC's R14, R12-r4
	MOV		r4, r2
	LDMFD	r4!, {r0-r3}	; load registers from privious saving
	STMFD	sp!, {r0-r3}	; --save SVC's r3-r0
	
	MRS		r5, CPSR
	STMFD	sp!, {r5}		; --save SVC's PSR

	B OSTickISR
	
;	LDMFD	sp!, {r0-r3}
;	SUBS	pc, lr, #4		; return to main

IPCIRQ
	LDR		r2, =IRQCT_CLEAR;
	MOV		r0, #IRQCT_ALL
	STR		r0, [r2]		; disable interrupt 

	MOV		r2, sp			; copy IRQ's sp
	ADD		sp, sp, #16		; recover IRQ's sp
	SUB		r3, lr, #4		; copy return address

	LDR		r0, =IRQ_21
	MOVS	pc, r0			; change mode to SVC
IRQ_21
	STMFD	sp!, {r3}		; save return address (task's PC)
	STMFD	sp!, {r4-r12,lr}; --save SVC's R14, R12-r4
	MOV		r4, r2
	LDMFD	r4!, {r0-r3}	; load registers from privious saving
	STMFD	sp!, {r0-r3}	; --save SVC's r3-r0
	
	MRS		r5, CPSR
	STMFD	sp!, {r5}		; --save SVC's PSR
	
	LDR    r0, =OSIntNesting             ; load OSIntNesting address
	LDRB   r1, [r0]                      ; load OSIntNesting value
	ADD    r1, r1, #1
	STRB   r1, [r0]                      ; save OSIntNesting
	BL     interrupt_receive             ; Process system tick
	
	LDR		r1, =IRQCT_SET
	MOV		r0, #IRQCT_ALL
	STR		r0, [r1]		; enable  interrupt

	BL     OSIntExit                     ; Notify uC/OS-II of ISR
	
	LDMFD  sp!, {r0}					; load CPSR
	MSR    CPSR_xsf, r0					; save CPSR
	LDMFD  sp!, {r0-r12, lr, pc}		; Reload register and return

IRQ_Handler
	STMFD	sp!, {r0-r3}
	LDR		r1, =IRQCT_BASE
	LDR		r0, [r1]
	TST		r0, #IRQCT_IPC	;To do TST r0, #Inter Processor Interrrupt Pin
	BNE 	IPCIRQ			;Added by HK
	TST		r0, #IRQCT_TIMER
	BNE		TimerIRQ		;junmp to TimerIRQ
	LDMFD	sp!, {r0-r3}
	SUBS	pc, lr #4
	END

