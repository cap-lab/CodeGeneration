;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;
; Copyright ARM Ltd 2007. All rights reserved.
;
; This module performs ROM/RAM remapping (if required), initializes stack pointers and 
; interrupts for each mode, and finally branches to __main in the C library (which 
; eventually calls main()).
;
; On reset, the ARM core starts up in Supervisor (SVC) mode, in ARM state, with IRQ and FIQ disabled.
;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


        AREA    Init, CODE, READONLY


; --- Standard definitions of mode bits and interrupt (I & F) flags in PSRs

Mode_USR        EQU     0x10
Mode_FIQ        EQU     0x11
Mode_IRQ        EQU     0x12
Mode_SVC        EQU     0x13
Mode_ABT        EQU     0x17
Mode_UNDEF      EQU     0x1B
Mode_SYS        EQU     0x1F ; available on ARM Arch 4 and later

I_Bit           EQU     0x80 ; when I bit is set, IRQ is disabled
F_Bit           EQU     0x40 ; when F bit is set, FIQ is disabled


; --- System memory locations

RAM_Limit       EQU     0x00080000          ; For unexpanded (512KByte) ARM Development (PID7T) board
                                            ; For 2MByte, change to 0x200000 

SVC_Stack       EQU     RAM_Limit           ; 256 byte SVC stack at top of memory
IRQ_Stack       EQU     RAM_Limit-256       ; followed by IRQ stack
; add FIQ_Stack, ABT_Stack, UNDEF_Stack here if you need them
USR_Stack       EQU     IRQ_Stack-256       ; followed by USR stack

ROM_Start       EQU     0x04000000          ; Base address of ROM after remapping
Instruct_2      EQU     ROM_Start + 4       ; Address of second instruction in ROM

ResetBase       EQU     0x0B000000          ; Base address of RPS Remap and Pause Controller
ClearResetMap   EQU     ResetBase + 0x20    ; Offset of remap control from base

        ENTRY

; --- Perform ROM/RAM remapping, if required
    IF :DEF: ROM_RAM_REMAP

; On reset, an aliased copy of ROM is at 0x0.
; Continue execution from 'real' ROM rather than aliased copy
        LDR     pc, =Instruct_2

; Remap by writing to ClearResetMap in the RPS Remap and Pause Controller
        MOV     r0, #0
        LDR     r1, =ClearResetMap
        STRB    r0, [r1]

; RAM is now at 0x0.
; The exception vectors (in vectors.s) must be copied from ROM to the RAM
; The copying is done later by the C library code inside __main

    ENDIF

        EXPORT  Reset_Handler

Reset_Handler

; --- Initialise stack pointer registers
; Enter SVC mode and set up the SVC stack pointer
        MSR     CPSR_c, #Mode_SVC:OR:I_Bit:OR:F_Bit ; No interrupts
        LDR     SP, =SVC_Stack

; Enter IRQ mode and set up the IRQ stack pointer
        MSR     CPSR_c, #Mode_IRQ:OR:I_Bit:OR:F_Bit ; No interrupts
        LDR     SP, =IRQ_Stack


						

; Set up other stack pointers if necessary
        ; ...

; --- Initialise memory system
        ; ...

; --- Initialise critical IO devices
        ; ...

; --- Initialise interrupt system variables here
        ; ...

; --- Now change to User mode and set up User mode stack.
        MSR     CPSR_c, #Mode_SYS:OR:F_Bit ; No interrupts
;        MSR     CPSR_c, #Mode_SVC:OR:F_Bit ; No interrupts

;; lab1		
;; 		MOV R0,PC
;; 		ADD	r1,r1,#1
;; 		ADD	r1,r1,#1
;; 		ADD	r1,r1,#1
;; 		ADD	r1,r1,#1
;; 		ADD	r1,r1,#1
;; 		ADD	r1,r1,#1
;; 		SUB	r1,r1,#1
;; 		SUB	r1,r1,#1
;; 		SUB	r1,r1,#1
;; 		SUB	r1,r1,#1
;; 		SUB	r1,r1,#1
;; 		SUB	r1,r1,#1

;; 		B lab1
		
		LDR     SP, =USR_Stack

        IMPORT  __main

		
; --- Now enter the C code
        B      __main   ; note use B not BL, because an application will never return this way

	EXPORT	EXIT

EXIT
	MOV		r0, #0x18
	STMDB	sp!, {lr}
	SWI		0x0123456
	LDMIA	sp!, {lr}
	END

        END

