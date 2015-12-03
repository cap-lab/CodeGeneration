/*
*********************************************************************************************************
*                                     MICIRUM BOARD SUPPORT PACKAGE
*
*                              (c) Copyright 2010; Micrium, Inc.; Weston, FL
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
*
*                             LUMINARY MICRO LM3S9B92 on the LM3S9B92-EVALBOT
*
* Filename      : bsp_dac.h
* Version       : V1.00
* Programmer(s) : DWN
*********************************************************************************************************
*/

#ifndef __BSP_DAC_H__
#define __BSP_DAC_H__
/*
*********************************************************************************************************
*                                               MODULE
*
* Note(s) : (1) This header file is protected from multiple pre-processor inclusion through use of the
*               BSP present pre-processor macro definition.
*********************************************************************************************************
*/

/*
*********************************************************************************************************
*                                               EXTERNS
*********************************************************************************************************
*/



/*
*********************************************************************************************************
*                                            INCLUDE FILES
*********************************************************************************************************
*/

#include  <bsp.h>

/*
*********************************************************************************************************
*                                               DEFINES
*********************************************************************************************************
*/

// The I2C pins that are used by this application.
#define DAC_I2C_PERIPH                  SYSCTL_PERIPH_I2C0
#define DAC_I2C_MASTER_BASE             I2C0_MASTER_BASE
#define DAC_I2CSCL_GPIO_PERIPH          SYSCTL_PERIPH_GPIOB
#define DAC_I2CSCL_GPIO_PORT            GPIO_PORTB_BASE
#define DAC_I2CSCL_PIN                  GPIO_PIN_2

#define DAC_RESET_GPIO_PERIPH           SYSCTL_PERIPH_GPIOA
#define DAC_RESET_GPIO_PORT             GPIO_PORTA_BASE
#define DAC_RESET_PIN                   GPIO_PIN_7

#define DAC_I2CSDA_GPIO_PERIPH          SYSCTL_PERIPH_GPIOB 
#define DAC_I2CSDA_GPIO_PORT            GPIO_PORTB_BASE
#define DAC_I2CSDA_PIN                  GPIO_PIN_3

// The values to use with the TLV320AIC3107VolumeSet() function.
#define TLV_LINEIN_VC_MAX               0x1f
#define TLV_LINEIN_VC_MIN               0x00
#define TLV_LINEIN_VC_0DB               0x17
#define TLV_LINEIN_VC_MUTE              0x80


//  TLV320AIC3107 Page 0 Register offsets.
#define TI_PAGE_SELECT_R                0        // Page Select Register
#define TI_SOFTWARE_RESET_R             1        // Software Reset Register
#define TI_CODEC_SAMPLE_RATE_R          2        // Code Sample Rate Select Register
#define TI_PLL_PROG_A_R                 3        // PLL Programming Register A
#define TI_PLL_PROG_B_R                 4        // PLL Programming Register B
#define TI_PLL_PROG_C_R                 5        // PLL Programming Register C
#define TI_PLL_PROG_D_R                 6        // PLL Programming Register D
#define TI_CODEC_DATAPATH_R      	7        // Codec Datapath Setup Register
#define TI_ASDI_CTL_A_R            	8        // Audio Serial Data Interface Control Register A
#define TI_ASDI_CTL_B_R            	9        // Audio Serial Data Interface Control Register B
#define TI_ASDI_CTL_C_R            	10       // Audio Serial Data Interface Control Register C
#define TI_ACO_FLAG_R              	11       // Audio Codec Overflow Flag Register
#define TI_ACDF_CTL_R              	12       // Audio Codec Digital Filter Control Register
#define TI_HBPD_A_R               	13       // Headset / Button Press Detection Register A
#define TI_HBPD_B_R               	14       // Headset / Button Press Detection Register B
#define TI_LEFT_ADC_PGA_GAIN_CTL_R 	15       // Left ADC PGA Gain Control Register
#define TI_RIGHT_ADC_PGA_GAIN_CTL_R     16       // Right ADC PGA Gain Control Register
#define TI_MIC3LR_LEFT_CTL_R            17       // MIC3L/R to Left ADC Control Register
#define TI_MIC3LR_RIGHT_CTL_R           18       // MIC3L/R to Right ADC Control Register
#define TI_LINE1L_LEFT_ADC_CTL_R        19       // LINE1L to Left ADC Control Register
#define TI_LINE2L_LEFT_ADC_CTL_R        20       // LINE2L to Left ADC Control Register
#define TI_LINE1R_LEFT_ADC_CTL_R        21       // LINE1R to Left ADC Control Register
#define TI_LINE1R_RIGHT_ADC_CTL_R       22       // LINE1R to Right ADC Control Register
#define TI_LINE2R_RIGHT_ADC_CTL_R       23       // LINE2R to Right ADC Control Register
#define TI_LINE1L_RIGHT_ADC_CTL_R       24       // LINE1L to Right ADC Control Register
#define TI_MICBIAS_CTL_R                25       // MICBIAS Control Register
#define TI_LEFT_AGC_CTL_A_R             26       // Left AGC Control Register A
#define TI_LEFT_AGC_CTL_B_R             27       // Left AGC Control Register B
#define TI_LEFT_AGC_CTL_C_R             28       // Left AGC Control Register C
#define TI_RIGHT_AGC_CTL_A_R            29       // Right AGC Control Register A
#define TI_RIGHT_AGC_CTL_B_R            30       // Right AGC Control Register B
#define TI_RIGHT_AGC_CTL_C_R            31       // Right AGC Control Register C
#define TI_LEFT_AGC_GAIN_R              32       // Left AGC Gain Register
#define TI_RIGHT_AGC_GAIN_R             33       // Right AGC Gain Register
#define TI_LEFT_AGC_NGD_R               34       // Left AGC Noise Gate Debounce Register
#define TI_RIGHT_AGC_NGD_R              35       // Right AGC Noise Gate Debounce Register
#define TI_ADC_FLAG_R             	36       // ADC Flag Register
#define TI_DACPOD_CTL_R                 37       // DAC Power and Output Driver Control Register
#define TI_HPOD_CTL_R                   38       // High Power Output Driver Control Register

#define TI_HPOS_CTL_R                   40       // High Power Output Stage Control Register
#define TI_DACOS_CTL_R             	41       // DAC Output Switching Control Register
#define TI_ODPR_R               	42       // Output Driver Pop Reduction Register
#define TI_LEFT_DAC_DIG_VOL_CTL_R  	43       // Left DAC Digital Volume Control Register
#define TI_RIGHT_DAC_DIG_VOL_CTL_R      44       // Right DAC Digital Voland youume Control Register
#define TI_LINE2L_HPLOUT_VOL_CTL_R      45       // LINE2L to HPLOUT Volume Control Register
#define TI_PGA_L_HPLOUT_VOL_CTL_R       46       // PGA_L to HPLOUT Volume Control Register
#define TI_DAC_L1_HPLOUT_VOL_CTL_R      47       // DAC_L1 to HPLOUT Volume Control Register
#define TI_LINE2R_HPLOUT_VOL_CTL_R      48       // LINE2R to HPLOUT Volume Control Register
#define TI_PGA_R_HPLOUT_VOL_CTL_R       49       // PGA_R to HPLOUT Volume Control Register
#define TI_DAC_R1_HPLOUT_VOL_CTL_R      50       // DAC_R1 to HPLOUT Volume Control Register
#define TI_HPLOUT_OUTPUT_LVL_CTL_R      51       // HPLOUT Output Level Control Register
#define TI_LINE2L_HPCOM_VOL_CTL_R       52       // LINE2L to HPCOM Volume Control Register
#define TI_PGA_L_HPCOM_VOL_CTL_R        53       // PGA_L to HPCOM Volume Control Register
#define TI_DAC_L1_HPCOM_VOL_CTL_R       54       // DAC_L1 to HPCOM Volume Control Register
#define TI_LINE2R_HPCOM_VOL_CTL_R       55       // LINE2R to HPCOM Volume Control Register
#define TI_PGA_R_HPCOM_VOL_CTL_R        56       // PGA_R to HPCOM Volume Control Register
#define TI_DAC_R1_HPCOM_VOL_CTL_R       57       // DAC_R1 to HPCOM Volume Control Regi
#define TI_HPCOM_OUTPUT_LVL_CTL_R       58       // HPCOM Output Level Control Register
#define TI_LINE2L_HPROUT_VOL_CTL_R      59       // LINE2L to HPROUT Volume Control Register
#define TI_PGA_L_HPROUT_VOL_CTL_R       60       // PGA_L to HPROUT Volume Control Register
#define TI_DAC_L1_HPROUT_VOL_CTL_R      61       // DAC_L1 to HPROUT Volume Control Register
#define TI_LINE2R_HPROUT_VOL_CTL_R      62       // LINE2R to HPROUT Volume Control Register
#define TI_PGA_R_HPROUT_VOL_CTL_R       63       // PGA_R to HPROUT Volume Control Register
#define TI_DAC_R1_HPROUT_VOL_CTL_R      64       // DAC_R1 to HPROUT Volume Control Register
#define TI_HPROUT_OUTPUT_LVL_CTL_R      65       // HPROUT Output Level Control Register

#define TI_CLASSD_BYPASS_SWITCH_CTL_R   73       // Class-D and Bypass Switch Control Register

#define TI_ADC_DC_DITHER_CTL_R          76       // ADC DC Dither Control Register

#define TI_LINE2L_LEFT_LOP_VOL_CTL_R    80       // LINE2L to LEFT_LOP Volume Control Register
#define TI_PGA_L_LEFT_LOP_VOL_CTL_R     81       // PGA_L to LEFT_LOP Volume Control Register
#define TI_DAC_L1_LEFT_LOP_VOL_CTL_R    82       // DAC_L1 to LEFT_LOP Volume Control Register
#define TI_LINE2R_LEFT_LOP_VOL_CTL_R    83       // LINE2R to LEFT_LOP Volume Control Register
#define TI_PGA_R_LEFT_LOP_VOL_CTL_R     84       // PGA_R to LEFT_LOP Volume Control Register
#define TI_DAC_R1_LEFT_LOP_VOL_CTL_R    85       // DAC_R1 to LEFT_LOP Volume Control Register
#define TI_LEFT_LOP_OUTPUT_LVL_CTL_R    86       // LEFT_LOP Output Level Control Register
#define TI_LINE2L_RIGHT_LOP_VOL_CTL_R   87       // LINE2L to RIGHT_LOP Volume Control Register
#define TI_PGA_L_RIGHT_LOP_VOL_CTL_R    88       // PGA_L to RIGHT_LOP Volume Control Register
#define TI_DAC_L1_RIGHT_LOP_VOL_CTL_R   89       // DAC_L1 to RIGHT_LOP Volume Control Register
#define TI_LINE2R_RIGHT_LOP_VOL_CTL_R   90       // LINE2R to RIGHT_LOP Volume Control Register
#define TI_PGA_R_RIGhT_LOP_VOL_CTL_R    91       // PGA_R to RIGHT_LOP Volume Control Register
#define TI_DAC_R1_RIGHT_LOP_VOL_CTL_R   92       // DAC_R1 to RIGHT_LOP Volume Control Register
#define TI_RIGHT_LOP_OUTPUT_LVL_CTL_R   93       // RIGHT_LOP Output Level Control Register
#define TI_MODULE_PWR_STAT_R            94       // Module Power Status Register
#define TI_ODSCD_STAT_R                 95       // Output Driver Short Circuit Detection Status Register
#define TI_STICKY_INT_FLAGS_R           96       // Sticky Interrupt Flags Register
#define TI_RT_INT_FLAGS_R               97       // Real-time Interrupt Flags Register
#define TI_GPIO1_CTL_R                  98       // GPIO1 Control Register

#define TI_CODEC_CLKIN_SRC_SEL_R        101      // CODEC CLKIN Source Selection Register
#define TI_CLK_GEN_CTL_R                102      // Clock Generation Control Register
#define TI_LEFT_AGC_ATTACK_TIME_R       103      // Left AGC New Programmable Attack Time Register
#define TI_LEFT_AGC_DECAY_TIME_R        104      // Left AGC Programmable Decay Time Register
#define TI_RIGHT_AGC_ATTACK_TIME_R      105      // Right AGC Programmable Attack Time Register
#define TI_RIGHT_AGC_DECAY_TIME_R       106      // Right AGC New Programmable Decay Time Register(
#define TI_ADC_DP_I2C_COND_R            107      // New Programmable ADC Digital Path and I2C Bus Condition Register
#define TI_PASBSDP_R               	108      // Passive Analog Signal Bypass Selection During Powerdown Register
#define TI_DAC_QCA_R               	109      // DAC Quiescent Current Adjustment Register


// I2C Addresses for the TI DAC.
#define TI_TLV320AIC3107_ADDR           0x018

/*
*********************************************************************************************************
*                                             DATA TYPES
*********************************************************************************************************
*/

/*
*********************************************************************************************************
*                                          GLOBAL VARIABLES
*********************************************************************************************************
*/

/*
*********************************************************************************************************
*                                               MACRO'S
*********************************************************************************************************
*/


/*
*********************************************************************************************************
*                                         FUNCTION PROTOTYPES
*********************************************************************************************************
*/

CPU_BOOLEAN BSP_DACInit(void);
void BSP_DACVolumeSet(CPU_INT32U ulVolume);
CPU_INT32U BSP_DACVolumeGet(void);
void  BSP_DACClassDEn (void);
void  BSP_DACClassDDis (void);

/*
*********************************************************************************************************
*                                        CONFIGURATION ERRORS
*********************************************************************************************************
*/


/*
*********************************************************************************************************
*                                             MODULE END
*********************************************************************************************************
*/

#endif // __BSP_DAC_H__
