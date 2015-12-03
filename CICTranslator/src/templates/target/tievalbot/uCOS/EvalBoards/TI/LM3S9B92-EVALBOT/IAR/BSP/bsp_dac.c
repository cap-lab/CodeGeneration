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
*
*                             LUMINARY MICRO LM3S9B90 on the EK-LM3S9B90
*
* Filename      : bsp_dac.c
* Version       : V1.00
* Programmer(s) : DWN
*                 EMO
*********************************************************************************************************
*/


/*
*********************************************************************************************************
*                                            INCLUDE FILES
*********************************************************************************************************
*/

#include <bsp_dac.h>

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


/*
*********************************************************************************************************
*                                      LOCAL FUNCTION PROTOTYPES
*********************************************************************************************************
*/

static CPU_BOOLEAN BSP_DACWriteRegister(CPU_INT08U ucRegister, CPU_INT32U ulData);
static CPU_BOOLEAN BSP_DACReadRegister(CPU_INT08U ucRegister, CPU_INT08U *pucData);

/*
*********************************************************************************************************
*                                     LOCAL CONFIGURATION ERRORS
*********************************************************************************************************
*/

static CPU_INT08U g_ucHPVolume = 100;

//*****************************************************************************
//
// This is the volume control settings table to use to scale the dB settings
// to a 0-100% scale.  There are 13 entries because 100/8 scaling is 12.5 steps
// which requires 13 entries.
//
//*****************************************************************************
/*static const CPU_INT08U pucVolumeTable[13] =
{
     0x00,
     0x30,
     0x38,
     0x40,
     0x48,
     0x50,
     0x58,
     0x60,
     0x64,
     0x68,
     0x70,
     0x74,
     0x79, // TI_LEFT_HP_VC_0DB,
};*/

/*
*********************************************************************************************************
*                        BSP_DACWriteRegister (CPU_INT08U ucRegister, CPU_INT32U ulData)
*
* Description : Write a register in the TLV320AIC3107 DAC.
*
* Argument(s) : ucRegister is the offset to the register to write.
*               ulData is the data to be written to the DAC register.
*
* Return(s)   : True on success or false on error.
*
* Caller(s)   : Sound driver.
*
* Note(s)     : This function will write the register passed in ucAddr with the value
*               passed in to ulData.  The data in ulData is actually 9 bits and the
*               value in ucAddr is interpreted as 7 bits.
*********************************************************************************************************
*/
static  CPU_BOOLEAN  BSP_DACWriteRegister (CPU_INT08U ucRegister, CPU_INT32U ulData)
{
    // Set the slave address.
    I2CMasterSlaveAddrSet(DAC_I2C_MASTER_BASE, TI_TLV320AIC3107_ADDR, false);

    // Write the first byte to the controller (register)
    I2CMasterDataPut(DAC_I2C_MASTER_BASE, ucRegister);

    // Continue the transfer.
    I2CMasterControl(DAC_I2C_MASTER_BASE, I2C_MASTER_CMD_BURST_SEND_START);

    // Wait until the current byte has been transferred.
    while(I2CMasterIntStatus(DAC_I2C_MASTER_BASE, false) == 0)
    {
    }

    if(I2CMasterErr(DAC_I2C_MASTER_BASE) != I2C_MASTER_ERR_NONE)
    {
        I2CMasterIntClear(DAC_I2C_MASTER_BASE);
        return(false);
    }

    // Wait until the current byte has been transferred.
    while(I2CMasterIntStatus(DAC_I2C_MASTER_BASE, false))
    {
        I2CMasterIntClear(DAC_I2C_MASTER_BASE);
    }

    // Write the data byte to the controller.
    I2CMasterDataPut(DAC_I2C_MASTER_BASE, ulData);

    // End the transfer.
    I2CMasterControl(DAC_I2C_MASTER_BASE, I2C_MASTER_CMD_BURST_SEND_FINISH);

    // Wait until the current byte has been transferred.
    while(I2CMasterIntStatus(DAC_I2C_MASTER_BASE, false) == 0)
    {
    }

    if(I2CMasterErr(DAC_I2C_MASTER_BASE) != I2C_MASTER_ERR_NONE)
    {
        return(false);
    }

    while(I2CMasterIntStatus(DAC_I2C_MASTER_BASE, false))
    {
        I2CMasterIntClear(DAC_I2C_MASTER_BASE);
    }

    return(true);
}

/*
*********************************************************************************************************
*                     BSP_DACReadRegister (CPU_INT08U ucRegister, CPU_INT08U *pucData)
*
* Description : Read a register in the TLV320AIC3107 DAC.
*
* Argument(s) : ucRegister is the offset to the register to write.
*               pucData is a pointer to the returned data.
*
* Return(s)   : True on success or false on error.
*
* Caller(s)   : Sound driver.
*
* Note(s)     : 
*
*********************************************************************************************************
*/
static  CPU_BOOLEAN  BSP_DACReadRegister (CPU_INT08U ucRegister, CPU_INT08U *pucData)
{
    // Set the slave address and "WRITE"/false.
    I2CMasterSlaveAddrSet(DAC_I2C_MASTER_BASE, TI_TLV320AIC3107_ADDR, false);

    // Write the first byte to the controller (register)
    I2CMasterDataPut(DAC_I2C_MASTER_BASE, ucRegister);

    // Continue the transfer.
    I2CMasterControl(DAC_I2C_MASTER_BASE, I2C_MASTER_CMD_BURST_SEND_START);

    // Wait until the current byte has been transferred.
    while(I2CMasterIntStatus(DAC_I2C_MASTER_BASE, false) == 0)
    {
    }

    if(I2CMasterErr(DAC_I2C_MASTER_BASE) != I2C_MASTER_ERR_NONE)
    {
        I2CMasterIntClear(DAC_I2C_MASTER_BASE);
        return(false);
    }

    // Wait until the current byte has been transferred.
    while(I2CMasterIntStatus(DAC_I2C_MASTER_BASE, false))
    {
        I2CMasterIntClear(DAC_I2C_MASTER_BASE);
    }


    // Set the slave address and "READ"/true.
    I2CMasterSlaveAddrSet(DAC_I2C_MASTER_BASE, TI_TLV320AIC3107_ADDR, true);

    // Read Data Byte.
    I2CMasterControl(DAC_I2C_MASTER_BASE, I2C_MASTER_CMD_SINGLE_RECEIVE);

    // Wait until the current byte has been transferred.
    while(I2CMasterIntStatus(DAC_I2C_MASTER_BASE, false) == 0)
    {
    }

    if(I2CMasterErr(DAC_I2C_MASTER_BASE) != I2C_MASTER_ERR_NONE)
    {
        I2CMasterIntClear(DAC_I2C_MASTER_BASE);
        return(false);
    }

    // Wait until the current byte has been transferred.
    while(I2CMasterIntStatus(DAC_I2C_MASTER_BASE, false))
    {
        I2CMasterIntClear(DAC_I2C_MASTER_BASE);
    }

	*pucData  = I2CMasterDataGet(DAC_I2C_MASTER_BASE);

    return(true);
}

/*
*********************************************************************************************************
*                                      BSP_DACInit (void)
*
* Description : Initialize the TLV320AIC3107 DAC.
*
* Argument(s) : None.
*
* Return(s)   : True on success or false on error.
*
* Caller(s)   : Sound driver.
*
* Note(s)     : This function initializes the I2C interface and the TLV320AIC3107 DAC.
*
*********************************************************************************************************
*/
CPU_BOOLEAN  BSP_DACInit (void)
{
    CPU_BOOLEAN bRetcode;
    CPU_INT08U ucTest;

    // Enable the GPIO port containing the I2C pins and set the SDA pin as a
    // GPIO input for now and engage a weak pull-down.  If the daughter board
    // is present, the pull-up on the board should easily overwhelm
    // the pull-down and we should read the line state as high.
    SysCtlPeripheralEnable(DAC_I2CSCL_GPIO_PERIPH);
    GPIOPinTypeGPIOInput(DAC_I2CSCL_GPIO_PORT, DAC_I2CSDA_PIN);
    GPIOPadConfigSet(DAC_I2CSCL_GPIO_PORT, DAC_I2CSDA_PIN, GPIO_STRENGTH_2MA,
                     GPIO_PIN_TYPE_STD_WPD);

    // Enable the I2C peripheral.
    SysCtlPeripheralEnable(DAC_I2C_PERIPH);

    // Delay a while to ensure that we read a stable value from the SDA
    // GPIO pin.  If we read too quickly, the result is unpredictable.
    // This delay is around 2mS.
    SysCtlDelay(SysCtlClockGet() / (3 * 500));
    
    // Configure the pin mux.
    GPIOPinConfigure(GPIO_PB2_I2C0SCL);
    GPIOPinConfigure(GPIO_PB3_I2C0SDA);
      
    // Configure the I2C SCL and SDA pins for I2C operation.
    GPIOPinTypeI2C(DAC_I2CSCL_GPIO_PORT, DAC_I2CSCL_PIN | DAC_I2CSDA_PIN);

    // Initialize the I2C master.
    I2CMasterInitExpClk(DAC_I2C_MASTER_BASE, SysCtlClockGet(), 0);

    // Enable the I2C peripheral.
    SysCtlPeripheralEnable(DAC_RESET_GPIO_PERIPH);

    // Configure the PH2 as a GPIO output.
    GPIOPinTypeGPIOOutput(DAC_RESET_GPIO_PORT, DAC_RESET_PIN);

    // Reset the DAC
    GPIOPinWrite(DAC_RESET_GPIO_PORT , DAC_RESET_PIN, 0);
    GPIOPinWrite(DAC_RESET_GPIO_PORT , DAC_RESET_PIN, DAC_RESET_PIN);

    // Reset the DAC.  Check the return code on this call since we use it to
    // indicate whether or not the DAC is present.  If the register write
    // fails, we assume the I2S daughter board and DAC are not present and
    // return false.
    bRetcode = BSP_DACWriteRegister(TI_SOFTWARE_RESET_R, 0x80);
    if(!bRetcode)
    {
        return(bRetcode);
    }

    // Codec Datapath Setup Register
    // ----------------------
    // D7     = 1  : Fsref = 44.1-kHz
    // D6     = 0  : ADC Dual rate mode is disabled
    // D5     = 0  : DAC Dual rate mode is disabled
    // D[4:3] =	11 : Left DAC datapath plays mono mix of left and right channel input data
    // D[1:1] =	00 : Right DAC datapath is off
    // D0     = 0  : reserved
    // ----------------------
    // D[7:0] =  10011010
    BSP_DACWriteRegister(TI_CODEC_DATAPATH_R, 0x98);

    // Audio Serial Data Interface Control Register A
    // ----------------------
    // D7     = 0  : BCLK is an input (slave mode)
    // D6     = 0  : WCLK (or GPIO1 if programmed as WCLK) is an input (slave mode)
    // D5     = 0  : Do not 3-state DOUT when valid data is not being sent
    // D4     = 0  : BCLK / WCLK (or GPIO1 if programmed as WCLK) will not continue to be transmitted when running in master mode if codec is powered down
    // D3     = 0  : Reserved.
    // D2     = 0  : Disable 3-D digital effect processing
    // D[1:0] = 00 : reserved
    // ----------------------
    // D[7:0] = 00000000 
    BSP_DACWriteRegister(TI_ASDI_CTL_A_R, 0x00);

    // Audio Serial Data Interface Control Register B
    // ----------------------
    // D[7:6] = 00 : Serial data bus uses I2S mode
    // D[5:4] = 00 : Audio data word length = 16-bits
    // D3     = 0  : Continuous-transfer mode used to determine master mode bit clock rate
    // D2     = 0  : Don’t Care
    // D1     = 0  : Don’t Care
    // D0     = 0  : Re-Sync is done without soft-muting the channel. (ADC/DAC)
    // ----------------------
    // D[7:0] = 00000000 
    BSP_DACWriteRegister(TI_ASDI_CTL_B_R, 0x00);

    // Audio Serial Data Interface Control Register C
    // ----------------------
    // D[7:0] = 00000000 : Data offset = 0 bit clocks
    // ----------------------
    // D[7:0] = 00000000 
    BSP_DACWriteRegister(TI_ASDI_CTL_C_R, 0x00);

    // DAC Power and Output Driver Control Register
    // ----------------------
    // D7     = 1  : Left DAC is powered up
    // D6     = 1  : Right DAC is powered up
    // D[5:4] = 00 : HPCOM configured as differential of HPLOUT
    // D[3:0] = 0  : reserved 
    // ----------------------
    // D[7:0] = 11000000 
    BSP_DACWriteRegister(TI_DACPOD_CTL_R, 0xC0);

    // Left DAC Digital Volume Control Register
    // ----------------------
    // D7     = 0  : The left DAC channel is not muted
    // D[6:0] = 0  : 
    // ----------------------
    // D[7:0] =  
    BSP_DACWriteRegister(TI_LEFT_DAC_DIG_VOL_CTL_R, 0x00);

    // Right DAC Digital Volume Control Register
    // ----------------------
    // D7     = 0  : The right DAC channel is not muted
    // D[6:0] = 0  : 
    // ----------------------
    // D[7:0] =  
    BSP_DACWriteRegister(TI_RIGHT_DAC_DIG_VOL_CTL_R, 0x00);
 
    // DAC_L1 to LEFT_LOP Volume Control Register
    // ----------------------
    // D7     = 1  : DAC_L1 is routed to LEFT_LOP
    // D[6:0] = 0110010 (50)  : Gain
    // ----------------------
    // D[7:0] = 10110010 
    BSP_DACWriteRegister(TI_DAC_L1_LEFT_LOP_VOL_CTL_R, 0xA0);

    // LEFT_LOP Output Level Control Register
    // ----------------------
    // D[7:4] = 0110  : Output level control = 6 dB
    // D3     = 1     :	LEFT_LOP is not muted
    // D2     = 0     :	Reserved.
    // D1     = 0     :	All programmed gains to LEFT_LOP have been applied
    // D0     = 1     :	LEFT_LOP is fully powered up
    // ----------------------
    // D[7:0] = 00001001 						  
    BSP_DACWriteRegister(TI_LEFT_LOP_OUTPUT_LVL_CTL_R, 0xC9);

    // From the TLV320AIC3107 datasheet:
    // The following initialization sequence must be written to the AIC3107 
    // registers prior to enabling the class-D amplifier:
    // register data:
    // 1. 0x00 0x0D
    // 2. 0x0D 0x0D
    // 3. 0x08 0x5C
    // 4. 0x08 0x5D
    // 5. 0x08 0x5C
    // 6. 0x00 0x00
    BSP_DACWriteRegister(0x00, 0x0D);
    BSP_DACWriteRegister(0x0D, 0x0D);
    BSP_DACWriteRegister(0x08, 0x5C);
    BSP_DACWriteRegister(0x08, 0x5D);
    BSP_DACWriteRegister(0x08, 0x5C);
    BSP_DACWriteRegister(0x00, 0x00);

    // Class-D and Bypass Switch Control Register
    // ----------------------
    // D[7:6] = 01 : Left Class-D amplifier gain = 6.0 dB
    // D[5:4] = 00 : Right Class-D amplifier gain = 0.0 dB
    // D3     = 1  : enable left class-D channel
    // D2     = 0  : disable right class-D channel
    // D1     = 0  : disable bypass switch
    // D0     = 0  : disable bypass switch bootstrap clock
    // ----------------------
    // D[7:0] = 01001000 
    BSP_DACWriteRegister(TI_CLASSD_BYPASS_SWITCH_CTL_R, 0x40);

    // Read Module Power Status Register
    bRetcode = BSP_DACReadRegister(TI_MODULE_PWR_STAT_R, &ucTest);
    if(!bRetcode)
    {
        return(bRetcode);
    }
 
    return(true);
}

/*
*********************************************************************************************************
*                                      BSP_DACVolumeSet (CPU_INT32U ulVolume)
*
* Description : Set the volume on the DAC.
*
* Argument(s) : ulVolume is the volume to set, specified as a percentage between 0%
*               (silence) and 100% (full volume), inclusive.
*
* Return(s)   : None.
*
* Caller(s)   : Sound driver.
*
* Note(s)     : This function adjusts the audio output up by the specified percentage.  The
*               adjusted volume will not go above 100% (full volume).
*
*********************************************************************************************************
*/
void  BSP_DACVolumeSet (CPU_INT32U ulVolume)
{
    g_ucHPVolume = (CPU_INT08U)ulVolume;

    //
    // Cap the volume at 100%
    //
    if(g_ucHPVolume >= 100)
    {
        g_ucHPVolume = 100;
    }
    
    // Invert the % value.  This is because the max volume is at 0x00 and
    // minimum volume is at 0x7F.
    ulVolume = 100 - ulVolume;

    // Find what % of (0x7F) to set in the register.
    ulVolume = (0x7F * ulVolume) / 100;

    // DAC_L1 to LEFT_LOP Volume Control Register
    // ----------------------
    // D7     = 1  : DAC_L1 is routed to LEFT_LOP
    // D[6:0] =    : Gain
    // ----------------------
    // D[7:0] = 1XXXXXXX
    BSP_DACWriteRegister(TI_DAC_L1_LEFT_LOP_VOL_CTL_R,
                         (0x80 | (CPU_INT08U)ulVolume));   
}

/*
*********************************************************************************************************
*                                      BSP_DACVolumeSet (CPU_INT32U ulVolume)
*
* Description : Returns volume on the DAC.
*
* Argument(s) : None.
*
* Return(s)   : The current volume.
*
* Caller(s)   : Sound driver.
*
* Note(s)     : This function teruns the current volume, specified as a percentage between 
*               0% (silence) and 100% (full volume), inclusive.
*
*********************************************************************************************************
*/
CPU_INT32U  BSP_DACVolumeGet (void)
{
    return(g_ucHPVolume);
}

/*
*********************************************************************************************************
*                         BSP_DACClassDEn (void)
*
* Description : This function enables the class D amplifier in the DAC.
*
* Argument(s) : None.
*
* Return(s)   : None.
*
* Caller(s)   : Sound driver.
*
* Note(s)     : None.
*********************************************************************************************************
*/
void  BSP_DACClassDEn (void)
{
    BSP_DACWriteRegister(TI_CLASSD_BYPASS_SWITCH_CTL_R, 0x48);  
}

/*
*********************************************************************************************************
*                         BSP_DACClassDDis (void)
*
* Description : This function disables the class D amplifier in the DAC.
*
* Argument(s) : None.
*
* Return(s)   : None.
*
* Caller(s)   : Sound driver.
*
* Note(s)     : None.
*********************************************************************************************************
*/
void  BSP_DACClassDDis (void)
{
    BSP_DACWriteRegister(TI_CLASSD_BYPASS_SWITCH_CTL_R, 0x40);  
}