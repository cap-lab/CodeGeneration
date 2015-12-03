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
* Filename      : bsp_sound.c
* Version       : V1.00
* Programmer(s) : EMO
*********************************************************************************************************
*/


/*
*********************************************************************************************************
*                                            INCLUDE FILES
*********************************************************************************************************
*/

#include <bsp_sound.h>

/*
*********************************************************************************************************
*                                            LOCAL DEFINES
*********************************************************************************************************
*/

#define SAMPLE_RATE             44100

#define SAMPLE_LEFT_UP          0x00000001
#define SAMPLE_RIGHT_UP         0x00000002

#define NUM_SAMPLES             512

#define NUM_BUFFERS             2

#define FLAG_RX_PENDING         0
#define FLAG_TX_PENDING         1

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

// Sawtooth state information, this allows for a phase difference between left
// and right waveforms.
volatile struct
{
    CPU_INT32S iSample;
    CPU_INT32U ulFlags;
} g_sSample;

// Buffer management structures and defines.
static struct
{
    // Pointer to the buffer.
    CPU_INT32U *pulData;

    // Size of the buffer.
    CPU_INT32U ulSize;

    // Callback function for this buffer.
    tBufferCallback pfnBufferCallback;
}
g_sOutBuffers[NUM_BUFFERS];

// The DMA control structure table.
#pragma data_alignment=1024
tDMAControlTable sDMAControlTable[64];

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

// The current volume of the music/sound effects.
static CPU_INT08U g_ucVolume = 100;

// A pointer to the song currently being played, if any.  The value of this
// variable is used to determine whether or not a song is being played.  Since
// each entry is a short, the maximum length of the song is 65536 / 200
// seconds, which is around 327 seconds.
static const CPU_INT16U *g_pusMusic = 0;

// Interrupt values for tone generation.
static CPU_INT32U g_ulFrequency;
static CPU_INT32U g_ulDACStep;
static CPU_INT32U g_ulSize;
static CPU_INT32U g_ulTicks;
static CPU_INT16U g_ulMusicCount;
static CPU_INT16U g_ulMusicSize;

static CPU_INT32U g_pulTxBuf[NUM_SAMPLES];

// A set of flags.  The flag bits are defined as follows:
//     0 -> A RX DMA transfer is pending.
//     1 -> A TX DMA transfer is pending.
static volatile CPU_INT32U g_ulDMAFlags;

// The buffer index that is currently playing.
static CPU_INT32U g_ulPlaying;

static CPU_INT32U g_ulSampleRate;
static CPU_INT16U g_usChannels;
static CPU_INT32U g_usBitsPerSample;

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

/*
*********************************************************************************************************
*                                            BSP_SoundPatternNext (void)
*
* Description : This function is used to generate a pattern to fill the TX buffer.
*
* Argument(s) : none.
*
* Return(s)   : Sample.
*
* Caller(s)   : BSP_SoundNextTone().
*
* Note(s)     : none.
*********************************************************************************************************
*/
static  CPU_INT32U  BSP_SoundPatternNext (void)
{
    CPU_INT32S iSample;

    if(g_sSample.ulFlags & SAMPLE_LEFT_UP)
    {
        g_sSample.iSample += g_ulDACStep;
        if(g_sSample.iSample >= 32767)
        {
            g_sSample.ulFlags &= ~SAMPLE_LEFT_UP;
            g_sSample.iSample = 32768 - g_ulDACStep;
        }
    }
    else
    {
        g_sSample.iSample -= g_ulDACStep;
        if(g_sSample.iSample <= -32768)
        {
            g_sSample.ulFlags |= SAMPLE_LEFT_UP;
            g_sSample.iSample = g_ulDACStep - 32768;
        }
    }

    // Copy the sample to prevent a compiler warning on the return line.
    iSample = g_sSample.iSample;
    return((iSample & 0xffff) | (iSample << 16));
}

/*
*********************************************************************************************************
*                                            BSP_SoundNextTone (void)
*
* Description : Generate the next tone.
*
* Argument(s) : none.
*
* Return(s)   : Sample.
*
* Caller(s)   : BSP_SoundPlay().
*
* Note(s)     : none.
*********************************************************************************************************
*/
static  CPU_INT32U  BSP_SoundNextTone (void)
{
    CPU_INT32S iIdx;

    g_sSample.iSample = 0;
    g_sSample.ulFlags = SAMPLE_LEFT_UP;

    // Set the frequency.
    g_ulFrequency = g_pusMusic[g_ulMusicCount + 1];

    // Calculate the step size for each sample.
    g_ulDACStep = ((65536 * 2 * g_ulFrequency) / SAMPLE_RATE);

    // How big is the buffer that needs to be restarted.
    g_ulSize = (SAMPLE_RATE/g_ulFrequency);

    // Cap the size in a somewhat noisy way.  This will affect frequencies below
    // 93.75 Hz or 48000/NUM_SAMPLES.
    if(g_ulSize > NUM_SAMPLES)
    {
        g_ulSize = NUM_SAMPLES;
    }

    // Move on to the next value.
    g_ulMusicCount += 2;

    // Stop if there are no more entries in the list.
    if(g_ulMusicCount < g_ulMusicSize)
    {
        g_ulTicks = (g_pusMusic[g_ulMusicCount] * g_ulFrequency) / 1000;
    }
    else
    {
        g_ulTicks = 0;
    }

    // Fill the buffer with the new tone.
    for(iIdx = 0; iIdx < g_ulSize; iIdx++)
    {
        g_pulTxBuf[iIdx] = BSP_SoundPatternNext();
    }

    // This should be the size in bytes and not words.
    g_ulSize <<= 2;

    return(g_ulTicks);
}

/*
*********************************************************************************************************
*                        BSP_SoundBufferCallback (void *pvBuffer, CPU_INT32U ulEvent)
*
* Description : Handles playback of the single buffer when playing tones.
*
* Argument(s) : pvBuffer is a pointer to the buffer.
*               ulEvent is the buffer event to check for.
*
* Return(s)   : None.
*
* Caller(s)   : BSP_SoundPlay().
*
* Note(s)     : none.
*********************************************************************************************************
*/
static  void  BSP_SoundBufferCallback (void *pvBuffer, CPU_INT32U ulEvent)
{
    if((ulEvent & BUFFER_EVENT_FREE) && (g_ulTicks != 0))
    {
        // Kick off another request for a buffer playback.
        BSP_SoundBufferPlay(pvBuffer, g_ulSize, BSP_SoundBufferCallback);

        // Count down before stopping.
        g_ulTicks--;
    }
    else
    {
        // Stop requesting transfers.
        I2STxDisable(I2S0_BASE);
    }
}

/*
*********************************************************************************************************
*                         BSP_SoundInit (void)
*
* Description : Initializes the sound output.
*
* Argument(s) : ulEnableReceive is set to 1 to enable the receive portion of the I2S
*                 controller and 0 to leave the I2S controller not configured.
*
* Return(s)   : None.
*
* Caller(s)   : Main application.
*
* Note(s)     : None.
*********************************************************************************************************
*/
void  BSP_SoundInit (void)
{
    // Set the current active buffer to zero.
    g_ulPlaying = 0;

    // Enable and reset the peripheral.
    SysCtlPeripheralEnable(SYSCTL_PERIPH_I2S0);
    SysCtlPeripheralEnable(SYSCTL_PERIPH_UDMA);
    
    // Set up the pin mux.
    GPIOPinConfigure(GPIO_PB6_I2S0TXSCK);
    GPIOPinConfigure(GPIO_PE4_I2S0TXWS);
    GPIOPinConfigure(GPIO_PE5_I2S0TXSD);
    GPIOPinConfigure(GPIO_PF1_I2S0TXMCLK);

    // Select alternate functions for all of the I2S pins.
    SysCtlPeripheralEnable(I2S0_SCLKTX_PERIPH);
    GPIOPinTypeI2S(I2S0_SCLKTX_PORT, I2S0_SCLKTX_PIN);

    SysCtlPeripheralEnable(I2S0_MCLKTX_PERIPH);
    GPIOPinTypeI2S(I2S0_MCLKTX_PORT, I2S0_MCLKTX_PIN);

    SysCtlPeripheralEnable(I2S0_LRCTX_PERIPH);
    GPIOPinTypeI2S(I2S0_LRCTX_PORT, I2S0_LRCTX_PIN);

    SysCtlPeripheralEnable(I2S0_SDATX_PERIPH);
    GPIOPinTypeI2S(I2S0_SDATX_PORT, I2S0_SDATX_PIN);
    
    // Set up the DMA.
    uDMAControlBaseSet(&sDMAControlTable[0]);
    uDMAEnable();

    // Initialize the DAC.
    BSP_DACInit();

    // Set the FIFO trigger limit
    I2STxFIFOLimitSet(I2S0_BASE, 4);

    // Clear out all pending interrupts.
    I2SIntClear(I2S0_BASE, I2S_INT_TXERR | I2S_INT_TXREQ );

    // Disable all uDMA attributes.
    uDMAChannelAttributeDisable(UDMA_CHANNEL_I2S0TX, UDMA_ATTR_ALL);
    
    // Enable the I2S Tx controller.
    I2STxEnable(I2S0_BASE);
      
    // Enable the I2S interrupt on the NVIC.
    BSP_IntVectSet(BSP_INT_ID_I2S0, BSP_SoundIntHandler);
}

/*
*********************************************************************************************************
*                         BSP_SoundIntHandler (void)
*
* Description : This function services the I2S interrupt and will call the callback function
*               provided with the buffer that was given to the BSP_SoundBufferPlay() or
*               BSPSoundBufferRead() functions to handle emptying or filling the buffers and
*               starting up DMA transfers again.  It is solely the responsibility of the
*               callback functions to continuing sending or receiving data to or from the
*               audio codec.
*
* Argument(s) : None.
*
* Return(s)   : None.
*
* Caller(s)   : Main application.
*
* Note(s)     : None.
*********************************************************************************************************
*/
void  BSP_SoundIntHandler (void)
{
    CPU_INT32U ulStatus;
    CPU_INT32U *pulTemp;
    
    // Get the interrupt status and clear any pending interrupts.
    ulStatus = I2SIntStatus(I2S0_BASE, 1);

    // Clear out any interrupts.
    I2SIntClear(I2S0_BASE, ulStatus);

    // Handle the TX channel interrupt
    if(HWREGBITW(&g_ulDMAFlags, FLAG_TX_PENDING))
    {
        // If the TX DMA is done, then call the callback if present.
        if(uDMAChannelModeGet(UDMA_CHANNEL_I2S0TX | UDMA_PRI_SELECT) ==
           UDMA_MODE_STOP)
        {
            // Save a temp pointer so that the current pointer can be set to
            // zero before calling the callback.
            pulTemp = g_sOutBuffers[0].pulData;

            // If at the mid point then refill the first half of the buffer.
            if((g_sOutBuffers[0].pfnBufferCallback) &&
               (g_sOutBuffers[0].pulData != 0))
            {
                g_sOutBuffers[0].pulData = 0;
                g_sOutBuffers[0].pfnBufferCallback(pulTemp, BUFFER_EVENT_FREE);
            }
        }

        // If the TX DMA is done, then call the callback if present.
        if(uDMAChannelModeGet(UDMA_CHANNEL_I2S0TX | UDMA_ALT_SELECT) ==
           UDMA_MODE_STOP)
        {
            // Save a temporary pointer so that the current pointer can be set
            // to zero before calling the callback.
            pulTemp = g_sOutBuffers[1].pulData;

            // If at the mid point then refill the first half of the buffer.
            if((g_sOutBuffers[1].pfnBufferCallback) &&
               (g_sOutBuffers[1].pulData != 0))
            {
                g_sOutBuffers[1].pulData = 0;
                g_sOutBuffers[1].pfnBufferCallback(pulTemp, BUFFER_EVENT_FREE);
            }
        }

        // If no more buffers are pending then clear the flag.
        if((g_sOutBuffers[0].pulData == 0) && (g_sOutBuffers[1].pulData == 0))
        {
            HWREGBITW(&g_ulDMAFlags, FLAG_TX_PENDING) = 0;
        }
    }
}

/*
*********************************************************************************************************
*                         BSP_SoundPlay (const CPU_INT16U *pusSong, CPU_INT32U ulLength)
*
* Description : This function starts the playback of a song or sound effect.  If a song
*               or sound effect is already being played, its playback is canceled and the
*               new song is started.
*
* Argument(s) : pusSong is a pointer to the song data structure.
*               ulLength is the length of the song data structure in bytes.
*
* Return(s)   : None.
*
* Caller(s)   : Main application.
*
* Note(s)     : None.
*********************************************************************************************************
*/
void  BSP_SoundPlay (const CPU_INT16U *pusSong, CPU_INT32U ulLength)
{
    //
    // Set the format of the audio stream.
    //
    BSP_SoundSetFormat(44100, 8, 1);

    //
    // Save the music buffer.
    //
    g_ulMusicCount = 0;
    g_ulMusicSize = ulLength * 2;
    g_pusMusic = pusSong;
    g_ulPlaying = 0;

    g_sOutBuffers[0].pulData = 0;
    g_sOutBuffers[1].pulData = 0;

    if(BSP_SoundNextTone() != 0)
    {
        BSP_SoundBufferPlay(g_pulTxBuf, g_ulSize, BSP_SoundBufferCallback);
        BSP_SoundBufferPlay(g_pulTxBuf, g_ulSize, BSP_SoundBufferCallback);
    }
}

/*
*********************************************************************************************************
*                         BSP_SoundPlay (const CPU_INT16U *pusSong, CPU_INT32U ulLength)
*
* Description : This function configures the I2S peripheral in preparation for playing
*               and recording audio data of a particular format.
*
* Argument(s) : ulSampleRate is the sample rate of the audio to be played in
*                 samples per second.
*               usBitsPerSample is the number of bits in each audio sample.
*               usChannels is the number of audio channels, 1 for mono, 2 for stereo.
*
* Return(s)   : None.
*
* Caller(s)   : Main application.
*
* Note(s)     : None.
*********************************************************************************************************
*/
void  BSP_SoundSetFormat (CPU_INT32U ulSampleRate, CPU_INT16U usBitsPerSample,
                          CPU_INT16U usChannels)
{
    CPU_INT32U ulFormat;
    CPU_INT32U ulDMASetting;
    CPU_INT32U ulI2SErrata;

    // Save these values for use when configuring I2S.
    g_usChannels = usChannels;
    g_usBitsPerSample = usBitsPerSample;

    I2SMasterClockSelect(I2S0_BASE, 0);

    // Always use have the controller be an I2S Master.
    ulFormat = I2S_CONFIG_FORMAT_I2S | I2S_CONFIG_CLK_MASTER;
    
    // Check if the missing divisor bits need to be taken into account.
    if(CLASS_IS_TEMPEST && REVISION_IS_B1)
    {
        ulI2SErrata = 1;
    }
    else
    {    
        ulI2SErrata = 0;
    }    

    // Mono or Stereo formats.
    if(g_usChannels == 1)
    {
        // 8 bit formats.
        if(g_usBitsPerSample == 8)
        {
            // On Tempest class devices rev B parts the divisor is
            // limited for lower samples rates (see errata).
            if((ulI2SErrata != 0) && (ulSampleRate < 24400))
            {
                ulFormat |= I2S_CONFIG_WIRE_SIZE_32 | I2S_CONFIG_MODE_MONO |
                            I2S_CONFIG_SAMPLE_SIZE_8;
                usBitsPerSample = 32;
            }
            else
            {
                ulFormat |= I2S_CONFIG_WIRE_SIZE_8 | I2S_CONFIG_MODE_MONO |
                            I2S_CONFIG_SAMPLE_SIZE_8;
            }
        }
        else if(g_usBitsPerSample == 16)
        {
            // On Tempest class devices rev B parts the divisor is
            // limited for lower samples rates (see errata).
            if((ulI2SErrata != 0) && (ulSampleRate < 12200))
            {
                ulFormat |= I2S_CONFIG_WIRE_SIZE_32 | I2S_CONFIG_MODE_MONO |
                            I2S_CONFIG_SAMPLE_SIZE_16;

                usBitsPerSample = 32;
            }
            else
            {
                ulFormat |= I2S_CONFIG_WIRE_SIZE_16 | I2S_CONFIG_MODE_MONO |
                            I2S_CONFIG_SAMPLE_SIZE_16;
            }
        }
        else if(g_usBitsPerSample == 24)
        {
            ulFormat |= I2S_CONFIG_WIRE_SIZE_24 | I2S_CONFIG_MODE_MONO |
                        I2S_CONFIG_SAMPLE_SIZE_24;
        }
        else
        {
            ulFormat |= I2S_CONFIG_WIRE_SIZE_32 | I2S_CONFIG_MODE_MONO |
                        I2S_CONFIG_SAMPLE_SIZE_32;
        }
    }
    else
    {
        if(g_usBitsPerSample == 8)
        {
            // On Tempest class devices rev B parts the divisor is
            // limited for lower samples rates (see errata).
             if((ulI2SErrata != 0) && (ulSampleRate < 12200))
            {
                ulFormat |= I2S_CONFIG_WIRE_SIZE_32 |
                            I2S_CONFIG_MODE_COMPACT_8 |
                            I2S_CONFIG_SAMPLE_SIZE_8;

                usBitsPerSample = 32;
            }
            else
            {
                ulFormat |= I2S_CONFIG_WIRE_SIZE_8 |
                            I2S_CONFIG_MODE_COMPACT_8 |
                            I2S_CONFIG_SAMPLE_SIZE_8;
            }

        }
        else if(g_usBitsPerSample == 16)
        {
            if((ulI2SErrata != 0) && (ulSampleRate < 12200))
            {
                ulFormat |= I2S_CONFIG_WIRE_SIZE_32 |
                            I2S_CONFIG_MODE_COMPACT_16 |
                            I2S_CONFIG_SAMPLE_SIZE_16;
                usBitsPerSample = 32;
            }
            else
            {
                ulFormat |= I2S_CONFIG_WIRE_SIZE_16 |
                            I2S_CONFIG_MODE_COMPACT_16 |
                            I2S_CONFIG_SAMPLE_SIZE_16;
            }
        }
        else if(g_usBitsPerSample == 24)
        {
            ulFormat |= I2S_CONFIG_WIRE_SIZE_24 | I2S_CONFIG_MODE_DUAL |
                        I2S_CONFIG_SAMPLE_SIZE_24;
        }
        else
        {
            ulFormat |= I2S_CONFIG_WIRE_SIZE_32 | I2S_CONFIG_MODE_DUAL |
                        I2S_CONFIG_SAMPLE_SIZE_32;
        }
    }

    // Configure the I2S TX format.
    I2STxConfigSet(I2S0_BASE, ulFormat);
    ulFormat = (ulFormat & ~I2S_CONFIG_FORMAT_MASK) | I2S_CONFIG_FORMAT_LEFT_JUST;

    // Internally both are masters but the pins may not be driven out.
    I2SMasterClockSelect(I2S0_BASE, I2S_TX_MCLK_INT);

    // Set the MCLK rate and save it for conversion back to sample rate.
    // The multiply by 8 is due to a 4X oversample rate plus a factor of two
    // since the data is always stereo on the I2S interface.
    g_ulSampleRate = SysCtlI2SMClkSet(0, ulSampleRate * usBitsPerSample * 8);

    // Convert the MCLK rate to sample rate.
    g_ulSampleRate = g_ulSampleRate / (usBitsPerSample * 8);

    // Configure the I2S TX DMA channel to use high priority burst transfer.
    uDMAChannelAttributeEnable(UDMA_CHANNEL_I2S0TX,
                               (UDMA_ATTR_USEBURST |
                                UDMA_ATTR_HIGH_PRIORITY));

    // Set the DMA channel configuration.
    if(g_usChannels == 1)
    {
        // Handle Mono formats.
        if(g_usBitsPerSample == 8)
        {
            // The transfer size is 8 bits from the TX buffer to the TX FIFO.
            ulDMASetting = UDMA_SIZE_8 | UDMA_SRC_INC_8 |
                           UDMA_DST_INC_NONE | UDMA_ARB_2;
        }
        else
        {
            // The transfer size is 16 bits from the TX buffer to the TX FIFO.
            ulDMASetting = UDMA_SIZE_16 | UDMA_SRC_INC_16 |
                           UDMA_DST_INC_NONE | UDMA_ARB_2;
        }
    }
    else
    {
        // Handle Stereo formats.
        if(g_usBitsPerSample == 8)
        {
            // The transfer size is 16 bits(stereo 8 bits) from the TX buffer
            // to the TX FIFO.
            ulDMASetting = UDMA_SIZE_16 | UDMA_SRC_INC_16 |
                           UDMA_DST_INC_NONE | UDMA_ARB_2;
        }
        else
        {
            // The transfer size is 32 bits(stereo 16 bits) from the TX buffer
            // to the TX FIFO.
            ulDMASetting = UDMA_SIZE_32 | UDMA_SRC_INC_32 |
                           UDMA_DST_INC_NONE | UDMA_ARB_2;
        }
    }

    // Configure the DMA settings for this channel.
    uDMAChannelControlSet(UDMA_CHANNEL_I2S0TX | UDMA_PRI_SELECT, ulDMASetting);
    uDMAChannelControlSet(UDMA_CHANNEL_I2S0TX | UDMA_ALT_SELECT, ulDMASetting);
}

/*
*********************************************************************************************************
*                         BSP_SoundSampleRateGet (void)
*
* Description : This function returns the sample rate that was set by a call to
*               BSP_SoundSetFormat().  This is needed to retrieve the exact sample rate that is
*               in use in case the requested rate could not be matched exactly.
*
* Argument(s) : None.
*
* Return(s)   : Current sample rate.
*
* Caller(s)   : Main application.
*
* Note(s)     : None.
*********************************************************************************************************
*/
CPU_INT32U  BSP_SoundSampleRateGet (void)
{
    return(g_ulSampleRate);
}

/*
*********************************************************************************************************
*                         BSP_SoundBufferPlay (const void *pvData, CPU_INT32U ulLength,
*                                              tBufferCallback pfnCallback)
*
* Description : This function starts the playback of a block of PCM audio samples.  If
*               playback of another buffer is currently ongoing, its playback is canceled
*               and the buffer starts playing immediately.
*
* Argument(s) : pvData is a pointer to the audio data to play.
*               ulLength is the length of the data in bytes.
*               pfnCallback is a function to call when this buffer has be played.
*
* Return(s)   : Return 0 if the buffer was accepted, returns non-zero if there was no
*               space available for this buffer.
*
* Caller(s)   : Main application.
*
* Note(s)     : None.
*********************************************************************************************************
*/
CPU_INT32U  BSP_SoundBufferPlay (const void *pvData, CPU_INT32U ulLength,
                                 tBufferCallback pfnCallback)
{
    CPU_INT32U ulChannel;

    // Must disable I2S interrupts during this time to prevent state problems.
    BSP_IntDis(BSP_INT_ID_I2S0);
    
    // Save the buffer information.
    g_sOutBuffers[g_ulPlaying].pulData = (CPU_INT32U *)pvData;
    g_sOutBuffers[g_ulPlaying].ulSize = ulLength;
    g_sOutBuffers[g_ulPlaying].pfnBufferCallback = pfnCallback;

    // Handle which half of the ping-pong DMA is in use.
    if(g_ulPlaying)
    {
        ulChannel = UDMA_CHANNEL_I2S0TX | UDMA_ALT_SELECT;
    }
    else
    {
        ulChannel = UDMA_CHANNEL_I2S0TX | UDMA_PRI_SELECT;
    }

    // Set the DMA channel configuration.
    if(g_usChannels == 1)
    {
        // Handle Mono formats.
        if(g_usBitsPerSample == 16)
        {
            // The transfer size is 16 bits from the TX buffer to the TX FIFO.
            // Modify the DMA transfer size at it is units not bytes.
            g_sOutBuffers[g_ulPlaying].ulSize >>= 1;
        }
    }
    else
    {
        // Handle Stereo formats.
        if(g_usBitsPerSample == 8)
        {
            // The transfer size is 16 bits(stereo 8 bits) from the TX buffer
            // to the TX FIFO.  Modify the DMA transfer size at it is units
            // not bytes.
            g_sOutBuffers[g_ulPlaying].ulSize >>= 1;
        }
        else
        {
            // The transfer size is 32 bits(stereo 16 bits) from the TX buffer
            // to the TX FIFO. Modify the DMA transfer size at it is units not
            // bytes.
            g_sOutBuffers[g_ulPlaying].ulSize >>= 2;
        }
    }

    // Set the addresses and the DMA mode to ping-pong.
    uDMAChannelTransferSet(ulChannel,
                           UDMA_MODE_PINGPONG,
                           (CPU_INT32U *)g_sOutBuffers[g_ulPlaying].pulData,
                           (void *)(I2S0_BASE + I2S_O_TXFIFO),
                           g_sOutBuffers[g_ulPlaying].ulSize);

    // Enable the TX channel.  At this point the uDMA controller will
    // start servicing the request from the I2S, and the transmit side
    // should start running.
    uDMAChannelEnable(UDMA_CHANNEL_I2S0TX);

    // Indicate that there is still a pending transfer.
    HWREGBITW(&g_ulDMAFlags, FLAG_TX_PENDING) = 1;

    // Toggle which ping-pong DMA setting is in use.
    g_ulPlaying ^= 1;

    // Enable the I2S controller to start transmitting.
    I2STxEnable(I2S0_BASE);
   
    // Re-enable I2S interrupts.
    BSP_IntEn(BSP_INT_ID_I2S0);

    return(0);
}

/*
*********************************************************************************************************
*                         BSP_SoundVolumeSet (CPU_INT32U ulPercent)
*
* Description : This function sets the volume of the sound output to a value between
*               silence (0%) and full volume (100%).
*
* Argument(s) : ulPercent is the volume percentage, which must be between 0% (silence) and 100% (full volume), inclusive.
*
* Return(s)   : None.
*
* Caller(s)   : Main application.
*
* Note(s)     : None.
*********************************************************************************************************
*/
void  BSP_SoundVolumeSet (CPU_INT32U ulPercent)
{
    BSP_DACVolumeSet(ulPercent);
}

/*
*********************************************************************************************************
*                         BSP_SoundVolumeDown (CPU_INT32U ulPercent)
*
* Description : This function adjusts the audio output down by the specified percentage.
*               The adjusted volume will not go below 0% (silence).
*
* Argument(s) : ulPercent is the amount to decrease the volume, specified as a
*               percentage between 0% (silence) and 100% (full volume)
*
* Return(s)   : None.
*
* Caller(s)   : Main application.
*
* Note(s)     : None.
*********************************************************************************************************
*/
void  BSP_SoundVolumeDown (CPU_INT32U ulPercent)
{
    // Do not let the volume go below 0%.
    if(g_ucVolume < ulPercent)
    {
        // Set the volume to the minimum.
        g_ucVolume = 0;
    }
    else
    {
        // Decrease the volume by the specified amount.
        g_ucVolume -= ulPercent;
    }

    // Set the new volume.
    BSP_SoundVolumeSet(g_ucVolume);
}

/*
*********************************************************************************************************
*                         BSP_SoundVolumeGet (void)
*
* Description : This function returns the current volume, specified as a percentage between
*               0% (silence) and 100% (full volume), inclusive.
*
* Argument(s) : None.
*
* Return(s)   : Returns the current volume.
*
* Caller(s)   : Main application.
*
* Note(s)     : None.
*********************************************************************************************************
*/
CPU_INT08U  BSP_SoundVolumeGet (void)
{
    // Return the current Audio Volume.
    return(g_ucVolume);
}

/*
*********************************************************************************************************
*                         BSP_SoundVolumeUp (CPU_INT32U ulPercent)
*
* Description : This function adjusts the audio output up by the specified percentage.  The
*               adjusted volume will not go above 100% (full volume).
*
* Argument(s) : ulPercent is the amount to increase the volume, specified as a
*               percentage between 0% (silence) and 100% (full volume), inclusive.
*
* Return(s)   : None.
*
* Caller(s)   : Main application.
*
* Note(s)     : None.
*********************************************************************************************************
*/
void  BSP_SoundVolumeUp (CPU_INT32U ulPercent)
{
    // Increase the volume by the specified amount.
    g_ucVolume += ulPercent;

    // Do not let the volume go above 100%.
    if(g_ucVolume > 100)
    {
        // Set the volume to the maximum.
        g_ucVolume = 100;
    }

    // Set the new volume.
    BSP_SoundVolumeSet(g_ucVolume);
}

/*
*********************************************************************************************************
*                         BSP_SoundClassDEn (void)
*
* Description : This function enables the class D amplifier in the DAC.
*
* Argument(s) : None.
*
* Return(s)   : None.
*
* Caller(s)   : Main application.
*
* Note(s)     : None.
*********************************************************************************************************
*/
void  BSP_SoundClassDEn (void)
{
    BSP_DACClassDEn(); 
}

/*
*********************************************************************************************************
*                         BSP_SoundClassDDis (void)
*
* Description : This function disables the class D amplifier in the DAC.
*
* Argument(s) : None.
*
* Return(s)   : None.
*
* Caller(s)   : Main application.
*
* Note(s)     : None.
*********************************************************************************************************
*/
void  BSP_SoundClassDDis (void)
{
    BSP_DACClassDDis();  
}
