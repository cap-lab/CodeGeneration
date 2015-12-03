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
* Filename      : bsp_wav.c
* Version       : V1.00
* Programmer(s) : EMO
*********************************************************************************************************
*/


/*
*********************************************************************************************************
*                                            INCLUDE FILES
*********************************************************************************************************
*/

#include <bsp_wav.h>

/*
*********************************************************************************************************
*                                            LOCAL DEFINES
*********************************************************************************************************
*/

//******************************************************************************
//
// Basic wav file RIFF header information used to open and read a wav file.
//
//******************************************************************************
#define RIFF_CHUNK_ID_RIFF      0x46464952
#define RIFF_CHUNK_ID_FMT       0x20746d66
#define RIFF_CHUNK_ID_DATA      0x61746164

#define RIFF_TAG_WAVE           0x45564157

#define RIFF_FORMAT_UNKNOWN     0x0000
#define RIFF_FORMAT_PCM         0x0001
#define RIFF_FORMAT_MSADPCM     0x0002
#define RIFF_FORMAT_IMAADPCM    0x0011


#define AUDIO_BUFFER_SIZE       4096

#define BUFFER_BOTTOM_EMPTY     0x00000001
#define BUFFER_TOP_EMPTY        0x00000002
#define BUFFER_PLAYING          0x00000004

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

// State information for keep track of time.
static CPU_INT32U g_ulBytesPlayed;
static CPU_INT32U g_ulNextUpdate;

// Buffer management and flags.
static CPU_INT08U g_pucBuffer[AUDIO_BUFFER_SIZE];
CPU_INT08U *g_pucDataPtr;
CPU_INT32U g_ulMaxBufferSize;

// Flags used in the g_ulFlags global variable.
static volatile CPU_INT32U g_ulFlags;

// Globals used to track playback position.
static CPU_INT32U g_ulBytesRemaining;
static CPU_INT16U g_usMinutes;
static CPU_INT16U g_usSeconds;

CPU_INT08S g_pcTime[40] = "";

/*
*********************************************************************************************************
*                                     LOCAL CONFIGURATION ERRORS
*********************************************************************************************************
*/

/*
*********************************************************************************************************
*                                BSP_BufferCallback (void *pvBuffer, CPU_INT32U ulEvent)
*
* Description : Handler for buffers being released.
*
* Argument(s) : none.
*
* Return(s)   : none.
*
* Caller(s)   : BSP_WavePlay().
*
* Note(s)     : none.
*********************************************************************************************************
*/
void  BSP_BufferCallback (void *pvBuffer, CPU_INT32U ulEvent)
{
    if(ulEvent & BUFFER_EVENT_FREE)
    {
        if(pvBuffer == g_pucBuffer)
        {
            // Flag if the first half is free.
            g_ulFlags |= BUFFER_BOTTOM_EMPTY;
        }
        else
        {
            // Flag if the second half is free.
            g_ulFlags |= BUFFER_TOP_EMPTY;
        }

        // Update the byte count.
        g_ulBytesPlayed += AUDIO_BUFFER_SIZE >> 1;
    }
}

/*
*********************************************************************************************************
*                           BSP_WaveConvert8Bit (CPU_INT08U *pucBuffer, CPU_INT32U ulSize)
*
* Description : Convert an 8 bit unsigned buffer to 8 bit signed buffer in place so that it
*               can be passed into the i2s playback.
*
* Argument(s) : pucBuffer is a pointer to the input data buffer.
*               ulSize is the size of the input data buffer.
*
* Return(s)   : none.
*
* Caller(s)   : BSP_WaveRead().
*
* Note(s)     : none.
*********************************************************************************************************
*/
static void  BSP_WaveConvert8Bit (CPU_INT08U *pucBuffer, CPU_INT32U ulSize)
{
    CPU_INT32U ulIdx;

    for(ulIdx = 0; ulIdx < ulSize; ulIdx++)
    {
        // In place conversion of 8 bit unsigned to 8 bit signed.
        *pucBuffer = ((short)(*pucBuffer)) - 128;
        pucBuffer++;
    }
}

/*
*********************************************************************************************************
*                           BSP_WaveOpen (CPU_INT32U *pulAddress, tWaveHeader *pWaveHeader)
*
* Description : This function can be used to test if a file is a wav file or not and will
*               also return the wav file header information in the pWaveHeader structure.
*
* Argument(s) : pulAddress is a pointer to the starting address of the wave file.
*               pWaveHeader is a pointer to the tWaveHeader data structure to be populated.
*
* Return(s)   : WAVE_OK on success.
*               WAVE_INVALID_RIFF if RIFF information is not supported.
*               WAVE_INVALID_CHUNK if chunk size is not supported.
*               WAVE_INVALID_FORMAT if file format is not supported.
*
* Caller(s)   : Main applicaiton.
*
* Note(s)     : none.
*********************************************************************************************************
*/
tWaveReturnCode  BSP_WaveOpen (CPU_INT32U *pulAddress, tWaveHeader *pWaveHeader)
{
    CPU_INT32U *pulBuffer;
    CPU_INT16U *pusBuffer;
    CPU_INT32U ulChunkSize;
    CPU_INT32U ulBytesPerSample;
    
    
    // Look for RIFF tag.
    pulBuffer = (CPU_INT32U *)pulAddress;
    
    if((pulBuffer[0] != RIFF_CHUNK_ID_RIFF) || (pulBuffer[2] != RIFF_TAG_WAVE))
    {
        return(WAVE_INVALID_RIFF);
    }

    if(pulBuffer[3] != RIFF_CHUNK_ID_FMT)
    {
        return(WAVE_INVALID_CHUNK);
    }

    // Read the format chunk size.
    ulChunkSize = pulBuffer[4];

    if(ulChunkSize > 16)
    {
        return(WAVE_INVALID_CHUNK);
    }

    //   // Read the next chunk header.
    pulBuffer = (CPU_INT32U *)&pulAddress[5];
    pusBuffer = (CPU_INT16U *)pulBuffer;
    
    pWaveHeader->usFormat = pusBuffer[0];
    pWaveHeader->usNumChannels =  pusBuffer[1];
    pWaveHeader->ulSampleRate = pulBuffer[1];
    pWaveHeader->ulAvgByteRate = pulBuffer[2];
    pWaveHeader->usBitsPerSample = pusBuffer[7];

    // Reset the byte count.
    g_ulBytesPlayed = 0;
    g_ulNextUpdate = 0;

    // Calculate the Maximum buffer size based on format.  There can only be
    // 1024 samples per ping pong buffer due to uDMA.
    ulBytesPerSample = (pWaveHeader->usBitsPerSample *
                        pWaveHeader->usNumChannels) >> 3;

    if(((AUDIO_BUFFER_SIZE >> 1) / ulBytesPerSample) > 1024)
    {
        // The maximum number of DMA transfers was more than 1024 so limit
        // it to 1024 transfers.
        g_ulMaxBufferSize = 1024 * ulBytesPerSample;
    }
    else
    {
        // The maximum number of DMA transfers was not more than 1024.
        g_ulMaxBufferSize = AUDIO_BUFFER_SIZE >> 1;
    }

    // Only mono and stereo supported.
    if(pWaveHeader->usNumChannels > 2)
    {
        return(WAVE_INVALID_FORMAT);
    }

    // Read the next chunk header.
    pulBuffer = (CPU_INT32U *)&pulAddress[5] + (ulChunkSize / 4);
    if(pulBuffer[0] != RIFF_CHUNK_ID_DATA)
    {
        return(WAVE_INVALID_CHUNK);
    }

    // Save the size of the data.
    pWaveHeader->ulDataSize = pulBuffer[1];

    g_usSeconds = pWaveHeader->ulDataSize / pWaveHeader->ulAvgByteRate;
    g_usMinutes = g_usSeconds / 60;
    g_usSeconds -= g_usMinutes * 60;
    
    g_pucDataPtr = (CPU_INT08U *)&pulBuffer[2];
    
    // Set the number of data bytes in the file.
    g_ulBytesRemaining = pWaveHeader->ulDataSize;

    // Adjust the average bit rate for 8 bit mono files.
    if((pWaveHeader->usNumChannels == 1) && (pWaveHeader->usBitsPerSample == 8))
    {
        pWaveHeader->ulAvgByteRate <<=1;
    }

    // Set the format of the playback in the sound driver.
    BSP_SoundSetFormat(pWaveHeader->ulSampleRate, pWaveHeader->usBitsPerSample,
                   pWaveHeader->usNumChannels);

    return(WAVE_OK);
}

/*
*********************************************************************************************************
*                           BSP_WaveStop (void)
*
* Description : This function will handle stopping the playback of audio.  It will not do
*               this immediately but will defer stopping audio at a later time.  This allows
*               this function to be called from an interrupt handler.
*
* Argument(s) : None.
*
* Return(s)   : None.
*
* Caller(s)   : Main applicaiton.
*
* Note(s)     : None.
*********************************************************************************************************
*/
void  BSP_WaveStop (void)
{
    // Stop playing audio.
    g_ulFlags &= ~BUFFER_PLAYING;
}

/*
*********************************************************************************************************
*                           BSP_WaveRead (tWaveHeader *pWaveHeader, CPU_INT08U *pucBuffer)
*
* Description : This function will handle reading the correct amount from the wav file and
*               will also handle converting 8 bit unsigned to 8 bit signed if necessary.
*
* Argument(s) : pWaveHeader is a pointer to the current wave file's header information.
*               pucBuffer is a pointer to the input data buffer.
*
* Return(s)   : The number of bytes read.
*
* Caller(s)   : BSP_WavePlay().
*
* Note(s)     : None.
*********************************************************************************************************
*/
static CPU_INT32U  BSP_WaveRead (tWaveHeader *pWaveHeader, CPU_INT08U *pucBuffer)
{
    int i;
    CPU_INT32U ulBytesToRead;

    // Either read a half buffer or just the bytes remaining if we are at the
    // end of the file.
    if(g_ulBytesRemaining < g_ulMaxBufferSize)
    {
        ulBytesToRead = g_ulBytesRemaining;
    }
    else
    {
        ulBytesToRead = g_ulMaxBufferSize;
    }
    
    // Copy data from flash to SRAM.  This is needed in case 8-bit audio is 
    // used.  In this case, the data must be converted from unsigned to signed.
    for(i = 0; i < ulBytesToRead; i++)
    {
        pucBuffer[i] = g_pucDataPtr[i];  
    }
    
    // Decrement the number of data bytes remaining to be read.
    g_ulBytesRemaining -= ulBytesToRead;
    
    // Update the global data pointer keeping track of the location in flash.
    g_pucDataPtr += ulBytesToRead;

    // Need to convert the audio from unsigned to signed if 8 bit
    // audio is used.
    if(pWaveHeader->usBitsPerSample == 8)
    {
        BSP_WaveConvert8Bit(pucBuffer, ulBytesToRead);
    }

    return(ulBytesToRead);
}

/*
*********************************************************************************************************
*                           BSP_WavePlay (tWaveHeader *pWaveHeader)
*
* Description : This will play the file passed in via the psFileObject parameter based on
*               the format passed in the pWaveHeader structure.  The WaveOpen() function
*               can be used to properly fill the pWaveHeader and psFileObject structures.
*
* Argument(s) : pWaveHeader is a pointer to the current wave file's header information.
*
* Return(s)   : None.
*
* Caller(s)   : Main application.
*
* Note(s)     : None.
*********************************************************************************************************
*/
void  BSP_WavePlay (tWaveHeader *pWaveHeader)
{
    CPU_INT32U ulCount;
    OS_ERR err;

    // Mark both buffers as empty.
    g_ulFlags = BUFFER_BOTTOM_EMPTY | BUFFER_TOP_EMPTY;

    // Indicate that the application is about to start playing.
    g_ulFlags |= BUFFER_PLAYING; 
    
    // Enable the Class D amp.  It's turned off when idle to save power.
    BSP_SoundClassDEn();

    while(1)
    {
        // Must disable I2S interrupts during this time to prevent state problems.
        BSP_IntDis(BSP_INT_ID_I2S0);

        // If the refill flag gets cleared then fill the requested side of the
        // buffer.
        if(g_ulFlags & BUFFER_BOTTOM_EMPTY)
        {
            // Read out the next buffer worth of data.
            ulCount = BSP_WaveRead(pWaveHeader, g_pucBuffer);  
            
            // Sleep this task for 40ms to allow other tasks to perform their duties.
            //OSTimeDlyHMSM(0, 0, 0, 40,
            //      OS_OPT_TIME_HMSM_STRICT,
            //      &err);  
            
            // Start the playback for a new buffer.
            BSP_SoundBufferPlay(g_pucBuffer, ulCount, BSP_BufferCallback);

            // Bottom half of the buffer is now not empty.
            g_ulFlags &= ~BUFFER_BOTTOM_EMPTY;
        }
        
        if(g_ulFlags & BUFFER_TOP_EMPTY)
        {
            // Read out the next buffer worth of data.
            ulCount = BSP_WaveRead(pWaveHeader, &g_pucBuffer[AUDIO_BUFFER_SIZE >> 1]);

            // Sleep this task for 40ms to allow other tasks to perform their duties.
            //OSTimeDlyHMSM(0, 0, 0, 40,
            //      OS_OPT_TIME_HMSM_STRICT,
            //      &err);              
            
            // Start the playback for a new buffer.
            BSP_SoundBufferPlay(&g_pucBuffer[AUDIO_BUFFER_SIZE >> 1],
                            ulCount, BSP_BufferCallback);

            // Top half of the buffer is now not empty.
            g_ulFlags &= ~BUFFER_TOP_EMPTY;
        }

        // If something reset this while playing then stop playing and break
        // out of the loop.
        if((g_ulFlags & BUFFER_PLAYING) == 0)
        {
            break; 
        }
        
        // Audio playback is done once the count is below a full buffer.
        if((ulCount < g_ulMaxBufferSize) || (g_ulBytesRemaining == 0))
        {
            // No longer playing audio.
            g_ulFlags &= ~BUFFER_PLAYING;

            // Wait for the buffer to empty.
            while(g_ulFlags != (BUFFER_TOP_EMPTY | BUFFER_BOTTOM_EMPTY))
            {
            }

            break;
        }        
        
        // Must disable I2S interrupts during this time to prevent state
        // problems.
        BSP_IntEn(BSP_INT_ID_I2S0);  
        
            // Sleep this task for 40ms to allow other tasks to perform their duties.
            OSTimeDlyHMSM(0, 0, 0, 40,
                  OS_OPT_TIME_HMSM_STRICT,
                  &err);          
    }  
    
    // Disable the Class D amp to save power.
    BSP_SoundClassDDis();
}

/*
*********************************************************************************************************
*                          BSP_WaveDisplayTime (tWaveHeader *pWaveHeader, CPU_INT32U ulForceUpdate)
*
* Description : This function is used to tell when to update the playback times for a file.
*               It will only update the screen at 1 second intervals but can be called more
*               often with no result.
*
* Argument(s) : pWaveHeader is a pointer to the current wave file's header information.
*               ulForceUpdate determines whether the update to the screen is forced on each function call,
                  or deferred to once per second.
*
* Return(s)   : None.
*
* Caller(s)   : Main application.
*
* Note(s)     : None.
*********************************************************************************************************
*/
void  BSP_WaveDisplayTime (tWaveHeader *pWaveHeader, CPU_INT32U ulForceUpdate)
{
    CPU_INT32U ulSeconds;
    CPU_INT32U ulMinutes;

    // Only display on the screen once per second.
    if((g_ulBytesPlayed >= g_ulNextUpdate) || (ulForceUpdate != 0))
    {
        // Set the next update time to one second later.
        g_ulNextUpdate = g_ulBytesPlayed + pWaveHeader->ulAvgByteRate;

        // Calculate the integer number of minutes and seconds.
        ulSeconds = g_ulBytesPlayed / pWaveHeader->ulAvgByteRate;
        ulMinutes = ulSeconds / 60;
        ulSeconds -= ulMinutes * 60;
        
        // If for some reason the seconds go over, clip to the right size.
        if(ulSeconds > g_usSeconds)
        {
            ulSeconds = g_usSeconds; 
        }

        // Print the time string in the format mm.ss/mm.ss
        sprintf((char *)g_pcTime, "%d:%02d/%d:%02d", ulMinutes, ulSeconds,
                 g_usMinutes, g_usSeconds);
        
        // Update the time on the display.
        BSP_DisplayStringDraw(g_pcTime, 18, 1);
    }  
}
/*
*********************************************************************************************************
*                          BSP_WavePlaybackStatus (void)
*
* Description : Returns the current playback status of the wave file.
*
* Argument(s) : None.
*
* Return(s)   : Current playback status - BUFFER_PLAYING (0x4) for play, 0 for stop.
*
* Caller(s)   : Main application.
*
* Note(s)     : None.
*********************************************************************************************************
*/
CPU_INT32U  BSP_WavePlaybackStatus (void)
{
    return(g_ulFlags & BUFFER_PLAYING); 
}


