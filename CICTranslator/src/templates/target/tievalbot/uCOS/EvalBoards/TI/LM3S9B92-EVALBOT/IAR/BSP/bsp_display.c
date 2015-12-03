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
*                                          DISPLAY SERVICES
*
*                             TEXAS INSTRUMENTS LM3S9B90 on the EK-LM3S9B90
*
* Filename      : bsp_display.c
* Version       : V1.00
* Programmer(s) : EMO
*********************************************************************************************************
*/


/*
*********************************************************************************************************
*                                            INCLUDE FILES
*********************************************************************************************************
*/

#define   BSP_DISPLAY_MODULE
#include <bsp_display.h>


/*
*********************************************************************************************************
*                                            LOCAL DEFINES
*********************************************************************************************************
*/

#define SIZE_CURSOR_ROW_COMMAND     6
#define SSD_ADDR                    0x3c

/*
*********************************************************************************************************
*                                           LOCAL CONSTANTS
*********************************************************************************************************
*/

// A 5x7 font (in a 6x8 cell, where the sixth column is omitted from this
// table) for displaying text on the OLED display.  The data is organized as
// bytes from the left column to the right column, with each byte containing
// the top row in the LSB and the bottom row in the MSB.
static const CPU_INT08U g_pucFont[95][5] =
{
    { 0x00, 0x00, 0x00, 0x00, 0x00 }, // " "
    { 0x00, 0x00, 0x4f, 0x00, 0x00 }, // !
    { 0x00, 0x07, 0x00, 0x07, 0x00 }, // "
    { 0x14, 0x7f, 0x14, 0x7f, 0x14 }, // #
    { 0x24, 0x2a, 0x7f, 0x2a, 0x12 }, // $
    { 0x23, 0x13, 0x08, 0x64, 0x62 }, // %
    { 0x36, 0x49, 0x55, 0x22, 0x50 }, // &
    { 0x00, 0x05, 0x03, 0x00, 0x00 }, // '
    { 0x00, 0x1c, 0x22, 0x41, 0x00 }, // (
    { 0x00, 0x41, 0x22, 0x1c, 0x00 }, // )
    { 0x14, 0x08, 0x3e, 0x08, 0x14 }, // *
    { 0x08, 0x08, 0x3e, 0x08, 0x08 }, // +
    { 0x00, 0x50, 0x30, 0x00, 0x00 }, // ,
    { 0x08, 0x08, 0x08, 0x08, 0x08 }, // -
    { 0x00, 0x60, 0x60, 0x00, 0x00 }, // .
    { 0x20, 0x10, 0x08, 0x04, 0x02 }, // /
    { 0x3e, 0x51, 0x49, 0x45, 0x3e }, // 0
    { 0x00, 0x42, 0x7f, 0x40, 0x00 }, // 1
    { 0x42, 0x61, 0x51, 0x49, 0x46 }, // 2
    { 0x21, 0x41, 0x45, 0x4b, 0x31 }, // 3
    { 0x18, 0x14, 0x12, 0x7f, 0x10 }, // 4
    { 0x27, 0x45, 0x45, 0x45, 0x39 }, // 5
    { 0x3c, 0x4a, 0x49, 0x49, 0x30 }, // 6
    { 0x01, 0x71, 0x09, 0x05, 0x03 }, // 7
    { 0x36, 0x49, 0x49, 0x49, 0x36 }, // 8
    { 0x06, 0x49, 0x49, 0x29, 0x1e }, // 9
    { 0x00, 0x36, 0x36, 0x00, 0x00 }, // :
    { 0x00, 0x56, 0x36, 0x00, 0x00 }, // ;
    { 0x08, 0x14, 0x22, 0x41, 0x00 }, // <
    { 0x14, 0x14, 0x14, 0x14, 0x14 }, // =
    { 0x00, 0x41, 0x22, 0x14, 0x08 }, // >
    { 0x02, 0x01, 0x51, 0x09, 0x06 }, // ?
    { 0x32, 0x49, 0x79, 0x41, 0x3e }, // @
    { 0x7e, 0x11, 0x11, 0x11, 0x7e }, // A
    { 0x7f, 0x49, 0x49, 0x49, 0x36 }, // B
    { 0x3e, 0x41, 0x41, 0x41, 0x22 }, // C
    { 0x7f, 0x41, 0x41, 0x22, 0x1c }, // D
    { 0x7f, 0x49, 0x49, 0x49, 0x41 }, // E
    { 0x7f, 0x09, 0x09, 0x09, 0x01 }, // F
    { 0x3e, 0x41, 0x49, 0x49, 0x7a }, // G
    { 0x7f, 0x08, 0x08, 0x08, 0x7f }, // H
    { 0x00, 0x41, 0x7f, 0x41, 0x00 }, // I
    { 0x20, 0x40, 0x41, 0x3f, 0x01 }, // J
    { 0x7f, 0x08, 0x14, 0x22, 0x41 }, // K
    { 0x7f, 0x40, 0x40, 0x40, 0x40 }, // L
    { 0x7f, 0x02, 0x0c, 0x02, 0x7f }, // M
    { 0x7f, 0x04, 0x08, 0x10, 0x7f }, // N
    { 0x3e, 0x41, 0x41, 0x41, 0x3e }, // O
    { 0x7f, 0x09, 0x09, 0x09, 0x06 }, // P
    { 0x3e, 0x41, 0x51, 0x21, 0x5e }, // Q
    { 0x7f, 0x09, 0x19, 0x29, 0x46 }, // R
    { 0x46, 0x49, 0x49, 0x49, 0x31 }, // S
    { 0x01, 0x01, 0x7f, 0x01, 0x01 }, // T
    { 0x3f, 0x40, 0x40, 0x40, 0x3f }, // U
    { 0x1f, 0x20, 0x40, 0x20, 0x1f }, // V
    { 0x3f, 0x40, 0x38, 0x40, 0x3f }, // W
    { 0x63, 0x14, 0x08, 0x14, 0x63 }, // X
    { 0x07, 0x08, 0x70, 0x08, 0x07 }, // Y
    { 0x61, 0x51, 0x49, 0x45, 0x43 }, // Z
    { 0x00, 0x7f, 0x41, 0x41, 0x00 }, // [
    { 0x02, 0x04, 0x08, 0x10, 0x20 }, // "\"
    { 0x00, 0x41, 0x41, 0x7f, 0x00 }, // ]
    { 0x04, 0x02, 0x01, 0x02, 0x04 }, // ^
    { 0x40, 0x40, 0x40, 0x40, 0x40 }, // _
    { 0x00, 0x01, 0x02, 0x04, 0x00 }, // `
    { 0x20, 0x54, 0x54, 0x54, 0x78 }, // a
    { 0x7f, 0x48, 0x44, 0x44, 0x38 }, // b
    { 0x38, 0x44, 0x44, 0x44, 0x20 }, // c
    { 0x38, 0x44, 0x44, 0x48, 0x7f }, // d
    { 0x38, 0x54, 0x54, 0x54, 0x18 }, // e
    { 0x08, 0x7e, 0x09, 0x01, 0x02 }, // f
    { 0x0c, 0x52, 0x52, 0x52, 0x3e }, // g
    { 0x7f, 0x08, 0x04, 0x04, 0x78 }, // h
    { 0x00, 0x44, 0x7d, 0x40, 0x00 }, // i
    { 0x20, 0x40, 0x44, 0x3d, 0x00 }, // j
    { 0x7f, 0x10, 0x28, 0x44, 0x00 }, // k
    { 0x00, 0x41, 0x7f, 0x40, 0x00 }, // l
    { 0x7c, 0x04, 0x18, 0x04, 0x78 }, // m
    { 0x7c, 0x08, 0x04, 0x04, 0x78 }, // n
    { 0x38, 0x44, 0x44, 0x44, 0x38 }, // o
    { 0x7c, 0x14, 0x14, 0x14, 0x08 }, // p
    { 0x08, 0x14, 0x14, 0x18, 0x7c }, // q
    { 0x7c, 0x08, 0x04, 0x04, 0x08 }, // r
    { 0x48, 0x54, 0x54, 0x54, 0x20 }, // s
    { 0x04, 0x3f, 0x44, 0x40, 0x20 }, // t
    { 0x3c, 0x40, 0x40, 0x20, 0x7c }, // u
    { 0x1c, 0x20, 0x40, 0x20, 0x1c }, // v
    { 0x3c, 0x40, 0x30, 0x40, 0x3c }, // w
    { 0x44, 0x28, 0x10, 0x28, 0x44 }, // x
    { 0x0c, 0x50, 0x50, 0x50, 0x3c }, // y
    { 0x44, 0x64, 0x54, 0x4c, 0x44 }, // z
    { 0x00, 0x08, 0x36, 0x41, 0x00 }, // {
    { 0x00, 0x00, 0x7f, 0x00, 0x00 }, // |
    { 0x00, 0x41, 0x36, 0x08, 0x00 }, // }
    { 0x02, 0x01, 0x02, 0x04, 0x02 }, // ~
};


// The sequence of commands used to initialize the SSD1300 controller.
static const CPU_INT08U g_pucRITInit[] =
{
    // Turn off the panel
    0x04, 0x80, 0xae, 0x80, 0xe3,

    // Internal dc/dc on/off
    0x06, 0x80, 0xad, 0x80, 0x8a, 0x80, 0xe3,

    // Multiplex ratio
    0x06, 0x80, 0xa8, 0x80, 0x1f, 0x80, 0xe3,

    // COM out scan direction
    0x04, 0x80, 0xc8, 0x80, 0xe3,

    // Segment map
    0x04, 0x80, 0xa0, 0x80, 0xe3,


    // Set area color mode
    0x04, 0x80, 0xd8, 0x80, 0xe3,

    // Low power save mode
    0x04, 0x80, 0x05, 0x80, 0xe3,

    // Start line
    0x04, 0x80, 0x40, 0x80, 0xe3,

    // Contrast setting
    0x06, 0x80, 0x81, 0x80, 0x5d, 0x80, 0xe3,

    // Pre-charge/discharge
    0x06, 0x80, 0xd9, 0x80, 0x11, 0x80, 0xe3,

    // Set display clock
    0x06, 0x80, 0xd5, 0x80, 0x01, 0x80, 0xe3,

    // Display offset
    0x06, 0x80, 0xd3, 0x80, 0x00, 0x80, 0xe3,

    // Display off
    0x04, 0x80, 0xaf, 0x80, 0xe3,
};

#define SIZE_INIT_CMDS (sizeof(g_pucRITInit))


// The sequence of commands used to set the cursor to the first column of the
// first and second rows of the display for each of the supported displays.
static const CPU_INT08U g_pucRow1[] =
{
    0xb0, 0x80, 0x04, 0x80, 0x10, 0x40
};
static const CPU_INT08U g_pucRow2[] =
{
    0xb1, 0x80, 0x04, 0x80, 0x10, 0x40
};

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

// The inter-byte delay required by the display OLED controller.
static CPU_INT32U g_ulDelay;
static CPU_INT32U g_ucColumnAdjust = 4;

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
*                               BSP_DisplayWriteFirst(CPU_INT08U ucChar)
*
* Description : Start a transfer to the SSD1300 controller.
*
* Argument(s) : ucChar is the first byte to be written to the controller.
*
* Return(s)   : none.
*
* Caller(s)   : Display driver.
*
* Note(s)     : This function will start a transfer to the display controller via the I2C bus.
*
*               The data is written in a polled fashion; this function will not return
*               until the byte has been written to the controller.
*********************************************************************************************************
*/
static void  BSP_DisplayWriteFirst (CPU_INT08U ucChar)
{
    // Set the slave address.
    I2CMasterSlaveAddrSet(I2C1_MASTER_BASE, SSD_ADDR, false);

    // Write the first byte to the controller.
    I2CMasterDataPut(I2C1_MASTER_BASE, ucChar);

    // Start the transfer.
    I2CMasterControl(I2C1_MASTER_BASE, I2C_MASTER_CMD_BURST_SEND_START);
}

/*
*********************************************************************************************************
*                              BSP_DisplayWriteByte (CPU_INT08U ucChar)
*
* Description : Write a byte to the  SSD1300 controller.
*
* Argument(s) : ucChar is the byte to be transmitted to the controller.
*
* Return(s)   : none.
*
* Caller(s)   : Display driver.
*
* Note(s)     : This function continues a transfer to the display controller by writing
*               another byte over the I2C bus.  This must only be called after calling
*               BSP_DisplayWriteFirst(), but before calling DisplayWriteFinal().
*               
*               The data is written in a polled faashion; this function will not return
*               until the byte has been written to the controller.
*********************************************************************************************************
*/
static void  BSP_DisplayWriteByte (CPU_INT08U ucChar)
{
    // Wait until the current byte has been transferred.
    while(I2CMasterIntStatus(I2C1_MASTER_BASE, false) == 0)
    {
    }

    // Provide the required inter-byte delay.
    SysCtlDelay(g_ulDelay);

    // Write the next byte to the controller.
    I2CMasterDataPut(I2C1_MASTER_BASE, ucChar);

    // Continue the transfer.
    I2CMasterControl(I2C1_MASTER_BASE, I2C_MASTER_CMD_BURST_SEND_CONT);
}

/*
*********************************************************************************************************
*                     BSP_DisplayWriteArray (const CPU_INT08U *pucBuffer, 
                                             CPU_INT32U ulCount)
*
* Description : Write a sequence of bytes to the SSD0303 or SD1300 controller.
*
* Argument(s) : pucBuffer is a pointer to the array of data bytes to be transfered to the controlller.
*
*               ulCount is the length of pucBuffer.
*
* Return(s)   : none.
*
* Caller(s)   : Display driver.
*
* Note(s)     : This function continues a transfer to the display controller by writing a
*               sequence of bytes over the I2C bus.  This must only be called after calling
*               BSP_DisplayWriteFirst(), but before calling BSP_DisplayWriteFinal().
*                
*               The data is written in a polled fashion; this function will not return
*               until the entire byte sequence has been written to the controller.
*********************************************************************************************************
*/
static void  BSP_DisplayWriteArray (const CPU_INT08U *pucBuffer, 
                                    CPU_INT32U ulCount)
{
    CPU_INT32U ulErrStatus;
    
    // Loop while there are more bytes left to be transferred.
    while(ulCount != 0)
    {
        // Wait until the current byte has been transferred.
        while(I2CMasterIntStatus(I2C1_MASTER_BASE, false) == 0)
        {
        }

        ulErrStatus = I2CMasterErr(I2C1_MASTER_BASE);
        if(ulErrStatus != I2C_MASTER_ERR_NONE)
        {
            //for(;;);
        }
        
        // Provide the required inter-byte delay.
        SysCtlDelay(g_ulDelay);

        // Write the next byte to the controller.
        I2CMasterDataPut(I2C1_MASTER_BASE, *pucBuffer++);
        ulCount--;

        // Continue the transfer.
        I2CMasterControl(I2C1_MASTER_BASE, I2C_MASTER_CMD_BURST_SEND_CONT);
    }
}

/*
*********************************************************************************************************
*                                   BSP_DisplayWriteFinal(CPU_INT08U ucChar)
*
* Description : Finish a transfer to the SD1300 controller.
*
* Argument(s) : ucChar is the final byte to be written to the controller.
*
* Return(s)   : none.
*
* Caller(s)   : Display driver.
*
* Note(s)     : This function will finish a transfer to the display controller via the I2C
*               bus.  This must only be called after calling BSP_DisplayWriteFirst().
*                
*               The data is written in a polled fashion; this function will not return
*               until the byte has been written to the controller.
*********************************************************************************************************
*/
static void  BSP_DisplayWriteFinal (CPU_INT08U ucChar)
{
    // Wait until the current byte has been transferred.
    while(I2CMasterIntStatus(I2C1_MASTER_BASE, false) == 0)
    {
    }

    // Provide the required inter-byte delay.
    SysCtlDelay(g_ulDelay);

    // Write the final byte to the controller.
    I2CMasterDataPut(I2C1_MASTER_BASE, ucChar);

    // Finish the transfer.
    I2CMasterControl(I2C1_MASTER_BASE, I2C_MASTER_CMD_BURST_SEND_FINISH);

    // Wait until the final byte has been transferred.
    while(I2CMasterIntStatus(I2C1_MASTER_BASE, false) == 0)
    {
    }

    // Provide the required inter-byte delay.
    SysCtlDelay(g_ulDelay);
}

/*
*********************************************************************************************************
*                                         BSP_DisplayClear (void)
*
* Description : Clears the OLED display.  All pixels in the display will be turned off.
*
* Argument(s) : none.
*
* Return(s)   : none.
*
* Caller(s)   : Application.
*
* Note(s)     : none.
*********************************************************************************************************
*/
void  BSP_DisplayClear (void)
{
    CPU_INT32U ulIdx;

    // Move the display cursor to the first column of the first row.
    BSP_DisplayWriteFirst(0x80);
    BSP_DisplayWriteArray(g_pucRow1, SIZE_CURSOR_ROW_COMMAND);

    // Fill this row with zeros.
    for(ulIdx = 0; ulIdx < 95; ulIdx++)
    {
        BSP_DisplayWriteByte(0x00);
    }
    
    BSP_DisplayWriteFinal(0x00);

    // Move the display cursor to the first column of the second row.
    BSP_DisplayWriteFirst(0x80);
    BSP_DisplayWriteArray(g_pucRow2, SIZE_CURSOR_ROW_COMMAND);

    // Fill this row with zeros.
    for(ulIdx = 0; ulIdx < 95; ulIdx++)
    {
        BSP_DisplayWriteByte(0x00);
    }
    
    BSP_DisplayWriteFinal(0x00);
}

/*
*********************************************************************************************************
*                            BSP_DisplayStringDraw (const CPU_INT08S *pcStr, 
                                           CPU_INT32U ulX, CPU_INT32U ulY)
*
* Description : Displays a string on the OLED display.
*
* Argument(s) : pcStr is a pointer to the string to display.
*
*               ulX is the horizontal position to display the string, specified in
*                 columns from the left edge of the display. 
*
*               ulY is the vertical position to display the string, specified in
*                 eight scan line blocks from the top of the display (that is, only 0 and 1
*                 are valid).
* Return(s)   : none.
*
* Caller(s)   : Application.
*
* Note(s)     : Only the ASCII characters between 32 (space) and 126 (tilde) are supported; 
*               other characters will result in random data being draw on the display (based 
*               on whatever appears before/after the font in memory).  The font is mono-spaced, 
*               so characters such as ``i'' and ``l'' have more white space around them than 
*               characters  such as ``m'' or ``w''.
*                
*               If the drawing of the string reaches the right edge of the display, no more
*               characters will be drawn.  Therefore, special care is not required to avoid
*               supplying a string that is ``too long'' to display.
*********************************************************************************************************
*/
void  BSP_DisplayStringDraw (const CPU_INT08S *pcStr, CPU_INT32U ulX, CPU_INT32U ulY)
{
    // Check the arguments.
    //ASSERT(ulX < 96);
    //ASSERT(ulY < 2);

    // Move the display cursor to the requested position on the display.
    BSP_DisplayWriteFirst(0x80);
    BSP_DisplayWriteByte((ulY == 0) ? 0xb0 : 0xb1);
    BSP_DisplayWriteByte(0x80);
    BSP_DisplayWriteByte((ulX + g_ucColumnAdjust) & 0x0f);
    BSP_DisplayWriteByte(0x80);
    BSP_DisplayWriteByte(0x10 | (((ulX + g_ucColumnAdjust) >> 4) & 0x0f));
    BSP_DisplayWriteByte(0x40);

    // Loop while there are more characters in the string.
    while(*pcStr != 0)
    {
        // See if there is enough space on the display for this entire
        // character.
        if(ulX <= 90)
        {
            // Write the contents of this character to the display.
            BSP_DisplayWriteArray(g_pucFont[*pcStr - ' '], 5);

            // See if this is the last character to display (either because the
            // right edge has been reached or because there are no more
            // characters).
            if((ulX == 90) || (pcStr[1] == 0))
            {
                // Write the final column of the display.
                BSP_DisplayWriteFinal(0x00);

                // The string has been displayed.
                return;
            }

            // Write the inter-character padding column.
            BSP_DisplayWriteByte(0x00);
        }
        else
        {
            // Write the portion of the character that will fit onto the
            // display.
            BSP_DisplayWriteArray(g_pucFont[*pcStr - ' '], 95 - ulX);
            BSP_DisplayWriteFinal(g_pucFont[*pcStr - ' '][95 - ulX]);

            // The string has been displayed.
            return;
        }

        // Advance to the next character.
        pcStr++;

        // Increment the X coordinate by the six columns that were just
        // written.
        ulX += 6;
    }
}
        
        
void BSP_printInt(short val, short pos, short line){
  char str[10];
  char result[10];
  int i, index = 0;
  int quotient = 0;
  int remainder = 0;
  
  quotient = val;
  
  while(1){
    remainder = quotient % 10;
    quotient = quotient / 10;
    
     switch (remainder){
     case 0:
      str[index] = '0';
    break;
     case 1:
      str[index] = '1';
    break;
     case 2:
      str[index] = '2';
    break;
     case 3:
      str[index] = '3';
    break;
     case 4:
      str[index] = '4';
    break;
    case 5:
      str[index] = '5';
    break;
    case 6:
      str[index] = '6';
    break;
    case 7:
      str[index] = '7';
    break;
    case 8:
      str[index] = '8';
    break;
    case 9:
      str[index] = '9';
    break;
    default:
      str[index] = '?';
    break;
     }

    index++;  
    if(quotient == 0)   break;
  }
  
  if(index == 1){
    str[1] = '0';
    index++;
  }
  
  for(i=0; i<index; i++){
    result[index - i - 1] = str[i];
  }
  
  BSP_DisplayStringDraw((CPU_INT08S const*)result, pos, line);
}


/*
*********************************************************************************************************
*                 BSP_DisplayImageDraw (const CPU_INT08U *pucImage, CPU_INT32U ulX,
                               CPU_INT32U ulY, CPU_INT32U ulWidth,
                               CPU_INT32U ulHeight)
*
* Description : Displays an image on the OLED display.
*
* Argument(s) : pucImage is a pointer to the image data.
*
*               ulX is the horizontal position to display this image, specified in
*                 columns from the left edge of the display.
*
*               ulY is the vertical position to display this image, specified in
*                 eight scan line blocks from the top of the display (that is, only 0 and 1
*                 are valid).
*
*               ulWidth is the width of the image, specified in columns.
*
*               ulHeight is the height of the image, specified in eight row blocks
 *                (that is, only 1 and 2 are valid).
*
* Return(s)   : none.
*
* Caller(s)   : Application.
*
* Note(s)     : none.
*********************************************************************************************************
*/
void  BSP_DisplayImageDraw (const CPU_INT08U *pucImage, CPU_INT32U ulX,
                               CPU_INT32U ulY, CPU_INT32U ulWidth,
                               CPU_INT32U ulHeight)
{
    // Check the arguments.
    //ASSERT(ulX < 96);
    //ASSERT(ulY < 2);
    //ASSERT((ulX + ulWidth) <= 96);
    //ASSERT((ulY + ulHeight) <= 2);

    // The first few columns of the LCD buffer are not displayed, so increment
    // the X coorddinate by this amount to account for the non-displayed frame
    // buffer memory.
    ulX += g_ucColumnAdjust;

    // Loop while there are more rows to display.
    while(ulHeight--)
    {
        // Write the starting address within this row.
        BSP_DisplayWriteFirst(0x80);
        BSP_DisplayWriteByte((ulY == 0) ? 0xb0 : 0xb1);
        BSP_DisplayWriteByte(0x80);
        BSP_DisplayWriteByte(ulX & 0x0f);
        BSP_DisplayWriteByte(0x80);
        BSP_DisplayWriteByte(0x10 | ((ulX >> 4) & 0x0f));
        BSP_DisplayWriteByte(0x40);

        // Write this row of image data.
        BSP_DisplayWriteArray(pucImage, ulWidth - 1);
        BSP_DisplayWriteFinal(pucImage[ulWidth - 1]);

        // Advance to the next row of the image.
        pucImage += ulWidth;
        ulY++;
    }
}

/*
*********************************************************************************************************
*                                           BSP_DisplayOn()
*
* Description : Turns on the OLED display.
*
* Argument(s) : none.
*
* Return(s)   : none.
*
* Caller(s)   : Application.
*
* Note(s)     : none.
*********************************************************************************************************
*/
void  BSP_DisplayOn (void)
{
    CPU_INT32U ulIdx;

    // Re-initialize the display controller.  Loop through the initialization
    // sequence doing a single I2C transfer for each command.
    for(ulIdx = 0; ulIdx < SIZE_INIT_CMDS;
        ulIdx += g_pucRITInit[ulIdx] + 1)
    {
        // Send this command.
        BSP_DisplayWriteFirst(g_pucRITInit[ulIdx + 1]);
        BSP_DisplayWriteArray(g_pucRITInit + ulIdx + 2,
                               g_pucRITInit[ulIdx] - 2);
        BSP_DisplayWriteFinal(g_pucRITInit[ulIdx + g_pucRITInit[ulIdx]]);
    }
}

/*
*********************************************************************************************************
*                                          BSP_DisplayOff()
*
* Description : Turns off the OLED display.
*
* Argument(s) : none.
*
* Return(s)   : none.
*
* Caller(s)   : Application.
*
* Note(s)     : none.
*********************************************************************************************************
*/
void  BSP_DisplayOff (void)
{
    // Turn off the DC-DC converter and the display.
    BSP_DisplayWriteFirst(0x80);
    BSP_DisplayWriteByte(0xae);
    BSP_DisplayWriteByte(0x80);
    BSP_DisplayWriteByte(0xad);
    BSP_DisplayWriteByte(0x80);
    BSP_DisplayWriteFinal(0x8a);
}

/*
*********************************************************************************************************
*                                             BSP_DisplayInit()
*
* Description : Initialize the board's display.
*
* Argument(s) : none.
*
* Return(s)   : none.
*
* Caller(s)   : Application.
*
* Note(s)     : none.
*********************************************************************************************************
*/
void  BSP_DisplayInit (void)
{
    CPU_INT32U ulIdx; 
    volatile unsigned int i;

    //
    // Configure pin muxing for I2C1
    //
    GPIOPinConfigure(GPIO_PG0_I2C1SCL);
    GPIOPinConfigure(GPIO_PG1_I2C1SDA);
    
    //
    // Deassert OLED reset signal
    //
    GPIOPinTypeGPIOOutput(GPIO_PORTF_BASE, GPIO_PIN_0);
    GPIOPinWrite(GPIO_PORTF_BASE, GPIO_PIN_0, GPIO_PIN_0);
    for (i=100; i ; i--);

    //
    // Assert OLED reset (PF0) signal for at least 3us
    //
    GPIOPinWrite(GPIO_PORTF_BASE, GPIO_PIN_0, 0);
    for (i=100; i ; i--);

    //
    // Deassert OLED reset signal
    //
    GPIOPinWrite(GPIO_PORTF_BASE, GPIO_PIN_0, GPIO_PIN_0);
    for (i=100; i ; i--);

    // Configure the I2C SCL and SDA pins for I2C operation.
    GPIOPinTypeI2C(GPIO_PORTG_BASE, GPIO_PIN_0 | GPIO_PIN_1);
    
    // Reset the I2C1 peripheral.
    SysCtlPeripheralReset(SYSCTL_PERIPH_I2C1);

    // Initialize the I2C master.
    I2CMasterInitExpClk(I2C1_MASTER_BASE, SysCtlClockGet(), true);
    
    // Compute the inter-byte delay for the display controller.  This delay is
    // dependent upon the I2C bus clock rate; the slower the clock the longer
    // the delay required.
    //
    // The derivation of this formula is based on a measured delay of
    // SysCtlDelay(1700) for a 100 kHz I2C bus with the CPU running at 50 MHz
    // (referred to as C).  To scale this to the delay for a different CPU
    // speed (since this is just a CPU-based delay loop) is:
    //
    //           f(CPU)
    //     C * ----------
    //         50,000,000
    //
    // To then scale this to the actual I2C rate (since it won't always be
    // precisely 100 kHz):
    //
    //           f(CPU)     100,000
    //     C * ---------- * -------
    //         50,000,000    f(I2C)
    //
    // This equation will give the inter-byte delay required for any
    // configuration of the I2C master.  But, as arranged it is impossible to
    // directly compute in 32-bit arithmetic (without loosing a lot of
    // accuracy).  So, the equation is simplified.
    //
    // Since f(I2C) is generated by dividing down from f(CPU), replace it with
    // the equivalent (where TPR is the value programmed into the Master Timer
    // Period Register of the I2C master, with the 1 added back):
    //
    //                        100,000
    //           f(CPU)       -------
    //     C * ---------- *    f(CPU)
    //         50,000,000   ------------
    //                      2 * 10 * TPR
    //
    // Inverting the dividend in the last term:
    //
    //           f(CPU)     100,000 * 2 * 10 * TPR
    //     C * ---------- * ----------------------
    //         50,000,000          f(CPU)
    //
    // The f(CPU) now cancels out.
    //
    //         100,000 * 2 * 10 * TPR
    //     C * ----------------------
    //               50,000,000
    //
    // Since there are no clock frequencies left in the equation, this equation
    // also works for 400 kHz bus operation as well, since the 100,000 in the
    // numerator becomes 400,000 but C is 1/4, which cancel out each other.
    // Reducing the constants gives:
    //
    //         TPR              TPR             TPR
    //     C * ---   =   1700 * ---   =   340 * ---   = 68 * TPR
    //         25               25               5
    //
    // Note that the constant C is actually a bit larger than it needs to be in
    // order to provide some safety margin.
    g_ulDelay = 68 * (HWREG(I2C1_MASTER_BASE + I2C_O_MTPR) + 1);

    // Initialize the display controller.  Loop through the initialization
    // sequence doing a single I2C transfer for each command.
    for(ulIdx = 0; ulIdx < SIZE_INIT_CMDS; ulIdx += g_pucRITInit[ulIdx] + 1)
    {
        // Send this command.
        BSP_DisplayWriteFirst(g_pucRITInit[ulIdx + 1]);
        BSP_DisplayWriteArray(g_pucRITInit + ulIdx + 2,
                               g_pucRITInit[ulIdx] - 2);
        BSP_DisplayWriteFinal(g_pucRITInit[ulIdx + g_pucRITInit[ulIdx]]);
    }

    // Clear the frame buffer.
    BSP_DisplayClear();    
    
    // Turn the display on.    
    BSP_DisplayOn();
}
