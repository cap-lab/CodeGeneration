/*
 * UCPrint.cpp
 *
 *  Created on: 2018. 9. 3.
 *      Author: chjej202
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#ifndef ARDUINO_OpenCR
	#include <Arduino.h>
	#define DEBUG_SERIAL Serial
#else
	#include <Arduino.h>
	#define DEBUG_SERIAL Serial2
#endif

#include <stdarg.h>

#define MAXSTRING_LEN (128)

#ifdef __cplusplus
extern "C" {
#endif

void UCPrint_format(const char *pszFormat, ... )
{
         char pszBuffer[MAXSTRING_LEN];
         va_list stArgs;
         va_start (stArgs, pszFormat );
         vsnprintf(pszBuffer, MAXSTRING_LEN, pszFormat, stArgs);
         va_end (stArgs);
         DEBUG_SERIAL.print(pszBuffer);
}

#ifdef __cplusplus
}
#endif
