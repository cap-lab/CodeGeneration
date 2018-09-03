/*
 * UCPrint.cpp
 *
 *  Created on: 2018. 9. 3.
 *      Author: chjej202
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <Arduino.h>
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
         Serial.print(pszBuffer);
}

#ifdef __cplusplus
}
#endif
