/*
 * UCString.h
 *
 *  Created on: 2017. 12. 24.
 *      Author: jej
 */

#ifndef SRC_COMMON_INCLUDE_UCSTRING_H_
#define SRC_COMMON_INCLUDE_UCSTRING_H_

typedef struct _SUCStringStruct {
	char *pszStr;
	int nStringLen;
	int nBufferLen;
} uem_string_struct;

typedef uem_string_struct *uem_string;

uem_result UCString_New(uem_string strToSet, char *pBuffer, int nBufLen);
uem_result UCString_SetLow(uem_string strDst, const char *pszSrc, int nSrcBufLen);
uem_bool UCString_IsEqual(uem_string strCompare1, uem_string strCompare2);
int UCString_ToInteger(uem_string strTarget, int nIndex, OUT int *pnEndIndex);

#define UEMSTRING_MAX (65536 - 1)
//#define UEMSTRING_MAX (2147483647 - 1)

#endif /* SRC_COMMON_INCLUDE_UCSTRING_H_ */
