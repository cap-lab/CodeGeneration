/*
 * UCString.h
 *
 *  Created on: 2017. 12. 24.
 *      Author: jej
 */

#ifndef SRC_COMMON_INCLUDE_UCSTRING_H_
#define SRC_COMMON_INCLUDE_UCSTRING_H_

#include <uem_common.h>

#ifdef __cplusplus
extern "C"
{
#endif

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
uem_result UCString_AppendLow(uem_string strDst, char *pszSrc, int nSrcBufLen);

#define UEMSTRING_MAX (65536 - 1)
//#define UEMSTRING_MAX (2147483647 - 1)

#ifdef __cplusplus
}
#endif

#endif /* SRC_COMMON_INCLUDE_UCSTRING_H_ */
