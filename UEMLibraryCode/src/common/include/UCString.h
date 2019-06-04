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

/**
 * @brief
 *
 * This function
 *
 * @param strToSet
 * @param pBuffer
 * @param nBufLen
 *
 * @return
 */
uem_result UCString_New(uem_string strToSet, char *pBuffer, int nBufLen);

/**
 * @brief
 *
 * This function
 *
 * @param strDst
 * @param strSrc
 *
 * @return
 */
uem_result UCString_Set(uem_string strDst, uem_string strSrc);

/**
 * @brief
 *
 * This function
 *
 * @param strDst
 * @param pszSrc
 * @param nSrcBufLen
 *
 * @return
 */
uem_result UCString_SetLow(uem_string strDst, const char *pszSrc, int nSrcBufLen);

/**
 * @brief
 *
 * This function
 *
 * @param strCompare1
 * @param strCompare2
 *
 * @return
 */
uem_bool UCString_IsEqual(uem_string strCompare1, uem_string strCompare2);

/**
 * @brief
 *
 * This function
 *
 * @param strTarget
 * @param nIndex
 * @param[out] pnEndIndex
 *
 * @return
 */
int UCString_ToInteger(uem_string strTarget, int nIndex, OUT int *pnEndIndex);

/**
 * @brief
 *
 * This function
 *
 * @param strDst
 * @param pszSrc
 * @param nSrcBufLen
 *
 * @return
 */
uem_result UCString_AppendLow(uem_string strDst, char *pszSrc, int nSrcBufLen);

/**
 * @brief
 *
 * This function
 *
 * @param strTarget
 *
 * @return
 */
int UCString_Length(uem_string strTarget);

#ifdef ARDUINO
	#define UEMSTRING_MAX (256 - 2)
	#define UEMSTRING_CONST (256 - 1) // if this value is set at UCString_New's nBufLen, it will consider the buffer size which is same to string length + 1
#else
	#define UEMSTRING_MAX (65536 - 2)
	#define UEMSTRING_CONST (65536 - 1) // if this value is set at UCString_New's nBufLen, it will consider the buffer size which is same to string length + 1
#endif
//#define UEMSTRING_MAX (2147483647 - 2)
//#define UEMSTRING_MAX (2147483647 - 1)

#ifdef __cplusplus
}
#endif

#endif /* SRC_COMMON_INCLUDE_UCSTRING_H_ */
