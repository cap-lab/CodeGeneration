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
 * @brief Assign a buffer to uem_string.
 *
 * This function assigns a buffer and buffer length to use bytes of array as a string. \n
 * By using this uem_string, it manages string length information to prevent string buffer overflow.
 * Because UCString does not allocate any extra memory,
 *
 * @param strToSet uem_string structure to set buffer and buffer length.
 * @param pBuffer buffer with a string.
 * @param nBufLen size of the buffer.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UCString_New(uem_string strToSet, char *pBuffer, int nBufLen);

/**
 * @brief Set uem_string.
 *
 * This function sets uem_string @a strSrc to @a strDst.
 *
 * @param strDst a destination string to be set.
 * @param strSrc a source string to be copied.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         @ref ERR_UEM_TRUNCATED is returned if the destination buffer is smaller than source buffer.
 *         Errors to be returned - @ref ERR_UEM_INVALID_HANDLE.
 */
uem_result UCString_Set(uem_string strDst, uem_string strSrc);

/**
 * @brief Set uem_string from byte array.
 *
 * This function sets the string from byte array to @a strDst. \n
 * The function can figure out NULL terminator in the buffer, and only copies the characters before the NULL terminator. \n
 * If there is no NULL terminator in the string, this function will copy the whole buffer.
 *
 * @param strDst a destination string to be set.
 * @param pszSrc a buffer which stores source string.
 * @param nSrcBufLen the size of source buffer.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         @ref ERR_UEM_TRUNCATED is returned if the destination buffer is smaller than source buffer.
 *         Errors to be returned - @ref ERR_UEM_INVALID_HANDLE.
 */
uem_result UCString_SetLow(uem_string strDst, const char *pszSrc, int nSrcBufLen);

/**
 * @brief Compare two strings.
 *
 * This function compares two strings which are equal.
 *
 * @param strCompare1 a string to compare.
 * @param strCompare2 a string to compare.
 *
 * @return If two strings are same, it returns TRUE. Otherwise it returns FALSE.
 */
uem_bool UCString_IsEqual(uem_string strCompare1, uem_string strCompare2);

/**
 * @brief Convert uem_string to an integer value.
 *
 * This function converts string to integer value. Corresponding libc function is strtol(). \n
 * It converts the string started from @a nIndex to an integer value.
 *
 * @param strTarget a target string to perform integer-value conversion.
 * @param nIndex a start index to convert an integer value.
 * @param[out] pnEndIndex an end index which points to the last converted character.
 *
 * @return This function returns an integer value. \n
 *         If the value is not converted, return value and retrieved value of @a pnEndIndex both become 0.
 *
 */
int UCString_ToInteger(uem_string strTarget, int nIndex, OUT int *pnEndIndex);

/**
 * @brief Append a byte array to the uem_string.
 *
 * This function appends a byte array @a pszSrc to uem_string @a strDst. \n
 * The function can figure out NULL terminator in the buffer, and only appends the characters before the NULL terminator. \n
 * If there is no NULL terminator in the string, this function will append the whole buffer.
 *
 * @param strDst a destination string to be appended.
 * @param pszSrc a buffer which stores source string.
 * @param nSrcBufLen the size of source buffer.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         @ref ERR_UEM_TRUNCATED is returned if the destination buffer is smaller than source buffer.
 *         Errors to be returned - @ref ERR_UEM_INVALID_HANDLE.
 */
uem_result UCString_AppendLow(uem_string strDst, char *pszSrc, int nSrcBufLen);

/**
 * @brief  Retrieve the length of the string.
 *
 * This function retrieves the length of the string.
 *
 * @param strTarget a target string to retrieve the string length.
 *
 * @return This function returns the string length. \n
 *         If an error occurs or string is empty, it will return 0.
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
