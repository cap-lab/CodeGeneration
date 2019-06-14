/*
 * UCBasic.h
 *
 *  Created on: 2017. 8. 11.
 *      Author: jej
 */

#ifndef SRC_COMMON_INCLUDE_UCBASIC_H_
#define SRC_COMMON_INCLUDE_UCBASIC_H_

#include <uem_common.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief Copy memory
 *
 * This function copies memory from @a pSrc to @a pDest. \n
 * This is a wrapper function of memcpy().
 *
 * @param pDest pointer of destination buffer where the content is to be copied.
 * @param pSrc pointer of source of data to be copied.
 * @param nSize number of bytes to copy.
 *
 * @return @a pDest is returned.
 */
void *UC_memcpy(void *pDest, const void *pSrc, int nSize);

/**
 * @brief Compare memory.
 *
 * This function compares memory @a pCompare1 and @a pCompare2.
 *
 * @param pCompare1 pointer of memory to compare.
 * @param pCompare2 pointer of memory to compare.
 * @param nSize number of bytes to compare.
 *
 * @return It returns zero, if two memory blocks are same. Otherwise, it will return the difference of first unmatched byte.
 */
int UC_memcmp(void *pCompare1, void *pCompare2, int nSize);

/**
 * @brief Set memory with specific value.
 *
 * This function sets all the bytes to @a nContents.
 *
 * @param pDest pointer to destination buffer.
 * @param nContents value to be set.
 * @param nSize number of bytes to set.
 *
 * @return @a pDest is returned.
 */
void *UC_memset(void *pDest, int nContents, int nSize);

#ifdef __cplusplus
}
#endif

#endif /* SRC_COMMON_INCLUDE_UCBASIC_H_ */
