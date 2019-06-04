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
 * @brief
 *
 * This function
 *
 * @param pDest
 * @param pSrc
 * @param nSize
 *
 * @return
 */
void *UC_memcpy(void *pDest, const void *pSrc, int nSize);

/**
 * @brief
 *
 * This function
 *
 * @param pCompare1
 * @param pCompare2
 * @param nSize
 *
 * @return
 */
int UC_memcmp(void *pCompare1, void *pCompare2, int nSize);

/**
 * @brief
 *
 * This function
 *
 * @param pDest
 * @param nContents
 * @param nSize
 *
 * @return
 */
void *UC_memset(void *pDest, int nContents, int nSize);

#ifdef __cplusplus
}
#endif

#endif /* SRC_COMMON_INCLUDE_UCBASIC_H_ */
