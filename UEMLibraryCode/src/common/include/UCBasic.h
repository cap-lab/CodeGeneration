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


void *UC_memcpy(void *pDest, const void *pSrc, int nSize);
int UC_memcmp(void *pCompare1, void *pCompare2, int nSize);
void *UC_memset(void *pDest, int nContents, int nSize);

#ifdef __cplusplus
}
#endif

#endif /* SRC_COMMON_INCLUDE_UCBASIC_H_ */
