/*
 * UCBasic.h
 *
 *  Created on: 2017. 8. 11.
 *      Author: jej
 */

#ifndef SRC_COMMON_INCLUDE_UCBASIC_H_
#define SRC_COMMON_INCLUDE_UCBASIC_H_

#include "uem_common.h"

#ifdef __cplusplus
extern "C"
{
#endif

void *UC_malloc(int nSize);
void *UC_calloc(int nNumOfElements, int nSize);
void *UC_realloc(void *pMem, int nSize);
void UC_free(void *pMem);
void *UC_memcpy(void *pDest, const void *pSrc, int nSize);
int UC_memcmp(void *pCompare1, void *pCompare2, int nSize);
void *UC_memset(void *pDest, int nContents, int nSize);

#define SAFEMEMFREE(mem) if((mem) != NULL){UC_free((mem));mem=NULL;}

#ifdef __cplusplus
}
#endif

#endif /* SRC_COMMON_INCLUDE_UCBASIC_H_ */
