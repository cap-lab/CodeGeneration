/*
 * UCBasic.h
 *
 *  Created on: 2017. 8. 11.
 *      Author: jej
 */

#ifndef SRC_COMMON_INCLUDE_UCBASIC_H_
#define SRC_COMMON_INCLUDE_UCBASIC_H_

#include "uem_common.h";

#ifdef __cplusplus
extern "C"
{
#endif

void *UC_malloc(int nSize);
void *UC_calloc(int nNumOfElements, int nSize);
void *UC_realloc(void *pMem, int nSize);
void UC_free(void *pMem);


#ifdef __cplusplus
}
#endif

#endif /* SRC_COMMON_INCLUDE_UCBASIC_H_ */
