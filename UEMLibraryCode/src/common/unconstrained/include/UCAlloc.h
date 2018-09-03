/*
 * UCAlloc.h
 *
 *  Created on: 2018. 9. 3.
 *      Author: chjej202
 */

#ifndef SRC_COMMON_UNCONSTRAINED_INCLUDE_UCALLOC_H_
#define SRC_COMMON_UNCONSTRAINED_INCLUDE_UCALLOC_H_

#ifdef __cplusplus
extern "C"
{
#endif

void *UCAlloc_malloc(int nSize);
void *UCAlloc_calloc(int nNumOfElements, int nSize);
void *UCAlloc_realloc(void *pMem, int nSize);
void UCAlloc_free(void *pMem);

#define SAFEMEMFREE(mem) if((mem) != NULL){UCAlloc_free((mem));mem=NULL;}

#ifdef __cplusplus
}
#endif

#endif /* SRC_COMMON_UNCONSTRAINED_INCLUDE_UCALLOC_H_ */
