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

void *UC_malloc(int nSize);
void *UC_calloc(int nNumOfElements, int nSize);
void *UC_realloc(void *pMem, int nSize);
void UC_free(void *pMem);

#define SAFEMEMFREE(mem) if((mem) != NULL){UC_free((mem));mem=NULL;}

#ifdef __cplusplus
}
#endif

#endif /* SRC_COMMON_UNCONSTRAINED_INCLUDE_UCALLOC_H_ */
