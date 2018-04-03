/*
 * UCCUDABasic.h
 *
 *  Created on: 2018. 4. 2.
 *      Author: DG-SHIN
 */

#ifndef SRC_COMMON_INCLUDE_UCCUDABASIC_H_
#define SRC_COMMON_INCLUDE_UCCUDABASIC_H_

#include "uem_common.h"

#ifdef __cplusplus
extern "C"
{
#endif

void *UC_cudaMalloc(int nSize);
void *UC_cudaHostAlloc(int nSize, unsigned int flags);
void UC_cudaFree(void *pMem);
void UC_cudaFreeHost(void *pMem);
void *UC_cudaMemcpy(void *pDest, const void *pSrc, int nSize, unsigned int flags);

#define SAFEMEMFREE(mem) if((mem) != NULL){UC_free((mem));mem=NULL;}

#ifdef __cplusplus
}
#endif



#endif /* SRC_COMMON_INCLUDE_UCCUDABASIC_H_ */
