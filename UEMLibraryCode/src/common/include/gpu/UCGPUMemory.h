/*
 * UCCUDABasic.h
 *
 *  Created on: 2018. 4. 2.
 *      Author: DG-SHIN
 */

#ifndef SRC_COMMON_INCLUDE_GPU_UCGPUMEMORY_H_
#define SRC_COMMON_INCLUDE_GPU_UCGPUMEMORY_H_

#include "uem_common.h"

#ifdef __cplusplus
extern "C"
{
#endif

void *UCGPUMemory_Malloc(int nSize);
void *UCGPUMemory_HostAlloc(int nSize, unsigned int flags);
void UCGPUMemory_Free(void *pMem);
void UCGPUMemory_FreeHost(void *pMem);
void *UCGPUMemory_Memcpy(void *pDest, const void *pSrc, int nSize, unsigned int flags);

#define SAFEMEMFREE(mem) if((mem) != NULL){UC_free((mem));mem=NULL;}

#ifdef __cplusplus
}
#endif



#endif /* SRC_COMMON_INCLUDE_GPU_UCGPUMEMORY_H_ */
