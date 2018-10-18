/*
 * UCFixedSizeQueue.h
 *
 *  Created on: 2018. 10. 11.
 *      Author: chjej202
 */

#ifndef SRC_COMMON_UNCONSTRAINED_INCLUDE_UCFIXEDSIZEQUEUE_H_
#define SRC_COMMON_UNCONSTRAINED_INCLUDE_UCFIXEDSIZEQUEUE_H_

#include <uem_common.h>


#ifdef __cplusplus
extern "C"
{
#endif

typedef struct _SFixedSizeQueue *HFixedSizeQueue;

uem_result UCFixedSizeQueue_Create(int nElementSize, int nElementNum, OUT HFixedSizeQueue *phQueue);
uem_result UCFixedSizeQueue_Destroy(IN OUT HFixedSizeQueue *phQueue);
uem_result UCFixedSizeQueue_PutItem(HFixedSizeQueue hQueue, void *pData, int nElementSize);
uem_result UCFixedSizeQueue_GetItem(HFixedSizeQueue hQueue, void *pData, int *pnElementSize);


#ifdef __cplusplus
}
#endif

#endif /* SRC_COMMON_UNCONSTRAINED_INCLUDE_UCFIXEDSIZEQUEUE_H_ */
