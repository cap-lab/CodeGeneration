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

/**
 * @brief
 *
 * This function
 *
 * @param nElementSize
 * @param nElementNum
 * @param[out] phQueue
 *
 * @return
 */
uem_result UCFixedSizeQueue_Create(int nElementSize, int nElementNum, OUT HFixedSizeQueue *phQueue);

/**
 * @brief
 *
 * This function
 *
 * @param phQueue
 *
 * @return
 */
uem_result UCFixedSizeQueue_Destroy(IN OUT HFixedSizeQueue *phQueue);

/**
 * @brief
 *
 * This function
 *
 * @param hQueue
 * @param pData
 * @param nElementSize
 *
 * @return
 */
uem_result UCFixedSizeQueue_PutItem(HFixedSizeQueue hQueue, void *pData, int nElementSize);

/**
 * @brief
 *
 * This function
 *
 * @param hQueue
 * @param pData
 * @param[out] pnElementSize
 *
 * @return
 */
uem_result UCFixedSizeQueue_GetItem(HFixedSizeQueue hQueue, void *pData, OUT int *pnElementSize);


#ifdef __cplusplus
}
#endif

#endif /* SRC_COMMON_UNCONSTRAINED_INCLUDE_UCFIXEDSIZEQUEUE_H_ */
