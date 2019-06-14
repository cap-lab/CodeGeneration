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
 * @brief Create a fixed-size queue.
 *
 * This function creates a fixed-size FIFO queue. All elements have same size. \n
 * Maximum element number and each element size are used as an input to create a new queue.
 *
 * @param nElementSize the item size of the queue.
 * @param nElementNum the maximum number of items in the queue.
 * @param[out] phQueue a queue handle to be created.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_OUT_OF_MEMORY, @ref ERR_UEM_MUTEX_ERROR, \n
 *         @ref ERR_UEM_INTERNAL_FAIL.
 */
uem_result UCFixedSizeQueue_Create(int nElementSize, int nElementNum, OUT HFixedSizeQueue *phQueue);

/**
 * @brief Destroy a queue.
 *
 * This function destroys a queue.
 *
 * @param[in,out] phQueue a queue handle to be destroyed.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_INVALID_HANDLE. \n
 */
uem_result UCFixedSizeQueue_Destroy(IN OUT HFixedSizeQueue *phQueue);

/**
 * @brief Put an item into the queue.
 *
 * This function puts an item into the queue.
 *
 * @param hQueue a queue handle.
 * @param pData an item to be put.
 * @param nElementSize the size of the item.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_TIME_EXPIRED, \n
 *         @ref ERR_UEM_SUSPEND. \n
 *         @ref ERR_UEM_TIME_EXPIRED can be returned when the queue is full and the item cannot be put to the queue during specific amount of time.
 *         @ref ERR_UEM_SUSPEND can be occurred when the queue is going to be destroyed by calling @ref UCFixedSizeQueue_Destroy.
 */
uem_result UCFixedSizeQueue_PutItem(HFixedSizeQueue hQueue, void *pData, int nElementSize);

/**
 * @brief Get an item from the queue.
 *
 * This function copies an item from the queue to @a pData. Because this is a FIFO-queue, the first-come item is retrieved. \n
 * The retrieved item is removed from the queue.
 *
 * @param hQueue a queue handle.
 * @param pData a buffer to retrieve an item.
 * @param[out] pnElementSize an item size.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_TIME_EXPIRED, \n
 *         @ref ERR_UEM_SUSPEND. \n
 *         @ref ERR_UEM_TIME_EXPIRED can be returned when the queue is full and the item cannot be put to the queue during specific amount of time.
 *         @ref ERR_UEM_SUSPEND can be occurred when the queue is going to be destroyed by calling @ref UCFixedSizeQueue_Destroy.
 */
uem_result UCFixedSizeQueue_GetItem(HFixedSizeQueue hQueue, void *pData, OUT int *pnElementSize);


#ifdef __cplusplus
}
#endif

#endif /* SRC_COMMON_UNCONSTRAINED_INCLUDE_UCFIXEDSIZEQUEUE_H_ */
