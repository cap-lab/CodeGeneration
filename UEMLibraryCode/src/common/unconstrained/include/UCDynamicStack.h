/*
 * UCDynamicStack.h
 *
 *  Created on: 2017. 10. 28.
 *      Author: jej
 */

#ifndef SRC_COMMON_INCLUDE_UCDYNAMICSTACK_H_
#define SRC_COMMON_INCLUDE_UCDYNAMICSTACK_H_

#include "uem_common.h"

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct _SUCDynamicStack *HStack;

typedef uem_result (*CbFnUCStack)(IN void *pData, IN void *pUserData);

/**
 * @brief Create a stack handle.
 *
 * This function creates an unbounded stack.
 *
 * @param[out] phStack a stack handle to be created.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_OUT_OF_MEMORY.
 */
uem_result UCDynamicStack_Create(OUT HStack *phStack);

/**
 * @brief Push an item into the stack.
 *
 * This function puts an item into the stack.
 *
 * @param hStack a stack handle.
 * @param pData an item to be pushed.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_OUT_OF_MEMORY.
 */
uem_result UCDynamicStack_Push(HStack hStack, IN void *pData);

/**
 * @brief Pop an item from the stack.
 *
 * This function retrieves a recently-pushed stack item and removes the item from the stack. \n
 * The length of the stack is decreased after calling this function because an item is popped out from the stack.\n
 *
 * @warning Popped data is no longer managed by the stack, so memory free is needed on that item if the popped item is memory-allocated. \n
 *
 * @param hStack a stack handle.
 * @param[out] ppData an item which is popped out.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_NO_DATA..
 *         @ref ERR_UEM_NO_DATA is occurred if the stack is empty.
 *
 */
uem_result UCDynamicStack_Pop(HStack hStack, OUT void **ppData);

/**
 * @brief Get the number of items in the stack.
 *
 * This function retrieves the number of items in the stack.
 *
 * @param hStack a stack handle.
 * @param[out] pnLength the number of items in the stack.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UCDynamicStack_Length(HStack hStack, OUT int *pnLength);

/**
 * @brief Destroy a stack.
 *
 * This function destroys a stack.
 * Because stack values can be memory-allocated items, callback function for each item destruction is also provided. \n
 *
 * @param[in,out] phStack a stack handle to be destroyed.
 * @param fnDestroyCallback  callback function for destroying internal stack values.
 * @param pUserData pointer to user data passing to the callback function.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 */
uem_result UCDynamicStack_Destroy(IN OUT HStack *phStack, IN CbFnUCStack fnDestroyCallback, IN void *pUserData);

#ifdef __cplusplus
}
#endif

#endif /* SRC_COMMON_INCLUDE_UCDYNAMICSTACK_H_ */
