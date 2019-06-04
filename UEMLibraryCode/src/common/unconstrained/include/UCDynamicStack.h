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
 * @brief
 *
 * This function
 *
 * @param[out] phStack
 *
 * @return
 */
uem_result UCDynamicStack_Create(OUT HStack *phStack);

/**
 * @brief
 *
 * This function
 *
 * @param hStack
 * @param pData
 *
 * @return
 */
uem_result UCDynamicStack_Push(HStack hStack, IN void *pData);

/**
 * @brief
 *
 * This function
 *
 * @param hStack
 * @param[out] ppData
 *
 * @return
 */
uem_result UCDynamicStack_Pop(HStack hStack, OUT void **ppData);

/**
 * @brief
 *
 * This function
 *
 * @param hStack
 * @param[out] pnLength
 *
 * @return
 */
uem_result UCDynamicStack_Length(HStack hStack, OUT int *pnLength);

/**
 * @brief
 *
 * This function
 *
 * @param phStack
 * @param fnDestroyCallback
 * @param pUserData
 *
 * @return
 */
uem_result UCDynamicStack_Destroy(IN OUT HStack *phStack, IN CbFnUCStack fnDestroyCallback, IN void *pUserData);

#ifdef __cplusplus
}
#endif

#endif /* SRC_COMMON_INCLUDE_UCDYNAMICSTACK_H_ */
