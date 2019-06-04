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

/**
 * @brief
 *
 * This function
 *
 * @param nSize
 *
 * @return
 */
void *UCAlloc_malloc(int nSize);

/**
 * @brief
 *
 * This function
 *
 * @param nNumOfElements
 * @param nSize
 *
 * @return
 */
void *UCAlloc_calloc(int nNumOfElements, int nSize);

/**
 * @brief
 *
 * This function
 *
 * @param pMem
 * @param nSize
 *
 * @return
 */
void *UCAlloc_realloc(void *pMem, int nSize);

/**
 * @brief
 *
 * This function
 *
 * @param pMem
 *
 * @return
 */
void UCAlloc_free(void *pMem);

#define SAFEMEMFREE(mem) if((mem) != NULL){UCAlloc_free((mem));mem=NULL;}

#ifdef __cplusplus
}
#endif

#endif /* SRC_COMMON_UNCONSTRAINED_INCLUDE_UCALLOC_H_ */
