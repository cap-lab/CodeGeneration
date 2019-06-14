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
 * @brief Allocate host memory.
 *
 * This function allocates host memory. This is a wrapper function of malloc().
 *
 * @param nSize the size of memory to be allocated.
 *
 * @return the pointer to allocated memory. If the malloc() fails, it will return NULL.
 */
void *UCAlloc_malloc(int nSize);

/**
 * @brief Allocate host memory and set bits to 0.
 *
 * This function allocates host memory and clear memory to 0. \n
 * This is a wrapper function of calloc(). Total allocated size will be @a nNumOfElements * @a nSize.
 *
 * @param nNumOfElements the number of elements to allocate.
 * @param nSize the size of each element.
 *
 * @return the pointer to allocated memory. If the calloc() fails, it will return NULL.
 */
void *UCAlloc_calloc(int nNumOfElements, int nSize);

/**
 * @brief Reallocate host memory.
 *
 * This function changes the size of the memory. This function may move the memory location. \n
 * This is a wrapper function of realloc(). If the retrieved pointer indicates the new location, \n
 * @a pMem may be deallocated.
 *
 * @param pMem pointer to allocated memory.
 * @param nSize new memory size.
 *
 * @return the pointer to reallocated memory. If the realloc() fails, it will return NULL.
 */
void *UCAlloc_realloc(void *pMem, int nSize);

/**
 * @brief Deallocate host memory.
 *
 * This function deallocates the allocated memory. This is a wrapper function of free().
 *
 * @param pMem pointer to allocated memory.
 */
void UCAlloc_free(void *pMem);

#define SAFEMEMFREE(mem) if((mem) != NULL){UCAlloc_free((mem));mem=NULL;}

#ifdef __cplusplus
}
#endif

#endif /* SRC_COMMON_UNCONSTRAINED_INCLUDE_UCALLOC_H_ */
