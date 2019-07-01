/*
 * UCThreadMutex.h
 *
 *  Created on: 2017. 8. 5.
 *      Author: jej
 */

#ifndef SRC_COMMON_UNCONSTRAINED_INCLUDE_UCTHREADMUTEX_H_
#define SRC_COMMON_UNCONSTRAINED_INCLUDE_UCTHREADMUTEX_H_

#include "uem_common.h"

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct _SThreadMutex *HThreadMutex;

extern HThreadMutex *g_ahStaticThreadMutexes;


/**
 * @brief Create a mutex.
 *
 * This function creates mutual exclusion area (or critical section).
 *
 * @param[out] phMutex a mutex handle to be created.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_OUT_OF_MEMORY, @ref ERR_UEM_MUTEX_ERROR.
 *         @ref ERR_UEM_MUTEX_ERROR can be occurred when the internal OS-dependent mutex operation is failed.
 */
uem_result UCThreadMutex_Create(HThreadMutex *phMutex);

/**
 * @brief Enter mutual exclusion area.
 *
 * This function enters mutual exclusion area. To leave mutual exclusion area, @ref UCThreadMutex_Unlock is used.
 *
 * @param hMutex a mutex handle.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_MUTEX_ERROR. \n
 *         @ref ERR_UEM_MUTEX_ERROR can be occurred when the internal OS-dependent mutex operation is failed.
 */
uem_result UCThreadMutex_Lock(HThreadMutex hMutex);

/**
 * @brief Leave mutual exclusion area.
 *
 * This function leaves mutual exclusion area.
 *
 * @param hMutex a mutex handle.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_MUTEX_ERROR. \n
 *         @ref ERR_UEM_MUTEX_ERROR can be occurred when the internal OS-dependent mutex operation is failed.
 */
uem_result UCThreadMutex_Unlock(HThreadMutex hMutex);

/**
 * @brief Destroy mutual exclusion area.
 *
 * This function destroys mutual exclusion area.
 *
 * @param[in,out] phMutex a mutex handle to be destroyed.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_MUTEX_ERROR. \n
 *         @ref ERR_UEM_MUTEX_ERROR can be occurred when the internal OS-dependent mutex operation is failed.
 */
uem_result UCThreadMutex_Destroy(IN OUT HThreadMutex *phMutex);

#ifdef __cplusplus
}
#endif

#endif /* SRC_COMMON_UNCONSTRAINED_INCLUDE_UCTHREADMUTEX_H_ */
