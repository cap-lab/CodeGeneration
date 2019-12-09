/*
 * UCThread.h
 *
 *  Created on: 2017. 8. 10.
 *      Author: jej
 */

#ifndef SRC_COMMON_UNCONSTRAINED_INCLUDE_UCTHREAD_H_
#define SRC_COMMON_UNCONSTRAINED_INCLUDE_UCTHREAD_H_

#include "uem_common.h"

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct _SThread *HThread;



typedef void * (*FnNativeThread)(void *pData);

/**
 * @brief Create a thread.
 *
 * This function creates a thread.
 *
 * @param fnThreadRoutine a routine running on a thread.
 * @param pUserData user data passed to a thread.
 * @param[out] phThread a thread handle to be created.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_OUT_OF_MEMORY, @ref ERR_UEM_INTERNAL_FAIL.
 */
uem_result UCThread_Create(FnNativeThread fnThreadRoutine, void *pUserData, OUT HThread *phThread);

/**
 * @brief Join and destroy a thread.
 *
 * This function joins and destroys a thread. If @a bDetach is TRUE, the thread is not joined but detached, \n
 * so this function does not wail until the thread is terminated. \n
 * Otherwise, the function waits until @a nTimeoutInMS milliseconds for thread termination. \n
 * If the thread is not terminated until the timeout, the function invokes cancel() and kill() functions to terminate a thread forcedly. \n
 * If @a nTimeoutInMS is 0, the function will wait for terminating a thread infinitely.
 *
 * @param[in,out] phThread a thread to be destroyed.
 * @param bDetach detach a thread (TRUE) or wait until the thread is terminated (FALSE).
 * @param nTimeoutInMS timeout for waiting thread termination with milliseconds time unit.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INTERNAL_FAIL. \n
 *         @ref ERR_UEM_INTERNAL_FAIL can be occurred when thread-related operations are failed.
 */
uem_result UCThread_Destroy(IN OUT HThread *phThread, uem_bool bDetach, int nTimeoutInMS);

/**
 * @brief Yield a current thread to other threads
 *
 * This function is a simple wrap-up function for sched_yield().
 */
void UCThread_Yield();

/**
 * @brief Pin a thread to the specific core.
 *
 * This function pins a thread to the core which is used for mapping a thread to the specific core. \n
 * @a nCoreId can be set from 0 to (maximum core number - 1).
 *
 * @param hThread a thread handle.
 * @param nCoreId a core id starting from 0.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INTERNAL_FAIL. \n
 *         @ref ERR_UEM_INTERNAL_FAIL can be occurred when mapping-related operations are failed.
 */
uem_result UCThread_SetMappedCPU(HThread hThread, int nCoreId);


#ifdef __cplusplus
}
#endif

#endif /* SRC_COMMON_UNCONSTRAINED_INCLUDE_UCTHREAD_H_ */
