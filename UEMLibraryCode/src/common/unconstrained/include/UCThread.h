/*
 * UCThread.h
 *
 *  Created on: 2017. 8. 10.
 *      Author: jej
 */

#ifndef SRC_COMMON_INCLUDE_UCTHREAD_H_
#define SRC_COMMON_INCLUDE_UCTHREAD_H_

#include "uem_common.h"

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct _SThread *HThread;



typedef void * (*FnNativeThread)(void *pData);

/**
 * @brief
 *
 * This function
 *
 * @param fnThreadRoutine
 * @param pUserData
 * @param[out] phThread
 *
 * @return
 */
uem_result UCThread_Create(FnNativeThread fnThreadRoutine, void *pUserData, OUT HThread *phThread);

/**
 * @brief
 *
 * This function
 *
 * @param[in,out] phThread
 * @param bDetach
 * @param nTimeoutInMS
 *
 * @return
 */
uem_result UCThread_Destroy(IN OUT HThread *phThread, uem_bool bDetach, int nTimeoutInMS);

/**
 * @brief
 *
 * This function
 *
 * @return
 */
void UCThread_Yield();

/**
 * @brief
 *
 * This function
 *
 * @param hThread
 * @param nCoreId
 *
 * @return
 */
uem_result UCThread_SetMappedCPU(HThread hThread, int nCoreId);


#ifdef __cplusplus
}
#endif

#endif /* SRC_COMMON_INCLUDE_UCTHREAD_H_ */
