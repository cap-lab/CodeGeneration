/*
 * UCThreadMutex.h
 *
 *  Created on: 2017. 8. 5.
 *      Author: jej
 */

#ifndef SRC_COMMON_INCLUDE_UCTHREADMUTEX_H_
#define SRC_COMMON_INCLUDE_UCTHREADMUTEX_H_

#include "uem_common.h"

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct _SThreadMutex *HThreadMutex;

extern HThreadMutex *g_ahStaticThreadMutexes;


/**
 * @brief
 *
 * This function
 *
 * @param[out] phMutex
 *
 * @return
 */
uem_result UCThreadMutex_Create(HThreadMutex *phMutex);

/**
 * @brief
 *
 * This function
 *
 * @param hMutex
 *
 * @return
 */
uem_result UCThreadMutex_Lock(HThreadMutex hMutex);

/**
 * @brief
 *
 * This function
 *
 * @param hMutex
 *
 * @return
 */
uem_result UCThreadMutex_Unlock(HThreadMutex hMutex);

/**
 * @brief
 *
 * This function
 *
 * @param phMutex
 *
 * @return
 */
uem_result UCThreadMutex_Destroy(HThreadMutex *phMutex);

#ifdef __cplusplus
}
#endif

#endif /* SRC_COMMON_INCLUDE_UCTHREADMUTEX_H_ */
