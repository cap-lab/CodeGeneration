/*
 * UFSystem.h
 *
 *  Created on: 2017. 8. 11.
 *      Author: jej
 */

#ifndef SRC_API_INCLUDE_UFSYSTEM_H_
#define SRC_API_INCLUDE_UFSYSTEM_H_

#include <uem_common.h>

#ifdef __cplusplus
extern "C"
{
#endif

#ifndef API_LITE
/**
 * @brief Kill the program.
 * *
 * @param nCallerTaskId id of caller task(currently not used).
 *
 * @return
 */
void UFSystem_Kill(IN int nCallerTaskId);

/**
 * @brief Stop execution of each task, then set whole program to be terminated.
 *
 * @param nCallerTaskId id of caller task(currently not used).
 *
 * @return
 */
void UFSystem_Stop(IN int nCallerTaskId);
#endif

#ifdef __cplusplus
}
#endif

#endif /* SRC_API_INCLUDE_UFSYSTEM_H_ */
