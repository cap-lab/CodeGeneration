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
void UFSystem_Kill(IN int nCallerTaskId);
void UFSystem_Stop(IN int nCallerTaskId);
#endif

#ifdef __cplusplus
}
#endif

#endif /* SRC_API_INCLUDE_UFSYSTEM_H_ */
