/*
 * UKLoopTaskController.h
 *
 *  Created on: 2019. 10. 7.
 *      Author: jej
 */

#ifndef SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKLOOPMODELCONTROLLER_H_
#define SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKLOOPMODELCONTROLLER_H_

#include <uem_common.h>

#include <uem_data.h>

#include <UKCPUTaskCommon.h>

#ifdef __cplusplus
extern "C"
{
#endif

uem_result UKLoopModelController_HandleConvergentLoop(STaskGraph *pstGraph, void *pCurrentTaskHandle, void *pCurrentThreadHandle);

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKLOOPMODELCONTROLLER_H_ */
