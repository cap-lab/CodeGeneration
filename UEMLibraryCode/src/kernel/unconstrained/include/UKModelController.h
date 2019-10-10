/*
 * UKModelController.h
 *
 *  Created on: 2019. 9. 30.
 *      Author: jej
 */

#ifndef SRC_KERNEL_INCLUDE_UKMODELCONTROLLER_H_
#define SRC_KERNEL_INCLUDE_UKMODELCONTROLLER_H_

#include <uem_common.h>

#include <uem_data.h>
#include <UCThreadMutex.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef uem_result (*FnTraverseModelControllerFunctions)(STaskGraph *pstCurrentTaskGraph, ETaskControllerType enControllerType,
														SModelControllerFunctionSet *pstFunctionSet, void *pUserData);

uem_result UKModelController_TraverseAndCallFunctions(STaskGraph *pstLeafTaskGraph, HThreadMutex hTaskGraphLock, FnTraverseModelControllerFunctions fnFunction,
													void *pUserData);
uem_result UKModelController_GetTopLevelLockHandle(STaskGraph *pstLeafTaskGraph, OUT HThreadMutex *phMutex);
uem_result UKModelController_CallSubGraphClearFunctions(STaskGraph *pstTaskGraph);

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_INCLUDE_UKMODELCONTROLLER_H_ */
