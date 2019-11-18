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

/**
 * @brief Traverse task graph model controllers.
 *
 * This function traverses task graph model controllers from @a pstLeafTaskGraph to the top parent task graph. \n
 * @a fnFunction is used as a callback function, and a user can select a function from a function set. \n
 * If @a hTaskGraphLock is not NULL, this function uses a lock handle to serialize the calls from multiple threads.
 *
 * @param pstLeafTaskGraph a task graph structure which calls this traverse functions.
 * @param hTaskGraphLock a highest task graph lock
 * @param fnFunction callback function to handle models
 * @param pUserData user data which is used in callback functions.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - return values from @a fnFunction.
 */
uem_result UKModelController_TraverseAndCallFunctions(STaskGraph *pstLeafTaskGraph, HThreadMutex hTaskGraphLock, FnTraverseModelControllerFunctions fnFunction,
													void *pUserData);
/**
 * @brief Retrieve a top level task graph lock handle.
 *
 * This function retrieves a top-level task graph lock handle.
 *
 * @param pstLeafTaskGraph a task graph structure to get a top-level task graph lock.
 * @param[out] phMutex a retrieved task graph lock handle.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_ILLEGAL_DATA.
 */
uem_result UKModelController_GetTopLevelLockHandle(STaskGraph *pstLeafTaskGraph, OUT HThreadMutex *phMutex);

/**
 * @brief Call clear functions of all child task graphs.
 *
 * This function calls clear functions of all child task graphs.
 *
 * @param pstTaskGraph a task graph structure to call subgraph clear functions.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_ILLEGAL_DATA.
 */
uem_result UKModelController_CallSubGraphClearFunctions(STaskGraph *pstTaskGraph);

/**
 * @brief Retrieve a top-level task graph structure.
 *
 * This function retrieves a top-level task graph structure.
 *
 * @param pstLeafTaskGraph a leaf task graph structure to find a top-level task graph
 * @param[out] ppstGraph a retrieved top-level task graph structure.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_ILLEGAL_DATA.
 */
uem_result UKModelController_GetTopLevelGraph(STaskGraph *pstLeafTaskGraph, OUT STaskGraph **ppstGraph);

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_INCLUDE_UKMODELCONTROLLER_H_ */
