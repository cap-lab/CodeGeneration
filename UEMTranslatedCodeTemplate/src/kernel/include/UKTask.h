/*
 * UKTask.h
 *
 *  Created on: 2017. 11. 11.
 *      Author: jej
 */

#ifndef SRC_KERNEL_INCLUDE_UKTASK_H_
#define SRC_KERNEL_INCLUDE_UKTASK_H_

#include <uem_common.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef uem_result (*FnTaskTraverse)(STask *pstTask, void *pUserData);

uem_result UKTask_GetTaskFromTaskId(int nTaskId, STask **ppstTask);
uem_result UKTask_TraverseAllTasks(FnTaskTraverse fnCallback, void *pUserData);

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_INCLUDE_UKTASK_H_ */
