/*
 * CPUThreadPool.h
 *
 *  Created on: 2017. 9. 20.
 *      Author: jej
 */

#ifndef SRC_MAIN_NATIVE_LINUX_CPUTHREADPOOL_H_
#define SRC_MAIN_NATIVE_LINUX_CPUTHREADPOOL_H_

#include <uem_common.h>

#include <uem_data.h>

#ifdef __cplusplus
extern "C"
{
#endif


typedef enum _ETaskState {
	TASK_STATE_STOP,
	TASK_STATE_RUNNING,
	TASK_STATE_
};

typedef union _UTargetTask {
	STask *pstTask;
	SScheduledTasks *pstScheduledTasks;
} UTargetTask;

typedef struct _STaskThread {
	HThread hThread;
	HThreadEvent hEvent;
	UTargetTask uTargetTask;
	int nNumOfMappedCPUs;
	int nMappedCPUId;
} STaskThread;

typedef struct _SCPUThreadPool {
	HLinkedList hTaskList;
} SCPUThreadPool;

typedef struct _SCPUThreadPool *HCPUThreadPool;

uem_result CPUThreadPool_Create(OUT HCPUThreadPool *phCPUThreadPool);
uem_result CPUThreadPool_RegisterTask(HCPUThreadPool hCPUThreadPool, STask *pstTask, int nCPUId);
uem_result CPUThreadPool_RegisterCompositeTask(HCPUThreadPool hCPUThreadPool, SScheduledTasks *pstScheduledTasks, int nCPUId);
uem_result CPUThreadPool_Destroy(IN OUT HCPUThreadPool *phCPUThreadPool);

#ifdef __cplusplus
}
#endif

#endif /* SRC_MAIN_NATIVE_LINUX_CPUTHREADPOOL_H_ */
