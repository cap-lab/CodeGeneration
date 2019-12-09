/*
 * uem_data.h
 *
 *  Created on: 2018. 8. 30.
 *      Author: chjej202
 */

#include <uem_common.h>

#include <uem_enum.h>
#include <uem_callbacks.h>

#include <uem_common_struct.h>
#include <uem_channel_data.h>

#ifndef SRC_KERNEL_CONSTRAINED_INCLUDE_UEM_DATA_H_
#define SRC_KERNEL_CONSTRAINED_INCLUDE_UEM_DATA_H_

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct _STaskGraph {
	ETaskGraphType enType;
	STask *astTasks;
	int nNumOfTasks;
	STask *pstParentTask;
} STaskGraph;

typedef struct _STask {
	int nTaskId;
	const char *pszTaskName;
	ETaskType enType;
	STaskFunctions stTaskFunctions;
	ERunCondition enRunCondition;
	int nPeriod;
	ETimeMetric enPeriodMetric;
	STaskGraph *pstSubGraph;
	STaskGraph *pstParentGraph;
	STaskParameter *astTaskParam;
	int nTaskParamNum;
	uem_bool bStaticScheduled;
	STimer *astTimer;
} STask;


typedef struct _SLibrary {
	char *pszLibraryName;
	FnUemLibraryInit fnInit;
	FnUemLibraryWrapup fnWrapup;
} SLibrary;


typedef struct _STaskIdToTaskMap {
	STask *pstTask;
} STaskIdToTaskMap;


typedef struct _SScheduledTasks {
	STaskGraph *pstParentTaskGraph;
	FnUemTaskGo fnCompositeGo;
} SScheduledTasks;


extern STask g_astTasks_top[];
extern int g_nNumOfTasks_top;

extern STaskGraph g_stGraph_top;

extern STaskIdToTaskMap g_astTaskIdToTask[];
extern int g_nTaskIdToTaskNum;

extern SLibrary g_stLibraryInfo[];
extern int g_nLibraryInfoNum;

extern int g_nTimerSlotNum;

extern SScheduledTasks g_astScheduledTaskList[];

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_CONSTRAINED_INCLUDE_UEM_DATA_H_ */
