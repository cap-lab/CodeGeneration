/*
 * uem_data.h
 *
 *  Created on: 2017. 8. 12.
 *      Author: jej
 *      Changed :
 *  	    1. 2019. 06. 20. wecracy
 */

#ifndef SRC_KERNEL_UNCONSTRAINED_INCLUDE_UEM_DATA_H_
#define SRC_KERNEL_UNCONSTRAINED_INCLUDE_UEM_DATA_H_

#include <uem_common.h>

#include <UCThreadMutex.h>
#include <UCThreadEvent.h>

#include <uem_enum.h>
#include <uem_callbacks.h>
#include <uem_channel_data.h>
#include <uem_multicast_data.h>
#include <uem_common_struct.h>

#include <UKCPUTaskCommon.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct _SModeTransitionMachine SModeTransitionMachine;

typedef uem_bool (*FnTaskModeTranstion)(SModeTransitionMachine *pstModeTransition);

typedef struct _SModeMap {
	int nModeId;
	char *pszModeName;
	STask **pastRelatedChildTasks; // this is used for self-timed or fully static schedule
	int nRelatedChildTaskNum; // this is used for self-timed or fully static schedule
} SModeMap;


typedef struct _SVariableIntMap {
	int nVariableId;
	char *pszVariableName;
	int nValue;
} SVariableIntMap;


#define MODE_TRANSITION_ARRAY_SIZE (1000)

typedef struct _SModeTransitionHistory {
	int nIteration;
	int nModeIndex;
} SModeTransitionHistory;

typedef struct _SModeTransitionMachine {
	int nTaskId;
	SModeMap *astModeMap;
	int nNumOfModes;
	SVariableIntMap *astVarIntMap;
	int nNumOfIntVariables;
	FnTaskModeTranstion fnTransition;
	int nCurModeIndex;
	int nNextModeIndex;
	EModeState enModeState;
	SModeTransitionHistory astModeTransition[MODE_TRANSITION_ARRAY_SIZE];
	int nCurHistoryStartIndex;
	int nCurHistoryLen;
} SModeTransitionMachine;


#define LOOP_HISTORY_ARRAY_SIZE (1000)

typedef struct _SLoopIterationHistory{
	int nPrevIteration;
	int nNextIteration;
} SLoopIterationHistory;

typedef struct _SLoopInfo {
	ELoopType enType;
	int nLoopCount;
	int nDesignatedTaskId;
	uem_bool bDesignatedTaskState; //flag to check whether the task should be terminated.
	SLoopIterationHistory astLoopIteration[LOOP_HISTORY_ARRAY_SIZE];
	int nCurHistoryStartIndex;
	int nCurHistoryLen;
} SLoopInfo;

typedef struct _STaskThreadContext {
	int nCurRunIndex; // the run count of the task which is responsible by the current thread
	int nCurThreadIteration;
	int nTargetThreadIteration;
} STaskThreadContext;

typedef struct _STaskIteration {
	int nModeId;
	int nRunInIteration;
} STaskIteration;

typedef enum _ETaskControllerType {
	CONTROLLER_TYPE_VOID,
	CONTROLLER_TYPE_CONTROL_TASK_INCLUDED,
	CONTROLLER_TYPE_DYNAMIC_MODE_TRANSITION,
	CONTROLLER_TYPE_STATIC_MODE_TRANSITION,
	CONTROLLER_TYPE_DYNAMIC_CONVERGENT_LOOP,
	CONTROLLER_TYPE_DYNAMIC_DATA_LOOP,
	CONTROLLER_TYPE_STATIC_CONVERGENT_LOOP,
	CONTROLLER_TYPE_STATIC_DATA_LOOP,
} ETaskControllerType;

typedef enum _EModelControllerFunction {
	FUNC_HANDLE_MODEL,
	FUNC_GET_ITERATION_INDEX,
	FUNC_CLEAR,
	FUNC_CHANGE_THREAD_STATE,
} EModelControllerFunction;

typedef enum _EScheduler {
	SCHEDULER_OTHER = 0, 
	SCHEDULER_FIFO = 1, // for Linux
	SCHEDULER_RR = 2,
	SCHEDULER_HIGH = 0x80, // for Windows
	SCHEDULER_REALTIME = 0x100, 
} EScheduler;


typedef uem_result (*FnHandleModel)(STaskGraph *pstGraph, void *pCurrentTaskHandle, void *pCurrentThreadHandle);
typedef uem_result (*FnControllerClear)(STaskGraph *pstTaskGraph);
typedef uem_result (*FnChangeTaskThreadState)(STaskGraph *pstGraph, void *pCurrentTaskHandle, void *pCurrentThreadHandle, ECPUTaskState enTargetState, OUT ECPUTaskState *penState);


typedef struct _SModelControllerFunctionSet {
	FnHandleModel fnHandleModel;
	FnControllerClear fnClear;
	FnChangeTaskThreadState fnChangeThreadState;
	FnHandleModel fnHandleStopping;
} SModelControllerFunctionSet;

typedef struct _SModelControllerCommon {
	HThreadMutex hMutex;
	int nThroughputConstraint; // Only used for composite schedule
	int nCurrentIteration;
	SModelControllerFunctionSet *pstFunctionSet;
} SModelControllerCommon;

typedef struct _SModeTransitionController {
	SModelControllerCommon stCommon;
	SModeTransitionMachine *pstMTMInfo;
} SModeTransitionController;

typedef struct _SLoopController {
	SModelControllerCommon stCommon;
	SLoopInfo *pstLoopInfo;
} SLoopController;

typedef struct _STaskGraph {
	ETaskGraphType enType;
	ETaskControllerType enControllerType;
	void *pController; // SLoopController (SDF/L) or SModeTransitionController (MTM) or STaskControllerCommon (Control-task-included)
	STask *astTasks;
	int nNumOfTasks;
	STask *pstParentTask;
} STaskGraph;


typedef struct _STask {
	int nTaskId;
	const char *pszTaskName;
	ETaskType enType;
	STaskFunctions *astTaskThreadFunctions;
	STaskThreadContext *astThreadContext;
	int nTaskThreadSetNum;
	ERunCondition enRunCondition;
	int nPeriod;
	ETimeMetric enPeriodMetric;
	STaskGraph *pstSubGraph;
	STaskGraph *pstParentGraph;
	STaskParameter *astTaskParam;
	int nTaskParamNum;
	uem_bool bStaticScheduled; // WILL BE REMOVED
	int nThroughputConstraint; // WILL BE REMOVED
	HThreadMutex hMutex;
	HThreadEvent hEvent;
	STaskIteration *astTaskIteration;
	int nTaskIterationArrayNum;
	int nCurRunInIteration;
	int nCurIteration;
	int nTargetIteration;
	STimer *astTimer;
} STask;


typedef struct _SLibrary {
	char *pszLibraryName;
	FnUemLibraryInit fnInit;
	FnUemLibraryWrapup fnWrapup;
} SLibrary;


typedef struct _STaskIdToTaskMap {
	int nTaskId;
	const char *pszTaskName;
	STask *pstTask;
} STaskIdToTaskMap;


typedef struct _SScheduleList {
	FnUemTaskGo fnCompositeGo;
	int nThroughputConstraint;
	uem_bool bHasSourceTask; // This variable is used for finding which composite task thread contains source task
} SScheduleList;

// SScheduledTasks can be existed per each task mode
typedef struct _SScheduledTasks {
	STaskGraph *pstParentTaskGraph;
	int nModeId; // mode ID
	SScheduleList *astScheduleList;
	int nScheduleNum; // number of schedules which satisfies throughput constraint
	int nScheduledIndex; // target schedule to be scheduled
	int nSeqInMode;
} SScheduledTasks;

typedef struct _SProcessor {
	int nProcessorId;
	uem_bool bIsCPU;
	const char *pszProcessorName;
	int nPoolSize;
} SProcessor;

typedef uem_result (*FnAddOnModuleInitialize)();
typedef uem_result (*FnAddOnModuleFinalize)();

typedef struct _SAddOnModule {
	FnAddOnModuleInitialize fnInitialize;
	FnAddOnModuleFinalize fnFinalize;
} SAddOnModule;

typedef uem_result (*FnMapProcessor)(HThread hThread, int nProcessorId, int nLocalId);
typedef uem_result (*FnMapPriority)(HThread hThread, int nScheduler, int nPriority);

typedef struct _SGenericMapProcessor {
	FnMapProcessor fnMapProcessor;
	FnMapPriority fnMapPriority;
} SGenericMapProcessor;

typedef struct _SMappedGeneralTaskInfo {
	ETaskType enType;
	STask *pstTask;
	int nProcessorId;
	int nLocalId;
	int nPriority;
	SGenericMapProcessor *pstMapProcessorAPI;
	const char *pszMappingSet;
} SMappedGeneralTaskInfo;

typedef struct _SMappedCompositeTaskInfo {
	SScheduledTasks *pstScheduledTasks;
	int nProcessorId;
	int nLocalId;
	int nPriority;
	SGenericMapProcessor *pstMapProcessorAPI;
} SMappedCompositeTaskInfo;

typedef struct _SMappedTaskInfo {
	SMappedGeneralTaskInfo *pstGeneralTaskMappingInfo;
	int nMappedGeneralTaskNum;
	SMappedCompositeTaskInfo *pstCompositeTaskMappingInfo;
	int nMappedCompositeTaskNum;
} SMappedTaskInfo;

typedef struct _SExecutionTime {
	int nValue;
	ETimeMetric enTimeMetric;
} SExecutionTime;

extern SExecutionTime g_stExecutionTime;

extern STask g_astTasks_top[];
extern int g_nNumOfTasks_top;

extern STaskGraph g_stGraph_top;

extern STaskIdToTaskMap g_astTaskIdToTask[];
extern int g_nTaskIdToTaskNum;

extern SProcessor g_astProcessorInfo[];
extern int g_nProcessorInfoNum;

extern SMappedTaskInfo g_stMappingInfo;

extern SLibrary g_stLibraryInfo[];
extern int g_nLibraryInfoNum;

extern int g_nTimerSlotNum;

extern uem_bool g_bSystemExit;

extern int g_nDeviceId;

extern int g_nScheduler;

#ifdef __cplusplus
}
#endif


#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_UEM_DATA_H_ */
