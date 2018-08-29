/*
 * uem_data.h
 *
 *  Created on: 2017. 8. 12.
 *      Author: jej
 */

#ifndef SRC_API_INCLUDE_UEM_DATA_H_
#define SRC_API_INCLUDE_UEM_DATA_H_

#include <uem_common.h>

#include <UCThreadMutex.h>
#include <UCThreadEvent.h>

#include <uem_enum.h>
#include <uem_channel_data.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef void (*FnUemTaskInit)(int nTaskId);
typedef void (*FnUemTaskGo)(int nTaskId);
typedef void (*FnUemTaskWrapup)();

typedef void (*FnUemLibraryInit)();
typedef void (*FnUemLibraryWrapup)();


typedef struct _SModeTransitionMachine SModeTransitionMachine;

typedef uem_bool (*FnTaskModeTranstion)(SModeTransitionMachine *pstModeTransition);

typedef struct _STask STask;

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
	int nCurrentIteration;
} SModeTransitionMachine;

typedef struct _SLoopInfo {
	ELoopType enType;
	int nLoopCount;
	int nDesignatedTaskId;
} SLoopInfo;

typedef struct _STaskGraph {
	ETaskGraphType enType;
	STask *astTasks;
	int nNumOfTasks;
	STask *pstParentTask;
} STaskGraph;


typedef union _UParamValue {
	int nParam;
	double dbParam;
} UParamValue;

typedef struct _STaskParameter {
	int nParamId;
	EParameterType enType;
	const char *pszParamName;
	UParamValue uParamValue;
} STaskParameter;

typedef struct _STaskFunctions {
	FnUemTaskInit fnInit;
	FnUemTaskGo fnGo;
	FnUemTaskWrapup fnWrapup;
} STaskFunctions;

typedef struct _STaskIteration {
	int nModeId;
	int nRunInIteration;
} STaskIteration;


typedef struct _STimer {
	int nSlotId;
	int nTimeInMilliSec;
	uem_bool bAlarmChecked;
	long long llAlarmTime;
} STimer;


typedef struct _STask {
	int nTaskId;
	const char *pszTaskName;
	ETaskType enType;
	STaskFunctions *astTaskFunctions;
	int nTaskFunctionSetNum;
	ERunCondition enRunCondition;
	int nRunRate;
	int nPeriod;
	ETimeMetric enPeriodMetric;
	STaskGraph *pstSubGraph;
	STaskGraph *pstParentGraph;
	SModeTransitionMachine *pstMTMInfo;
	SLoopInfo *pstLoopInfo;
	STaskParameter *astTaskParam;
	int nTaskParamNum;
	uem_bool bStaticScheduled;
	int nThroughputConstraint;
	HThreadMutex hMutex;
	HThreadEvent hEvent;
	STaskIteration *astTaskIteration;
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
	STask *pstParentTask;
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

typedef struct _SMappedGeneralTaskInfo {
	ETaskType enType;
	STask *pstTask;
	int nProcessorId;
	int nLocalId;
} SMappedGeneralTaskInfo;

typedef struct _SMappedCompositeTaskInfo {
	SScheduledTasks *pstScheduledTasks;
	int nProcessorId;
	int nLocalId;
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

extern SChannel g_astChannels[];
extern int g_nChannelNum;

extern SChannelAPI *g_astChannelAPIList[];
extern int g_nChannelAPINum;

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

#ifdef __cplusplus
}
#endif


#endif /* SRC_API_INCLUDE_UEM_DATA_H_ */
