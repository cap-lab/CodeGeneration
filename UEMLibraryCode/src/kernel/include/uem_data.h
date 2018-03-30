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

#ifdef __cplusplus
extern "C"
{
#endif

#define INVALID_TASK_ID (-1)
#define INVALID_SCHEDULE_ID (-1)
#define INVALID_MODE_ID (-1)
#define INVALID_CHANNEL_ID (-1)
#define INVALID_CHUNK_INDEX (-1)
#define INVALID_TIMER_SLOT_ID (-1)
#define VARIABLE_SAMPLE_RATE (-1)
#define MAPPING_NOT_SPECIFIED (-1)
#define CHUNK_NUM_NOT_INITIALIZED (-1)
#define INVALID_ARRAY_INDEX (-1)

typedef enum _EParameterType {
	PARAMETER_TYPE_DOUBLE,
	PARAMETER_TYPE_INT,
} EParameterType;

typedef enum _ETaskType {
	TASK_TYPE_COMPUTATIONAL,
	TASK_TYPE_CONTROL,
	TASK_TYPE_LOOP,
	TASK_TYPE_COMPOSITE,
} ETaskType;

typedef enum _ERunCondition {
	RUN_CONDITION_DATA_DRIVEN,
	RUN_CONDITION_TIME_DRIVEN,
	RUN_CONDITION_CONTROL_DRIVEN,
} ERunCondition;

typedef enum _EPortType {
	PORT_TYPE_QUEUE,
	PORT_TYPE_BUFFER,
} EPortType;

typedef enum _EPortMapType {
	PORTMAP_TYPE_DISTRIBUTING,
	PORTMAP_TYPE_BROADCASTING,
} EPortMapType;

typedef enum _ETimeMetric {
	TIME_METRIC_CYCLE,
	TIME_METRIC_COUNT,
	TIME_METRIC_MICROSEC,
	TIME_METRIC_MILLISEC,
	TIME_METRIC_SEC,
	TIME_METRIC_MINUTE,
	TIME_METRIC_HOUR,
} ETimeMetric;


typedef enum _EPortDirection {
	PORT_DIRECTION_OUTPUT,
	PORT_DIRECTION_INPUT,
} EPortDirection;

typedef enum _EPortSampleRateType {
	PORT_SAMPLE_RATE_FIXED,
	PORT_SAMPLE_RATE_VARIABLE,
	PORT_SAMPLE_RATE_MULTIPLE,
} EPortSampleRateType;


typedef enum _ECommunicationType {
	COMMUNICATION_TYPE_SHARED_MEMORY,
	COMMUNICATION_TYPE_TCP_SERVER,
	COMMUNICATION_TYPE_TCP_CLIENT,
} ECommunicationType;

typedef enum _EChannelType {
	CHANNEL_TYPE_GENERAL,
	CHANNEL_TYPE_INPUT_ARRAY,
	CHANNEL_TYPE_OUTPUT_ARRAY,
	CHANNEL_TYPE_FULL_ARRAY,
} EChannelType;

typedef enum _ELoopType {
	LOOP_TYPE_CONVERGENT,
	LOOP_TYPE_DATA,
} ELoopType;

typedef enum _ETaskGraphType {
	GRAPH_TYPE_PROCESS_NETWORK,
	GRAPH_TYPE_DATAFLOW,
} ETaskGraphType;


typedef enum _EModeState {
	MODE_STATE_NORMAL,
	MODE_STATE_TRANSITING,
} EModeState;


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
} STask;



typedef struct _SPortSampleRate {
	char *pszModeName; // Except MTM, all mode name becomes "Default"
	int nSampleRate; // sample rate (for general task, nSampleRate and nTotalSampleRate are same)
	int nMaxAvailableDataNum; // for broadcast loop
} SPortSampleRate;


typedef struct _SLibrary {
	char *pszLibraryName;
	FnUemLibraryInit fnInit;
	FnUemLibraryWrapup fnWrapup;
} SLibrary;


typedef struct _SPort SPort;

// nBufSize /  (nTotalSampleRate *nSampleSize) => number of loop queue?

typedef struct _SPort {
	int nTaskId;
	char *pszPortName;
	EPortSampleRateType enSampleRateType;
	SPortSampleRate *astSampleRates; // If the task is MTM, multiple sample rates can be existed.
	int nNumOfSampleRates;
	int nCurrentSampleRateIndex;
	int nSampleSize;
	EPortType enPortType;
	SPort *pstSubGraphPort;
} SPort;


/*
typedef struct _SPortMap {
	int nTaskId;
	char *pszPortName;
	int nChildTaskId;
	char *pszChildTaskPortName;
	EPortDirection enPortDirection;
	EPortMapType enPortMapType;
} SPortMap;
*/

typedef struct _SChannel {
	int nChannelIndex;
	int nNextChannelIndex;
	ECommunicationType enType;
	EChannelType enChannelType;
	int nBufSize;
	SPort stInputPort;
	SPort stOutputPort;
	int nInitialDataLen;
	void *pChannelStruct;
} SChannel;

typedef struct _STaskIdToTaskMap {
	int nTaskId;
	char *pszTaskName;
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
	char *pszProcessorName;
	int nPoolSize;
} SProcessor;

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


extern uem_bool g_bSystenExit;

#ifdef __cplusplus
}
#endif

#include "uem_channel_data.h"

#endif /* SRC_API_INCLUDE_UEM_DATA_H_ */
