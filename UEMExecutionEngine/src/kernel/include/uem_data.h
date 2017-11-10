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
#define INVALID_CHANNEL_ID (-1)
#define INVALID_CHUNK_INDEX (-1)
#define INVALID_TIMER_SLOT_ID (-1)
#define VARIABLE_SAMPLE_RATE (-1)
#define MAPPING_NOT_SPECIFIED (-1)
#define CHUNK_NUM_NOT_INITIALIZED (-1)


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

typedef void (*FnUemTaskInit)(int nTaskId);
typedef void (*FnUemTaskGo)();
typedef void (*FnUemTaskWrapup)();


typedef uem_bool (*FnTaskModeTranstion)();

typedef struct _STask STask;

typedef struct _SModeMap {
	int nModeId;
	char *pszModeName;
	STask **pastRelatedChildTasks;
	int nRelatedChildTaskNum;
} SModeMap;


typedef struct _SVariableIntMap {
	int nVariableId;
	char *pszVariableName;
	int nValue;
} SVariableIntMap;


typedef struct _SModeTransitionMachine {
	int nTaskId;
	SModeMap *astModeMap;
	SVariableIntMap *astVarIntMap;
	FnTaskModeTranstion fnTransition;
	int nCurModeIndex;
} SModeTransitionMachine;

typedef struct _SLoopInfo {
	ELoopType enType;
	int nLoopCount;
	int nDesignatedTaskId;
} SLoopInfo;

typedef struct _STaskGraph {
	ETaskGraphType enType;
	STask *astTasks;
	STask *pstParentTask;
} STaskGraph;


typedef union _UParamValue {
	int nParam;
	double dbParam;
} UParamValue;

typedef struct _STaskParameter {
	int nParamId;
	const char *pszParamName;
	UParamValue uParamValue;
} STaskParameter;

typedef struct _STaskFunctions {
	FnUemTaskInit fnInit;
	FnUemTaskGo fnGo;
	FnUemTaskWrapup fnWrapup;
} STaskFunctions;

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
	uem_bool bStaticScheduled; // TRUE if a task is mapped or scheduled
	HThreadMutex hMutex;
	HThreadEvent hEvent;
} STask;

typedef struct _SChunk {
	void *pChunkStart; // fixed
	void *pDataStart; // vary
	void *pDataEnd; // vary
	int nChunkDataLen; // written data length
	int nAvailableDataNum; // for broadcast loop
} SChunk;

typedef struct _SAvailableChunk SAvailableChunk;

typedef struct _SAvailableChunk {
	int nChunkIndex;
	int nSampleNum;
	SAvailableChunk *pstPrev;
	SAvailableChunk *pstNext;
} SAvailableChunk;

typedef struct _SPortSampleRate {
	char *pszModeName; // Except MTM, all mode name becomes "Default"
	int nSampleRate; // sample rate (for general task, nSampleRate and nTotalSampleRate are same)
	int nMaxAvailableDataNum; // for broadcast loop
} SPortSampleRate;


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

typedef struct _SChunkInfo {
	// These values can be changed during execution depending on Mode transition
	SChunk *astChunk;
	int nChunkNum; // nTotalSampleRate / nSampleRate
	int nChunkLen; // nSampleRate * nSampleSize => maximum size of each chunk item
} SChunkInfo;

typedef struct _SChannel {
	int nChannelIndex;
	ECommunicationType enType;
	EChannelType enChannelType;
	void *pBuffer;
	int nBufSize;
	void *pDataStart;
	void *pDataEnd;
	int nDataLen;
	int nReferenceCount;
	HThreadMutex hMutex; // Channel global mutex
	HThreadEvent hEvent; // Channel global conditional variable

	SPort stInputPort;
	SPort stOutputPort;
	SChunkInfo stInputPortChunk;
	SChunkInfo stOutputPortChunk;
	int nWrittenOutputChunkNum;

	// These values can be changed during execution depending on Mode transition
	SAvailableChunk *astAvailableInputChunkList; // size
	int nMaxChunkNum; // maximum chunk size for all port sample rate cases
	SAvailableChunk *pstAvailableInputChunkHead;
	SAvailableChunk *pstAvailableInputChunkTail;

} SChannel;


typedef struct _STaskIdToTaskMap {
	int nTaskId;
	char *pszTaskName;
	STask *pstTask;
} STaskIdToTaskMap;

typedef struct _SScheduleItem {
	int nTaskId;
	FnUemTaskGo fnGo;
	int nRepetition;
} SScheduleItem;


typedef struct _SScheduleList {
	int nScheduleId;
	SScheduleItem *astScheduleItemList;
	int nScheduleItemNum;
	int nThroughputConstraint;
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

typedef union _UMappingTarget {
	STask *pstTask;
	SScheduledTasks *pstScheduledTasks;
} UMappingTarget;


typedef struct _SMappingSchedulingInfo {
	ETaskType enType;
	UMappingTarget uMappedTask;
	int nProcessorId;
	int nLocalId;
} SMappingSchedulingInfo;

extern SChannel g_astChannels[];
extern int g_nChannelNum;

extern STask g_astTasks_top[];
extern int g_nNumOfTasks_top;

extern STaskGraph g_stGraph_top;

extern STaskIdToTaskMap g_astTaskIdToTask[];
extern int g_nTaskIdToTaskNum;

extern SProcessor g_astProcessorInfo[];
extern int g_nProcessorInfoNum;

extern SMappingSchedulingInfo g_astMappingAndSchedulingInfo[];
extern int g_nMappingAndSchedulingInfoNum;

#ifdef __cplusplus
}
#endif

#endif /* SRC_API_INCLUDE_UEM_DATA_H_ */
