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
#define VARIABLE_SAMPLE_RATE (-1)
#define MAPPING_NOT_SPECIFIED (-1)


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


typedef enum _EChannelType {
	CHANNEL_TYPE_SHARED_MEMORY,
	CHANNEL_TYPE_TCP_SERVER,
	CHANNEL_TYPE_TCP_CLIENT,
} EChannelType;

typedef enum _ELoopType {
	LOOP_TYPE_CONVERGENT,
	LOOP_TYPE_DATA,
} ELoopType;

typedef void (*FnUemTaskInit)(int nTaskId);
typedef void (*FnUemTaskGo)();
typedef void (*FnUemTaskWrapup)();


typedef uem_bool (*FnTaskModeTranstion)();

typedef struct _STask STask;

typedef struct _SModeMap {
	int nModeId;
	char *pszModeName;
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
	char *pszCurrentMode;
} SModeTransitionMachine;

typedef struct _SLoopInfo {
	ELoopType enType;
	int nLoopCount;
	int nDesignatedTaskId;
} SLoopInfo;

typedef struct _STaskGraph {
	STask *astTasks;
	STask *pstParentTask;
} STaskGraph;


typedef struct _STask {
	int nTaskId;
	const char *pszTaskName;
	ETaskType enType;
	FnUemTaskInit fnInit;
	FnUemTaskGo fnGo;
	FnUemTaskWrapup fnWrapup;
	ERunCondition enRunCondition;
	int nRunRate;
	int nPeriod;
	ETimeMetric enPeriodMetric;
	STaskGraph *pstSubGraph;
	STaskGraph *pstParentGraph;
	SModeTransitionMachine *pstMTMInfo;
	SLoopInfo *pstLoopInfo;
	uem_bool bStaticScheduled; // static-scheduled task
	HThreadMutex hMutex;
	HThreadEvent hEvent;
} STask;

typedef struct _SChunk {
	void *pChunkStart; // fixed
	char *pDataStart; // vary
	char *pDataEnd; // vary
	int nChunkDataLen; // written data length
	int nAvailableDataNum; // for broadcast loop
	HThreadMutex hChunkMutex; // chunk mutex
	HThreadEvent hChunkEvent; // chunk event for blocking/read
} SChunk;

typedef struct _SAvailableChunk SAvailableChunk;

typedef struct _SAvailableChunk {
	int nChunkIndex;
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
	EChannelType enType;
	void *pBuffer;
	int nBufSize;
	void *pDataStart;
	int nDataLen;
	HThreadMutex hMutex; // channel global mutex

	SPort stInputPort;
	SPort stOutputPort;
	SChunkInfo stInputPortChunk;
	SChunkInfo stOutputPortChunk;

	// These values can be changed during execution depending on Mode transition
	SAvailableChunk *astAvailableInputChunkList; // Same size of nChunkNum
	SAvailableChunk *pstAvailableInputChunkHead;
	SAvailableChunk *pstAvailableInputChunkTail;

} SChannel;

typedef struct _STaskIdToTaskMap {
	int nTaskId;
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

typedef struct _SScheduleMode {
	char *pszModeName;
	FnUemTaskInit *afnInitList;
	FnUemTaskWrapup *afnWrapupList;
	int nScheduledTaskNum;
	SScheduleList *astScheduleList;
	int nScheduleNum;
} SScheduleMode;

typedef struct _SScheduledTasks {
	SScheduleMode *astScheduleModeList;
	int nNumOfScheduleMode;
} SScheduledTasks;

typedef struct _SProcessor {
	int nProcessorId;
	char *pszProcessorName;
	int nPoolSize;
} SProcessor;

typedef union _UMappingTarget {
	int nTaskId;
	SScheduledTasks stScheduledTasks;
} UMappingTarget;

typedef struct _SMappingSchedulingInfo {
	ETaskType enType;
	UMappingTarget uMappedTask;
	int nProcessorId;
	int nLocalId;
} SMappingSchedulingInfo;

extern SChannel g_astChannels[];
extern STask g_astTopTasks[];
extern STaskGraph g_pstTopGraph[];
extern STaskIdToTaskMap g_astTaskIdToTask[];
extern SProcessor g_astProcessorInfo[];
extern SMappingSchedulingInfo g_astMappingAndSchedulingInfo[];


#ifdef __cplusplus
}
#endif

#endif /* SRC_API_INCLUDE_UEM_DATA_H_ */
