/*
 * uem_data.h
 *
 *  Created on: 2017. 8. 12.
 *      Author: jej
 */

#ifndef SRC_API_INCLUDE_UEM_DATA_H_
#define SRC_API_INCLUDE_UEM_DATA_H_

#include <uem_common.h>

#ifdef __cplusplus
extern "C"
{
#endif

#define INVALID_TASK_ID (-1)
#define INVALID_CHANNEL_ID (-1)


typedef enum _ETaskType {
	TASK_TYPE_COMPUTATIONAL,
	TASK_TYPE_CONTROL,
	TASK_TYPE_LOOP,
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
} SModeTransitionMachine;

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
	int nThreadNum; // data-type loop count
	STaskGraph *pstSubGraph;
	STaskGraph *pstParentGraph;
	SModeTransitionMachine *pstMTMInfo;
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


// nBufSize /  (nTotalSampleRate *nSampleSize) => number of loop queue?

typedef struct _SPort {
	int nTaskId;
	char *pszPortName;
	int nTotalSampleRate; // for nested loop : this becomes outer loop task's sample size  (or most inner-task's sample size * (all loop counts except broadcasting port)
	int nSampleRate; // most inner-task's sample rate (for general task, nSampleRate and nTotalSampleRate are same)
	int nSampleSize;
	EPortType enPortType;
	SChunk *astChunk;
	int nChunkNum; // nTotalSampleRate / nSampleRate
	int nMaxAvailableDataNum; // for broadcast loop
	int nChunkLen; // nSampleRate * nSampleSize => maximum size of each chunk item
	SAvailableChunk *astAvailableChunkList; // Same size of nChunkNum
	SAvailableChunk *pstAvailableChunkHead;
	SAvailableChunk *pstAvailableChunkTail;
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
	void *pBuffer;
	int nBufSize;
	HThreadMutex hMutex; // channel global mutex
	SPort stInputPort;
	SPort stOutputPort;
} SChannel;


typedef struct _STaskIdToTaskMap {
	int nTaskId;
	STask *pstTask;
} TaskIdToTaskMap;

void Loop1_Replace_init();
void Loop1_Replace_go();
void Loop1_Replace_wrapup();


SChannel g_pstChannels[] = {
		{
			0,
			4,
			NULL,
			4,
			NULL,
			{
					0,
					"in_f",
					4,
					PORT_TYPE_QUEUE,
					PORTMAP_TYPE_DISTRIBUTING,
					NULL,
					1,
					0,
			},
			{
					0,
					"out_r",
					4,
					PORT_TYPE_QUEUE,
					PORTMAP_TYPE_DISTRIBUTING,
					NULL,
					1,
					0,
			},
		},
};

STask g_pstTopTasks[] = {
		{ 	0,
			"Loop1_Replace",
			TASK_TYPE_COMPUTATIONAL,
			Loop1_Replace_init,
			Loop1_Replace_go,
			Loop1_Replace_wrapup,
			RUN_CONDITION_DATA_DRIVEN,
			1,
			1,
			TIME_METRIC_MICROSEC,
			1,
			NULL,
			g_pstTopGraph,
			NULL,
			NULL,
			NULL,
		},
};

STaskGraph g_pstTopGraph[] = { g_pstTopTasks, NULL };


#ifdef __cplusplus
}
#endif

#endif /* SRC_API_INCLUDE_UEM_DATA_H_ */
