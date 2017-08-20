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
	PORTMAP_TYPE_BUFFER,
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
	int nThreadNum;
	STaskGraph *pstSubGraph;
	STaskGraph *pstParentGraph;
	SModeTransitionMachine *pstMTMInfo;
	HThreadMutex hMutex;
	HThreadEvent hEvent;
} STask;

typedef struct _SChunk {
	void *pChunkStart;
	void *pDataStart;
	void *pDataEnd;
	int nChunkDataLen;
	int nAvailableNum;
	int nMaxAvailableDataNum;
	HThreadMutex hChunkMutex; // chunk mutex
	HThreadEvent hChunkEvent; // chunk event for blocking/read
} SChunk;

typedef struct _SAvailableChunk SAvailableChunk;

typedef struct _SAvailableChunk {
	int nChunkIndex;
	SAvailableChunk *pstPrev;
	SAvailableChunk *pstNext;
} SAvailableChunk;


typedef struct _SPort {
	int nTaskId;
	char *pszPortName;
	int nSampleSize;
	EPortType enPortType;
	EPortMapType enPortMapType;
	SChunk *astChunk;
	int nChunkNum; // loop count number, all channel has at least one chunk
	int nChunkLen; // nChunkLen * nChunkNum = SChannel's nBufSize
	SAvailableChunk *astAvailableChunkList;
} SPort;


typedef struct _SChannel {
	int nChannelIndex;
	int nSampleSize;
	void *pBuffer;
	int nBufSize;
	HThreadMutex hMutex; // channel global mutex
	SPort stInputPort;
	SPort stOutputPort;
} SChannel;

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
