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

void MatA_init();
void MatA_go();
void MatA_wrapup();

void MatB_init();
void MatB_go();
void MatB_wrapup();

void VecMul_init();
void VecMul_go();
void VecMul_wrapup();

void Display_init();
void Display_go();
void Display_wrapup();

#define CHANNEL_0_SIZE (3)
#define CHANNEL_1_SIZE (3)
#define CHANNEL_2_SIZE (36)

char pChannel_0_buffer[CHANNEL_0_SIZE];
char pChannel_1_buffer[CHANNEL_1_SIZE];
char pChannel_2_buffer[CHANNEL_2_SIZE];


SChunk *g_astChunk_sdf_matrix_channel_0_out[] = {
	{
		pChannel_0_buffer, // Chunk start pointer
		pChannel_0_buffer, // Data start pointer
		pChannel_0_buffer, // Data end pointer
		0, // wWitten data length
		0, // Available data number;
		NULL, // Chunk mutex
		NULL, // Chunk conditional variable
	},
};

SChunk *g_astChunk_sdf_matrix_channel_0_in[] = {
	{
		pChannel_0_buffer, // Chunk start pointer
		pChannel_0_buffer, // Data start pointer
		pChannel_0_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number (for broadcast loop)
		NULL, // Chunk mutex
		NULL, // Chunk conditional variable
	},
};

SChunk *g_astChunk_sdf_matrix_channel_1_out[] = {
	{
		pChannel_1_buffer, // Chunk start pointer
		pChannel_1_buffer, // Data start pointer
		pChannel_1_buffer, // Data end pointer
		0, // wWitten data length
		0, // Available data number;
		NULL, // Chunk mutex
		NULL, // Chunk conditional variable
	},
};

SChunk *g_astChunk_sdf_matrix_channel_1_in[] = {
	{
		pChannel_1_buffer, // Chunk start pointer
		pChannel_1_buffer, // Data start pointer
		pChannel_1_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number (for broadcast loop)
		NULL, // Chunk mutex
		NULL, // Chunk conditional variable
	},
};

SChunk *g_astChunk_sdf_matrix_channel_2_out[] = {
	{
		pChannel_2_buffer, // Chunk start pointer
		pChannel_2_buffer, // Data start pointer
		pChannel_2_buffer, // Data end pointer
		0, // wWitten data length
		0, // Available data number;
		NULL, // Chunk mutex
		NULL, // Chunk conditional variable
	},
};

SChunk *g_astChunk_sdf_matrix_channel_2_in[] = {
	{
		pChannel_2_buffer, // Chunk start pointer
		pChannel_2_buffer, // Data start pointer
		pChannel_2_buffer, // Data end pointer
		0, // Written data length
		0, // Available data number (for broadcast loop)
		NULL, // Chunk mutex
		NULL, // Chunk conditional variable
	},
};

SPortSampleRate g_astPortSampleRate_sdf_matrix_MatA_out[] = {
	{ "Default", // Mode name
	   1, // Sample rate
	   1, // Available number of data
	},
};


SPortSampleRate g_astPortSampleRate_sdf_matrix_VecMul_in1[] = {
	{ "Default", // Mode name
	   1, // Sample rate
	   1, // Available number of data
	},
};


SPortSampleRate g_astPortSampleRate_sdf_matrix_MatB_out[] = {
	{ "Default", // Mode name
	   1, // Sample rate
	   1, // Available number of data
	},
};


SPortSampleRate g_astPortSampleRate_sdf_matrix_VecMul_in2[] = {
	{ "Default", // Mode name
	   1, // Sample rate
	   1, // Available number of data
	},
};


SPortSampleRate g_astPortSampleRate_sdf_matrix_VecMul_out[] = {
	{ "Default", // Mode name
	   1, // Sample rate
	   1, // Available number of data
	},
};


SPortSampleRate g_astPortSampleRate_sdf_matrix_Display_in[] = {
	{ "Default", // Mode name
	   9, // Sample rate
	   1, // Available number of data
	},
};


// ports which located inside a subgraph and port-mapped to outer graph
//SPort g_astPortMapList[] = {
//	// all port information which are port-mapped by other ports
//};


SChannel g_astChannels[] = {
	{
		0, // Channel ID
		CHANNEL_TYPE_SHARED_MEMORY, // Channel type
		pChannel_0_buffer, // Channel buffer pointer
		CHANNEL_0_SIZE, // Channel size
		pChannel_0_buffer, // Channel data start
		0, // Channel data length
		NULL, // Mutex
		{
			2, // Task ID
			"in1", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_sdf_matrix_VecMul_in1, // Array of sample rate list
			1, // Array element number of sample rate list
			12, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			0, // Task ID
			"out", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_sdf_matrix_MatA_out, // Array of sample rate list
			1, // Array element number of sample rate list
			12, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
		{
			g_astChunk_sdf_matrix_channel_0_in, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Input chunk information
		{
			g_astChunk_sdf_matrix_channel_0_out, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Output chunk information
		NULL, // Available chunk list
		NULL, // Chunk list head
		NULL, // Chunk list tail
	},
	{
		1, // Channel ID
		CHANNEL_TYPE_SHARED_MEMORY, // Channel type
		pChannel_1_buffer, // Channel buffer pointer
		CHANNEL_1_SIZE, // Channel size
		pChannel_1_buffer, // Channel data start
		0, // Channel data length
		NULL, // Mutex
		{
			2, // Task ID
			"in2", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_sdf_matrix_VecMul_in2, // Array of sample rate list
			1, // Array element number of sample rate list
			12, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			1, // Task ID
			"out", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_sdf_matrix_MatB_out, // Array of sample rate list
			1, // Array element number of sample rate list
			12, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
		{
			g_astChunk_sdf_matrix_channel_1_in, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Input chunk information
		{
			g_astChunk_sdf_matrix_channel_1_out, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Output chunk information
		NULL, // Available chunk list
		NULL, // Chunk list head
		NULL, // Chunk list tail
	},
	{
		2, // Channel ID
		CHANNEL_TYPE_SHARED_MEMORY, // Channel type
		pChannel_2_buffer, // Channel buffer pointer
		CHANNEL_2_SIZE, // Channel size
		pChannel_2_buffer, // Channel data start
		0, // Channel data length
		NULL, // Mutex
		{
			3, // Task ID
			"in", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_sdf_matrix_Display_in, // Array of sample rate list
			1, // Array element number of sample rate list
			36, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Input port information
		{
			2, // Task ID
			"out", // Port name
			PORT_SAMPLE_RATE_FIXED, // Port sample rate type
			g_astPortSampleRate_sdf_matrix_VecMul_out, // Array of sample rate list
			1, // Array element number of sample rate list
			4, // Sample size
			PORT_TYPE_QUEUE, // Port type
			NULL, // Pointer to Subgraph port
		}, // Output port information
		{
			g_astChunk_sdf_matrix_channel_2_in, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Input chunk information
		{
			g_astChunk_sdf_matrix_channel_2_out, // Array of chunk
			1, // Chunk number
			1, // Chunk size
		}, // Output chunk information
		NULL, // Available chunk list
		NULL, // Chunk list head
		NULL, // Chunk list tail
	},
};

STask g_astTopTasks[] = {
	{ 	0, // Task ID
		"MatA", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		MatA_init, // Task init function
		MatA_go, // Task go function
		MatA_wrapup, // Task wrapup function
		RUN_CONDITION_TIME_DRIVEN, // Run condition
		1, // Run rate
		10, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // Subgraph
		g_pstTopGraph, // Parent task graph
		NULL, // MTM information
		NULL, // Loop information
		FALSE, // Statically scheduled or not
		NULL, // Mutex
		NULL, // Conditional variable
	},
	{ 	1, // Task ID
		"MatB", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		MatB_init, // Task init function
		MatB_go, // Task go function
		MatB_wrapup, // Task wrapup function
		RUN_CONDITION_TIME_DRIVEN, // Run condition
		1, // Run rate
		10, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // subgraph
		g_pstTopGraph, //parent task graph
		NULL, // MTM information
		NULL, // Loop information
		FALSE, // statically scheduled (run by difference tasks)
		NULL, // mutex
		NULL, // conditional variable
	},
	{ 	2, // Task ID
		"VecMul", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		VecMul_init, // Task init function
		VecMul_go, // Task go function
		VecMul_wrapup, // Task wrapup function
		RUN_CONDITION_DATA_DRIVEN, // Run condition
		1, // Run rate
		10, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // subgraph
		g_pstTopGraph, //parent task graph
		NULL, // MTM information
		NULL, // Loop information
		FALSE, // statically scheduled (run by difference tasks)
		NULL, // mutex
		NULL, // conditional variable
	},
	{ 	3, // Task ID
		"Display", // Task name
		TASK_TYPE_COMPUTATIONAL, // Task Type
		VecMul_init, // Task init function
		VecMul_go, // Task go function
		VecMul_wrapup, // Task wrapup function
		RUN_CONDITION_DATA_DRIVEN, // Run condition
		1, // Run rate
		10, // Period
		TIME_METRIC_MICROSEC, // Period metric
		NULL, // subgraph
		g_pstTopGraph, //parent task graph
		NULL, // MTM information
		NULL, // Loop information
		FALSE, // statically scheduled (run by difference tasks)
		NULL, // mutex
		NULL, // conditional variable
	},
};

STaskGraph g_pstTopGraph[] = { g_astTopTasks, NULL };

STaskIdToTaskMap g_astTaskIdToTask[] = {
	{ 	0, // Task ID
		g_astTopTasks[0] // Task structure pointer
	},
	{ 	1, // Task ID
		g_astTopTasks[1] // Task structure pointer
	},
	{ 	2, // Task ID
		g_astTopTasks[2] // Task structure pointer
	},
};

SProcessor g_astProcessorInfo[] = {
	{ 	0, // Processor ID
		"i7_0", // Processor name
		4, // Processor pool size
	},
};

SMappingSchedulingInfo g_astMappingAndSchedulingInfo[] = {
	{	TASK_TYPE_COMPUTATIONAL, // Task type
		{ .nTaskId = 0 }, // Task ID or composite task information
		0, // Processor ID
		0, // Processor local ID
	},
	{	TASK_TYPE_COMPUTATIONAL, // Task type
		{ .nTaskId = 1 }, // Task ID or composite task information
		0, // Processor ID
		1, // Processor local ID
	},
	{	TASK_TYPE_COMPUTATIONAL, // Task type
		{ .nTaskId = 2 }, // Task ID or composite task information
		0, // Processor ID
		2, // Processor local ID
	},
	{	TASK_TYPE_COMPUTATIONAL, // Task type
		{ .nTaskId = 3 }, // Task ID or composite task information
		0, // Processor ID
		3, // Processor local ID
	},
};


#ifdef __cplusplus
}
#endif

#endif /* SRC_API_INCLUDE_UEM_DATA_H_ */
