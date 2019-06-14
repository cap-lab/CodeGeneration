/*
 * UKTaskControl.h
 *
 *  Created on: 2019. 4. 2.
 *      Author: chjej202
 */

#ifndef SRC_KERNEL_CONSTRAINED_INCLUDE_UKTASKCONTROL_H_
#define SRC_KERNEL_CONSTRAINED_INCLUDE_UKTASKCONTROL_H_


#include <uem_data.h>

#include <UKTask.h>


#ifdef __cplusplus
extern "C"
{
#endif

typedef struct _SGeneralTaskRuntimeInfo {
	STask *pstTask;
	uem_time tNextTime;
	int nRunCount;
	uem_bool bRunning;
} SGeneralTaskRuntimeInfo;


typedef struct _SCompositeTaskRuntimeInfo {
	SScheduledTasks *pstCompositeTaskSchedule;
	uem_time tNextTime;
	int nRunCount;
	uem_bool bRunning;
} SCompositeTaskRuntimeInfo;

// no mapping info with task list (leaf task, no static scheduling)
// static schedule

extern SGeneralTaskRuntimeInfo g_astControlTaskRuntimeInfo[];
extern SGeneralTaskRuntimeInfo g_astGeneralTaskRuntimeInfo[];
extern SCompositeTaskRuntimeInfo g_astCompositeTaskRuntimeInfo[];

extern int g_nControlTaskNum;
extern int g_nGeneralTaskNum;
extern int g_nCompositeTaskNum;

/**
 * @brief Execute task.
 *
 * This function makes target task to be run. \n
 * This function cleans subgraph channels, and do running function. \n
 * if task is static scheduled and has subgraph, initialize task runtimeinfo and call its subtasks' task init function.\n
 * else if task has subgraph, initialize task runtimeinfo and call its subtasks' task init function. \n
 * if task is a single general task, initialize task runtimeinfo and call task init function.
 *
 * @param pstTask target task pointer to be run.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_ILLEGAL_CONTROL for invalid channelAPI clear function while cleaning subgraph channels. \n
 * @ref ERR_UEM_NOT_FOUND if task not found. \n
 * @ref ERR_UEM_ILLEGAL_DATA if target task contains Process Network type subGraph.
 */
uem_result UKTaskControl_RunTask(STask *pstTask);

/**
 * @brief Stop task execution.
 *
 * This function makes target task to be stopped. \n
 * if target task is static scheduled and has subgraph, set runtimeinfo running value false and calls its subtasks' task wrapup function. \n
 * otherwise, set runtimeinfo running value false and calls its task wrapup function.
 *
 * @param pstTask target task pointer to be stopped.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_NOT_FOUND if task not found. \n
 * @ref ERR_UEM_ILLEGAL_DATA if target task contains Process Network type subGraph.
 *
 */
uem_result UKTaskControl_StopTask(STask *pstTask);

/**
 * @brief (not used) Stop stopping task.
 *
 * this function stops stopping task. \n
 * work exactly smae as @ref UKTaskControl_StopTask, since constrained device does not support stopping task.
 *
 * @param pstTask target task pointer to be stopped.
 *
  @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_NOT_FOUND if task not found. \n
 * @ref ERR_UEM_ILLEGAL_DATA if target task contains Process Network type subGraph.
 */
uem_result UKTaskControl_StoppingTask(STask *pstTask);

/**
 * @brief Return the task status information.
 *
 * This function gets task state information. \n
 * it checks task running state by task runtimeinfo running value. \n
 * if target task is static scheduled and has subgraph,
 * if at least one subtask is running, this function returns INTERNAL_STATE_RUN. \n
 * otherwise, this function returns INTERNAL_STATE_STOP.
 *
 * @param pstTask target task to get state from.
 * @param penTaskState returned task state value.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_NOT_FOUND if task not found. \n
 * @ref ERR_UEM_ILLEGAL_DATA if target task contains Process Network type subGraph.
 */
uem_result UKTaskControl_GetTaskState(STask *pstTask, EInternalTaskState *penTaskState);

#ifdef __cplusplus
}
#endif


#endif /* SRC_KERNEL_CONSTRAINED_INCLUDE_UKTASKCONTROL_H_ */
