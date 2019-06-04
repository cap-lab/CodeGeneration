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
 * @brief
 *
 * This function
 *
 * @param pstTask
 *
 * @return
 */
uem_result UKTaskControl_RunTask(STask *pstTask);

/**
 * @brief
 *
 * This function
 *
 * @param pstTask
 *
 * @return
 */
uem_result UKTaskControl_StopTask(STask *pstTask);

/**
 * @brief
 *
 * This function
 *
 * @param pstTask
 *
 * @return
 */
uem_result UKTaskControl_StoppingTask(STask *pstTask);

/**
 * @brief
 *
 * This function
 *
 * @param pstTask
 * @param penTaskState
 *
 * @return
 */
uem_result UKTaskControl_GetTaskState(STask *pstTask, EInternalTaskState *penTaskState);

#ifdef __cplusplus
}
#endif


#endif /* SRC_KERNEL_CONSTRAINED_INCLUDE_UKTASKCONTROL_H_ */
