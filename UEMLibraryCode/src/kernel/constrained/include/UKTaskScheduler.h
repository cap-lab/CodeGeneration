/*
 * UKTaskScheduler.h
 *
 *  Created on: 2018. 9. 6.
 *      Author: chjej202
 */

#ifndef SRC_KERNEL_CONSTRAINED_INCLUDE_UKTASKSCHEDULER_H_
#define SRC_KERNEL_CONSTRAINED_INCLUDE_UKTASKSCHEDULER_H_

#include <uem_common.h>

#include <uem_data.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct _SGeneralTaskRuntimeInfo {
	STask *pstTask;
	unsigned long ulNextTime;
	int nRunCount;
	uem_bool bRunning;
} SGeneralTaskRuntimeInfo;


typedef struct _SCompositeTaskRuntimeInfo {
	SScheduledTasks *pstCompositeTaskSchedule;
	unsigned long ulNextTime;
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

uem_result UKTaskScheduler_Init();
uem_result UKTaskScheduler_Run();

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_CONSTRAINED_INCLUDE_UKTASKSCHEDULER_H_ */
