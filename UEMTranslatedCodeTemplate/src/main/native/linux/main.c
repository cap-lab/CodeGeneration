/*
 * main.c
 *
 *  Created on: 2017. 9. 7.
 *      Author: jej
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif


#include <stdio.h>

#include <uem_common.h>

#include <uem_data.h>

#include <UCTime.h>
#include <UCThreadMutex.h>

#include <UKTask.h>
#include <UKChannel.h>
#include <UKCPUTaskManager.h>
#include <UKProcessor.h>
#include <UKLibrary.h>

// not static which is used globally
HCPUTaskManager g_hCPUTaskManager = NULL;

uem_result createTasks(HCPUTaskManager hManager)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;
	uem_bool bIsCPU = FALSE;

	/*for(nLoop = 0 ; nLoop < g_nMappingAndSchedulingInfoNum ; nLoop++)
	{
		result = UKProcessor_IsCPUByProcessorId(g_astMappingAndSchedulingInfo[nLoop].nProcessorId, &bIsCPU);
		ERRIFGOTO(result, _EXIT);

		if(bIsCPU == TRUE)
		{
			if(g_astMappingAndSchedulingInfo[nLoop].enType == TASK_TYPE_COMPOSITE)
			{
				result = UKCPUTaskManager_RegisterCompositeTask(hManager, g_astMappingAndSchedulingInfo[nLoop].uMappedTask.pstScheduledTasks,
															g_astMappingAndSchedulingInfo[nLoop].nLocalId);
				ERRIFGOTO(result, _EXIT);
			}
			else // TASK_TYPE_CONTROL, TASK_TYPE_LOOP, TASK_TYPE_COMPUTATIONAL
			{
				result = UKCPUTaskManager_RegisterTask(hManager, g_astMappingAndSchedulingInfo[nLoop].uMappedTask.pstTask, g_astMappingAndSchedulingInfo[nLoop].nLocalId);
				ERRIFGOTO(result, _EXIT);
			}
		}
	}*/


	for(nLoop = 0 ; nLoop < g_stMappingInfo.nMappedCompositeTaskNum ; nLoop++)
	{
		result = UKProcessor_IsCPUByProcessorId(g_stMappingInfo.pstCompositeTaskMappingInfo[nLoop].nProcessorId, &bIsCPU);
		ERRIFGOTO(result, _EXIT);

		if(bIsCPU == TRUE)
		{
			result = UKCPUTaskManager_RegisterCompositeTask(hManager, &(g_stMappingInfo.pstCompositeTaskMappingInfo[nLoop]));
			ERRIFGOTO(result, _EXIT);
		}
	}

	for(nLoop = 0 ; nLoop < g_stMappingInfo.nMappedGeneralTaskNum ; nLoop++)
	{
		result = UKProcessor_IsCPUByProcessorId(g_stMappingInfo.pstGeneralTaskMappingInfo[nLoop].nProcessorId, &bIsCPU);
		ERRIFGOTO(result, _EXIT);

		if(bIsCPU == TRUE)
		{
			result = UKCPUTaskManager_RegisterTask(hManager, &(g_stMappingInfo.pstGeneralTaskMappingInfo[nLoop]));
			ERRIFGOTO(result, _EXIT);
		}
	}

_EXIT:
	return result;
}


uem_result runTasks()
{
	uem_result result = ERR_UEM_UNKNOWN;



	return result;
}

uem_result destroyTasks()
{
	uem_result result = ERR_UEM_UNKNOWN;



	return result;
}

#define SEC_UNIT (1000)
#define MINUTE_UNIT (60)
#define HOUR_UNIT (60)

long long getEndTime(long long llStartTime)
{
	long long llEndTime;
	int nValue;

	nValue = g_stExecutionTime.nValue;

	switch(g_stExecutionTime.enTimeMetric)
	{
	case TIME_METRIC_COUNT: // currently, same to 1 ms
		llEndTime = llStartTime + 1 * nValue;
		break;
	case TIME_METRIC_CYCLE: // currently, same to 1 ms
		llEndTime = llStartTime + 1 * nValue;
		break;
	case TIME_METRIC_MICROSEC:
		llEndTime = llStartTime + 1 * nValue;
		break;
	case TIME_METRIC_MILLISEC:
		llEndTime = llStartTime + 1 * nValue;
		break;
	case TIME_METRIC_SEC:
		llEndTime = llStartTime + SEC_UNIT * nValue;
		break;
	case TIME_METRIC_MINUTE:
		llEndTime = llStartTime + SEC_UNIT * MINUTE_UNIT * nValue;
		break;
	case TIME_METRIC_HOUR:
		llEndTime = llStartTime + SEC_UNIT * MINUTE_UNIT * HOUR_UNIT * nValue;
		break;
	default:
		llEndTime = llStartTime + 1;
		break;
	}

	return llEndTime;
}

#define DEFAULT_LONG_SLEEP_PERIOD (100)
#define DEFAULT_SHORT_SLEEP_PERIOD (10)

uem_result executeTasks()
{
	uem_result result = ERR_UEM_UNKNOWN;
	HCPUTaskManager hManager = NULL;
	long long llCurTime;
	long long llEndTime;
	int nSleepAdd = 0;

	result = UKTask_Initialize();
	ERRIFGOTO(result, _EXIT);

	result = UKCPUTaskManager_Create(&hManager);
	ERRIFGOTO(result, _EXIT);

	// set global variable to be used outside of this thread
	g_hCPUTaskManager = hManager;

	result = createTasks(hManager);
	ERRIFGOTO(result, _EXIT);

	result = UKCPUTaskManager_RunRegisteredTasks(hManager);
	ERRIFGOTO(result, _EXIT);

	result = UCTime_GetCurTickInMilliSeconds(&llCurTime);
	ERRIFGOTO(result, _EXIT);

	llEndTime = getEndTime(llCurTime);

	while(llEndTime >= llCurTime)
	{
		if(llCurTime + DEFAULT_LONG_SLEEP_PERIOD <= llEndTime)
		{
			UCTime_Sleep(DEFAULT_LONG_SLEEP_PERIOD);
			nSleepAdd += DEFAULT_LONG_SLEEP_PERIOD;
		}
		else if(llCurTime + DEFAULT_SHORT_SLEEP_PERIOD <= llEndTime)
		{
			UCTime_Sleep(DEFAULT_SHORT_SLEEP_PERIOD);
			nSleepAdd += DEFAULT_SHORT_SLEEP_PERIOD;
		}
		else
		{
			// otherwise, busy wait
		}

		if(nSleepAdd % 10000 == 0)
		{
			printf("Merong world: %d\n", nSleepAdd);
		}

		result = UCTime_GetCurTickInMilliSeconds(&llCurTime);
		ERRIFGOTO(result, _EXIT);
	}

	// MTM initialize

	// control thread count++

	// pthread create TASK_ROUTINE (CONTROL)

	// pthread create (REST OF TASKS)

	// pthread create virtual tasks

	// check task running
	// until end_flag is set by top-level tasks

	// thread_cancel

	// thread cancel



_EXIT:
	if(hManager != NULL)
	{
		result = UKCPUTaskManager_Destroy(&hManager);
	}
	UKTask_Finalize();
	return result;
}

int main(int argc, char *argv[])
{
	setvbuf(stdout, NULL, _IONBF, 0);
	setvbuf(stderr, NULL, _IONBF, 0);


	printf("Hello world: %d\n", g_astChannels[0].nChannelIndex);

	// Channel initialization
	UKLibrary_Initialize();
	UKChannel_Initialize();

	// Execute tasks
	executeTasks();

	// Channel finalization
	UKChannel_Finalize();
	UKLibrary_Finalize();

	printf("Merong world: huhu\n");

	return 0;
}



