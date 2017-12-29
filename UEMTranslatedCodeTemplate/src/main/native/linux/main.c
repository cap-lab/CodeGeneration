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

#include <UKChannel.h>
#include <UKCPUTaskManager.h>


#include <UCThreadMutex.h>

uem_result createTasks(HCPUTaskManager hManager)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;

	for(nLoop = 0 ; nLoop < g_nMappingAndSchedulingInfoNum ; nLoop++)
	{
		// TODO: processorId must be checked to distinguish CPU or GPU
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



	//UKCPUTaskManager_RegisterCompositeTask(hManager, , )


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

	result = UKCPUTaskManager_Create(&hManager);
	ERRIFGOTO(result, _EXIT);

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
		}
		else if(llCurTime + DEFAULT_SHORT_SLEEP_PERIOD <= llEndTime)
		{
			UCTime_Sleep(DEFAULT_SHORT_SLEEP_PERIOD);
		}
		else
		{
			// otherwise, busy wait
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

	result = UKCPUTaskManager_Destroy(&hManager);
	ERRIFGOTO(result, _EXIT);

_EXIT:
	return result;
}

int main(int argc, char *argv[])
{
	printf("Hello world: %d\n", g_astChannels[0].nChannelIndex);

	// Channel initialization
	UKChannel_Initialize();

	// Execute tasks
	executeTasks();

	// Channel finalization
	UKChannel_Finalize();

	return 0;
}



