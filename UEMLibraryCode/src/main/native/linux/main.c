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
#include <signal.h>

#include <uem_common.h>

#include <uem_data.h>

#include <UCTime.h>
#include <UCThreadMutex.h>

#include <UKTask.h>
#include <UKChannel.h>
#include <UKMulticast.h>
#include <UKCPUTaskManager.h>
#include <UKProcessor.h>
#include <UKLibrary.h>
#include <UKTime.h>
#include <UKModule.h>

#include <UFSystem.h>

// not static which is used globally
HCPUTaskManager g_hCPUTaskManager = NULL;

uem_result createTasks(HCPUTaskManager hManager)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;
	uem_bool bIsCPU = FALSE;
	int nValue;
	ETimeMetric enMetric;


	result = UKTime_GetProgramExecutionTime(&nValue, &enMetric);
	ERRIFGOTO(result, _EXIT);

	if(nValue > 0 && enMetric == TIME_METRIC_COUNT)
	{
		result = UKTask_SetAllTargetIteration(nValue);
		ERRIFGOTO(result, _EXIT);
	}

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
		result = UKCPUTaskManager_RegisterTask(hManager, &(g_stMappingInfo.pstGeneralTaskMappingInfo[nLoop]));
		ERRIFGOTO(result, _EXIT);
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
#define MILLISEC_UNIT (1000)

long long getEndTime(long long llStartTime)
{
	long long llEndTime;
	int nValue;

	nValue = g_stExecutionTime.nValue;

	if(nValue == 0) // endless execution
	{
		llEndTime = 0;
	}
	else
	{
		switch(g_stExecutionTime.enTimeMetric)
		{
		case TIME_METRIC_COUNT: // end time is not used for count metric
			llEndTime = 0;
			break;
		case TIME_METRIC_CYCLE: // currently, same to 1 ms
			llEndTime = llStartTime + 1 * nValue;
			break;
		case TIME_METRIC_MICROSEC:
			if(nValue/MILLISEC_UNIT <= 0)
			{
				llEndTime = llStartTime + 1;
			}
			else
			{
				llEndTime = llStartTime + nValue/MILLISEC_UNIT;
			}
			break;
		case TIME_METRIC_MILLISEC:
			llEndTime = llStartTime + nValue;
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
	long long llStartTime;
	int nSleepAdd = 0;
	uem_bool bStopped = FALSE;

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

	llStartTime = llCurTime;
	llEndTime = getEndTime(llCurTime);

	// llEndTime == 0 means infinite loop
	while((llEndTime == 0 || (llEndTime > 0 && llEndTime >= llCurTime)) && g_bSystemExit == FALSE)
	{
		if(llEndTime > 0)
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
		}
		else // for endless execution, sleep long
		{
			UCTime_Sleep(DEFAULT_LONG_SLEEP_PERIOD);
			nSleepAdd += DEFAULT_LONG_SLEEP_PERIOD;

			result = UKCPUTaskManager_IsAllTaskStopped(hManager, &bStopped);
			ERRIFGOTO(result, _EXIT);

			if(bStopped == TRUE)
			{
				break;
			}
		}

		if(nSleepAdd > 0 && nSleepAdd % 10000 == 0)
		{
			printf("Elapsed time: %d seconds\n", (int) (llCurTime - llStartTime)/1000);
		}

		result = UCTime_GetCurTickInMilliSeconds(&llCurTime);
		ERRIFGOTO(result, _EXIT);
	}

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
	uem_result result;
	setvbuf(stdout, NULL, _IONBF, 0);
	setvbuf(stderr, NULL, _IONBF, 0);

#ifndef WIN32
	signal(SIGPIPE, SIG_IGN);
#endif

	printf("Program start\n");

	// module initialization
	result = UKModule_Initialize();
	ERRIFGOTO(result, _EXIT);

	UKLibrary_Initialize();

	// Channel initialization
	result = UKChannel_Initialize();
	ERRIFGOTO(result, _EXIT);

	// Multicast initialization
	result = UKMulticast_Initialize();
	ERRIFGOTO(result, _EXIT);

	// Execute tasks
	executeTasks();

	// Set stop flag (set TRUE to escape block of all tasks)
	UFSystem_Stop(0);

	UKMulticast_Finalize();
	// Channel finalization
	UKChannel_Finalize();
	UKLibrary_Finalize();

	UKModule_Finalize();
_EXIT:
	if(result != ERR_UEM_NOERROR)
	{
		printf("Error is occurred during execution: %d\n", result);
	}
	fflush(stdout);
	printf("Program end\n");

	return 0;
}



