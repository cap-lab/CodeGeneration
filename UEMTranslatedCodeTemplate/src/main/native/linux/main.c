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

uem_result executeTasks()
{
	uem_result result = ERR_UEM_UNKNOWN;
	HCPUTaskManager hManager = NULL;

	result = UKCPUTaskManager_Create(&hManager);
	ERRIFGOTO(result, _EXIT);

	result = createTasks(hManager);
	ERRIFGOTO(result, _EXIT);

	result = UKCPUTaskManager_RunRegisteredTasks(hManager);
	ERRIFGOTO(result, _EXIT);

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



