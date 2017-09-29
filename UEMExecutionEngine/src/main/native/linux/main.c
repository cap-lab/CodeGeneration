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


#include <UCThreadMutex.h>

uem_result createTasks()
{
	uem_result result = ERR_UEM_UNKNOWN;
	HThreadMutex hMutex = NULL;

	hMutex = NULL;


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
	// MTM initialize

	// control thread count++

	// pthread create TASK_ROUTINE (CONTROL)

	// pthread create (REST OF TASKS)

	// pthread create virtual tasks

	// check task running
	// until end_flag is set by top-level tasks

	// thread_cancel

	// thread cancel

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



