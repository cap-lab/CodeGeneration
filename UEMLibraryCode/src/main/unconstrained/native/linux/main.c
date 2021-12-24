/*
 * main.c
 *
 *  Created on: 2017. 9. 7.
 *      Author: jej
 *      Changed :
 *  	    1. 2019. 06. 20. wecracy
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif


#include <stdio.h>
#include <signal.h>
#include <sched.h>
#include <errno.h>
#include <string.h>

#include <uem_common.h>

#include <UKChannel.h>
#include <UKMulticast.h>
#include <UKLibrary.h>
#include <UKModule.h>

#include <UFSystem.h>

#include <UEMMainCommon.h>

uem_result setScheduler() {
	uem_result result = ERR_UEM_NOERROR;

	if(g_nScheduler != SCHEDULER_OTHER) {
		int nMaxPriority = sched_get_priority_max(g_nScheduler);

		struct sched_param schedparam = {
			.sched_priority = nMaxPriority,
		};

		int ret = sched_setscheduler(0, g_nScheduler, &schedparam);
		if (ret == -1) {
			printf("Scheduler configuration failed: %d %s\n", errno, strerror(errno));
			result = ERR_UEM_ILLEGAL_CONTROL;
		}
	}

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

	// set scheduler
	result = setScheduler();
	ERRIFGOTO(result, _EXIT);

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
	UEMMainCommon_ExecuteTasks();

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



