/*
 * UCCPUSchedule.c
 *
 *  Created on: 2017. 9. 27.
 *      Author: jej
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <sched.h>

#include <UCCPUSchedule.h>

typedef struct _SCPUSchedule {
	//cpu_set_t cpu_set;
} SCPUSchedule;

uem_result UCCPUSchedule_Init(SCPUSchedule *pstCPU)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UCCPUSchedule_SetCPU(SCPUSchedule *pstCPU);
uem_result UCCPUSchedule_ClearCPU(SCPUSchedule *pstCPU);

