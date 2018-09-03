/*
 * UKTask.c
 *
 *  Created on: 2018. 8. 28.
 *      Author: chjej202
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <uem_data.h>

uem_result UKTask_Initialize()
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;

	for(nLoop = 0 ; nLoop < g_nTaskIdToTaskNum ; nLoop++)
	{
		g_astTaskIdToTask[nLoop].nTaskId;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	if(result != ERR_UEM_NOERROR)
	{
		UKTask_Finalize();
	}
	return result;
}


void UKTask_Finalize()
{
	int nLoop = 0;

	for(nLoop = 0 ; nLoop < g_nTaskIdToTaskNum ; nLoop++)
	{

	}
}



