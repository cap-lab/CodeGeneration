/*
 * UKTCPServerManager.c
 *
 *  Created on: 2018. 6. 2.
 *      Author: jej
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <UKServiceRunner.h>

#include <uem_remote_data.h>

#include <uem_tcp_data.h>


uem_result UKTCPServerManager_Initialize()
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;

	for(nLoop = 0 ; nLoop < g_nTCPServerInfoNum ; nLoop++)
	{
		result = UKServiceRunner_StartIndividualService(&(g_astTCPServerInfo[nLoop].stServiceInfo), &(g_astTCPServerInfo[nLoop].stTCPInfo));
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKTCPServerManager_Finalize()
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;

	for(nLoop = 0 ; nLoop < g_nTCPServerInfoNum ; nLoop++)
	{
		result = UKServiceRunner_StopIndividualService(&(g_astTCPServerInfo[nLoop].stServiceInfo));
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}
