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

#ifndef AGGREGATE_TCP_CONNECTION
	for(nLoop = 0 ; nLoop < g_nTCPServerInfoNum ; nLoop++)
	{
		result = UKServiceRunner_StartIndividualService(&(g_astTCPServerInfo[nLoop].stServiceInfo), &(g_astTCPServerInfo[nLoop].stTCPInfo));
		ERRIFGOTO(result, _EXIT);
	}
#else
	for(nLoop = 0 ; nLoop < g_nTCPAggregateServerInfoNum ; nLoop++)
	{
		result = UKServiceRunner_StartAggregatedService(&(g_astTCPAggregateServerInfo[nLoop].stServiceInfo), &(g_astTCPAggregateServerInfo[nLoop].stTCPInfo));
		ERRIFGOTO(result, _EXIT);
	}

	for(nLoop = 0 ; nLoop < g_nTCPAggregateClientInfoNum ; nLoop++)
	{
		result = UKServiceRunner_StartAggregatedClientService(&(g_astTCPAggregateClientInfo[nLoop].stServiceInfo), &(g_astTCPAggregateClientInfo[nLoop].stTCPInfo));
		ERRIFGOTO(result, _EXIT);
	}
#endif
	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKTCPServerManager_Finalize()
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;
#ifdef AGGREGATE_TCP_CONNECTION
	for(nLoop = 0 ; nLoop < g_nTCPAggregateClientInfoNum ; nLoop++)
	{
		result = UKServiceRunner_StopAggregatedClientService(&(g_astTCPAggregateClientInfo[nLoop].stServiceInfo));
		ERRIFGOTO(result, _EXIT);
	}

	for(nLoop = 0 ; nLoop < g_nTCPAggregateServerInfoNum ; nLoop++)
	{
		result = UKServiceRunner_StopAggregatedService(&(g_astTCPAggregateServerInfo[nLoop].stServiceInfo));
		ERRIFGOTO(result, _EXIT);
	}
#else
	for(nLoop = 0 ; nLoop < g_nTCPServerInfoNum ; nLoop++)
	{
		result = UKServiceRunner_StopIndividualService(&(g_astTCPServerInfo[nLoop].stServiceInfo));
		ERRIFGOTO(result, _EXIT);
	}
#endif
	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}
