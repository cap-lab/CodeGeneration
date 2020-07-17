/*
 * UKSSLTCPServerManager.c
 *
 *  Created on: 2020. 5. 21.
 *      Author: jrkim
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <UCSSLTCPSocket.h>

#include <UKServiceRunner.h>

#include <uem_remote_data.h>

#include <uem_ssl_tcp_data.h>


uem_result UKSSLTCPServerManager_Initialize()
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;

	result = UCSSLTCPSocket_Initialize();
	ERRIFGOTO(result, _EXIT);

#ifndef AGGREGATE_SSL_TCP_CONNECTION
	for(nLoop = 0 ; nLoop < g_nSSLTCPServerInfoNum ; nLoop++)
	{
		result = UKServiceRunner_StartIndividualService(&(g_astSSLTCPServerInfo[nLoop].stServiceInfo), &(g_astSSLTCPServerInfo[nLoop].stSSLTCPInfo));
		ERRIFGOTO(result, _EXIT);
	}
#else
	for(nLoop = 0 ; nLoop < g_nSSLTCPAggregateServerInfoNum ; nLoop++)
	{
		result = UKServiceRunner_StartAggregatedService(&(g_astSSLTCPAggregateServerInfo[nLoop].stServiceInfo), &(g_astSSLTCPAggregateServerInfo[nLoop].stSSLTCPInfo));
		ERRIFGOTO(result, _EXIT);
	}

	for(nLoop = 0 ; nLoop < g_nSSLTCPAggregateClientInfoNum ; nLoop++)
	{
		result = UKServiceRunner_StartAggregatedClientService(&(g_astSSLTCPAggregateClientInfo[nLoop].stServiceInfo), &(g_astSSLTCPAggregateClientInfo[nLoop].stSSLTCPInfo));
		ERRIFGOTO(result, _EXIT);
	}
#endif
	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKSSLTCPServerManager_Finalize()
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;
#ifdef AGGREGATE_SSL_TCP_CONNECTION
	for(nLoop = 0 ; nLoop < g_nSSLTCPAggregateClientInfoNum ; nLoop++)
	{
		result = UKServiceRunner_StopAggregatedClientService(&(g_astSSLTCPAggregateClientInfo[nLoop].stServiceInfo));
		ERRIFGOTO(result, _EXIT);
	}

	for(nLoop = 0 ; nLoop < g_nSSLTCPAggregateServerInfoNum ; nLoop++)
	{
		result = UKServiceRunner_StopAggregatedService(&(g_astSSLTCPAggregateServerInfo[nLoop].stServiceInfo));
		ERRIFGOTO(result, _EXIT);
	}
#else
	for(nLoop = 0 ; nLoop < g_nSSLTCPServerInfoNum ; nLoop++)
	{
		result = UKServiceRunner_StopIndividualService(&(g_astSSLTCPServerInfo[nLoop].stServiceInfo));
		ERRIFGOTO(result, _EXIT);
	}
#endif
	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}
