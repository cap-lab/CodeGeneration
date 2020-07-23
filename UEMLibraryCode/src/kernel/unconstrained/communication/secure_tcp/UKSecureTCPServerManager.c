/*
 * UKSecureTCPServerManager.c
 *
 *  Created on: 2020. 5. 21.
 *      Author: jrkim
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <UCSecureTCPSocket.h>

#include <UKServiceRunner.h>

#include <uem_remote_data.h>

#include <uem_secure_tcp_data.h>


uem_result UKSecureTCPServerManager_Initialize()
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;

	result = UCSecureTCPSocket_Initialize();
	ERRIFGOTO(result, _EXIT);

#ifndef AGGREGATE_SECURE_TCP_CONNECTION
	for(nLoop = 0 ; nLoop < g_nSecureTCPServerInfoNum ; nLoop++)
	{
		result = UKServiceRunner_StartIndividualService(&(g_astSecureTCPServerInfo[nLoop].stServiceInfo), &(g_astSecureTCPServerInfo[nLoop].stSSLTCPInfo));
		ERRIFGOTO(result, _EXIT);
	}
#else
	for(nLoop = 0 ; nLoop < g_nSecureTCPAggregateServerInfoNum ; nLoop++)
	{
		result = UKServiceRunner_StartAggregatedService(&(g_astSecureTCPAggregateServerInfo[nLoop].stServiceInfo), &(g_astSecureTCPAggregateServerInfo[nLoop].stSSLTCPInfo));
		ERRIFGOTO(result, _EXIT);
	}

	for(nLoop = 0 ; nLoop < g_nSecureTCPAggregateClientInfoNum ; nLoop++)
	{
		result = UKServiceRunner_StartAggregatedClientService(&(g_astSecureTCPAggregateClientInfo[nLoop].stServiceInfo), &(g_astSecureTCPAggregateClientInfo[nLoop].stSSLTCPInfo));
		ERRIFGOTO(result, _EXIT);
	}
#endif
	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKSecureTCPServerManager_Finalize()
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;
#ifdef AGGREGATE_SECURE_TCP_CONNECTION
	for(nLoop = 0 ; nLoop < g_nSecureTCPAggregateClientInfoNum ; nLoop++)
	{
		result = UKServiceRunner_StopAggregatedClientService(&(g_astSecureTCPAggregateClientInfo[nLoop].stServiceInfo));
		ERRIFGOTO(result, _EXIT);
	}

	for(nLoop = 0 ; nLoop < g_nSecureTCPAggregateServerInfoNum ; nLoop++)
	{
		result = UKServiceRunner_StopAggregatedService(&(g_astSecureTCPAggregateServerInfo[nLoop].stServiceInfo));
		ERRIFGOTO(result, _EXIT);
	}
#else
	for(nLoop = 0 ; nLoop < g_nSecureTCPServerInfoNum ; nLoop++)
	{
		result = UKServiceRunner_StopIndividualService(&(g_astSecureTCPServerInfo[nLoop].stServiceInfo));
		ERRIFGOTO(result, _EXIT);
	}
#endif
	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}
