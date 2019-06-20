/*
 * UKBluetoothModule.c
 *
 *  Created on: 2018. 10. 8.
 *      Author: chjej202
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <uem_remote_data.h>
#include <uem_bluetooth_data.h>

#include <UKServiceRunner.h>

uem_result UKBluetoothModule_Initialize()
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;

	for(nLoop = 0 ; nLoop < g_nBluetoothSlaveNum ; nLoop++)
	{
		result = UKServiceRunner_StartAggregatedService(&(g_astBluetoothSlaveInfo[nLoop].stAggregateInfo), &(g_astBluetoothSlaveInfo[nLoop].stAccessInfo));
		ERRIFGOTO(result, _EXIT);
	}

	for(nLoop = 0 ; nLoop < g_nBluetoothMasterNum ; nLoop++)
	{
		result = UKServiceRunner_StartAggregatedClientService(&(g_astBluetoothMasterInfo[nLoop].stAggregateInfo), &(g_astBluetoothMasterInfo[nLoop].stAccessInfo));
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKBluetoothModule_Finalize()
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;

	for(nLoop = 0 ; nLoop < g_nBluetoothMasterNum ; nLoop++)
	{
		UKServiceRunner_StopAggregatedClientService(&(g_astBluetoothMasterInfo[nLoop].stAggregateInfo));
	}

	for(nLoop = 0 ; nLoop < g_nBluetoothSlaveNum ; nLoop++)
	{
		UKServiceRunner_StopAggregatedService(&(g_astBluetoothSlaveInfo[nLoop].stAggregateInfo));
	}

	result = ERR_UEM_NOERROR;
//_EXIT:
	return result;
}

