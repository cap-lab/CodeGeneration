/*
 * UKSerialModule.c
 *
 *  Created on: 2019. 02. 18., modified from UKBluetoothModule.c
 *      Author: dowhan1128
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <uem_remote_data.h>
#include <uem_serial_data.h>

#include <UKServiceRunner.h>

uem_result UKSerialModule_Initialize()
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;

	for(nLoop = 0 ; nLoop < g_nSerialSlaveInfoNum ; nLoop++)
	{
		result = UKServiceRunner_StartAggregatedService(&(g_astSerialSlaveInfo[nLoop].stAggregateInfo), &(g_astSerialSlaveInfo[nLoop].stAccessInfo));
		ERRIFGOTO(result, _EXIT);
	}

	for(nLoop = 0 ; nLoop < g_nSerialMasterInfoNum ; nLoop++)
	{
		result = UKServiceRunner_StartAggregatedClientService(&(g_astSerialMasterInfo[nLoop].stAggregateInfo), &(g_astSerialMasterInfo[nLoop].stAccessInfo));
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKSerialModule_Finalize()
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;

	for(nLoop = 0 ; nLoop < g_nSerialMasterInfoNum ; nLoop++)
	{
		UKServiceRunner_StopAggregatedClientService(&(g_astSerialMasterInfo[nLoop].stAggregateInfo));
	}

	for(nLoop = 0 ; nLoop < g_nSerialSlaveInfoNum ; nLoop++)
	{
		UKServiceRunner_StopAggregatedService(&(g_astSerialSlaveInfo[nLoop].stAggregateInfo));
	}

	result = ERR_UEM_NOERROR;
//_EXIT:
	return result;
}

