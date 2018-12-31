/*
 * UKAddOnHandler.c
 *
 *  Created on: 2018. 10. 26.
 *      Author: jej
 */


#ifdef HAVE_CONFIG_H
#include <config.h>
#endif


#include <uem_common.h>

#include <UKAddOnHandler.h>


uem_result UKAddOnHandler_Init()
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;

	for(nLoop = 0 ; nLoop < g_nAddOnNum ; nLoop++)
	{
		if(g_astAddOns[nLoop].fnInit != NULL)
		{
			result = g_astAddOns[nLoop].fnInit();
			ERRIFGOTO(result, _EXIT);
		}
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKAddOnHandler_Run()
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;

	for(nLoop = 0 ; nLoop < g_nAddOnNum ; nLoop++)
	{
		if(g_astAddOns[nLoop].fnRun != NULL)
		{
			result = g_astAddOns[nLoop].fnRun();
			ERRIFGOTO(result, _EXIT);
		}
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKAddOnHandler_Fini()
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;

	for(nLoop = 0 ; nLoop < g_nAddOnNum ; nLoop++)
	{
		if(g_astAddOns[nLoop].fnWrapup != NULL)
		{
			result = g_astAddOns[nLoop].fnWrapup();
			ERRIFGOTO(result, _EXIT);
		}
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}



