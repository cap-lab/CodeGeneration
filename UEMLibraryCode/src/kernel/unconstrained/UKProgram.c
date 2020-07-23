/*
 * UKProgram.c
 *
 *  Created on: 2020. 7. 23.
 *      Author: jrkim
 */


#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <uem_data.h>

#include <UKProgram.h>

uem_result UKTime_GetProgramExecutionTime(OUT int *pnValue, OUT ETimeMetric *penMetric)
{
	uem_result result = ERR_UEM_UNKNOWN;

#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(pnValue, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	IFVARERRASSIGNGOTO(penMetric, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif

	*pnValue = g_stExecutionTime.nValue;
	*penMetric = g_stExecutionTime.enTimeMetric;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}
