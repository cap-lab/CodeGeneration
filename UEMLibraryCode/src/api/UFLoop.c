/*
 * UFLoop.c
 *
 *  Created on: 2018. 8. 30.
 *      Author: DG-SHIN
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <UFLoop.h>

uem_result UFLoop_GetIteration(IN int nCallerTaskId, IN int nTaskThreadId, OUT int *pnTaskIteration)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = UKLoop_GetLoopTaskIteration(nCallerTaskId, nTaskThreadId, pnTaskIteration);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UFLoop_StopNextIteration(IN int nCallerTaskId)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = UKLoop_StopNextIteration(nCallerTaskId);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}
