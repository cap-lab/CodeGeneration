/*
 * UCTime.c
 *
 *  Created on: 2018. 9. 6.
 *      Author: chjej202
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <Arduino.h>

#include <uem_common.h>

// Function name is same but argument type is different compared to unconstrained target UCTime_GetCurTickInMilliSeconds
uem_result UCTime_GetCurTickInMilliSeconds(unsigned long *pulTime)
{
	uem_result result = ERR_UEM_UNKNOWN;
	unsigned long ulMiiliSec = 0;
#ifdef ARGUMENT_CHECK
    IFVARERRASSIGNGOTO(pulTime, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	ulMiiliSec = millis();

	*pulTime = ulMiiliSec;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}
