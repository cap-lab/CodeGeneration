/*
 * Module_ev3dev.c
 *
 *  Created on: 2018. 6. 15.
 *      Author: jej
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <ev3.h>
#include <ev3_sensor.h>
#include <ev3_tacho.h>

#include <uem_common.h>

uem_result Module_ev3dev_Initialize()
{
	uem_result result = ERR_UEM_UNKNOWN;
	int ret = 0;

	ret = ev3_init();

	if(ret < 0)
	{
		ERRASSIGNGOTO(result, ERR_UEM_INTERNAL_FAIL, _EXIT);
	}

	ret = ev3_sensor_init();
	if(ret < 0)
	{
		ERRASSIGNGOTO(result, ERR_UEM_INTERNAL_FAIL, _EXIT);
	}

	ret = ev3_tacho_init();
	if(ret < 0)
	{
		ERRASSIGNGOTO(result, ERR_UEM_INTERNAL_FAIL, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result Module_ev3dev_Finalize()
{
	ev3_uninit();
_EXIT:
	return ERR_UEM_NOERROR;
}
