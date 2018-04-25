/*
 * UFSystem.c
 *
 *  Created on: 2017. 8. 12.
 *      Author: jej
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif


#include <uem_common.h>

#include <UCSystem.h>

#include <UFSystem.h>

#include <uem_data.h>


void UFSystem_Kill(int nCallerTaskId)
{
	UCSystem_Exit();
}


void UFSystem_Stop(int nCallerTaskId)
{
	g_bSystemExit = TRUE;
}


