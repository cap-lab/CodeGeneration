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

#ifndef API_LITE
#include <UCSystem.h>
#endif

#include <UFSystem.h>

#include <uem_data.h>

#ifndef API_LITE
void UFSystem_Kill(int nCallerTaskId)
{
	UCSystem_Exit();
}


void UFSystem_Stop(int nCallerTaskId)
{
	g_bSystemExit = TRUE;
}
#endif

