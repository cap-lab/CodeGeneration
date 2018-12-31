/*
 * UFSystem_deprecated.c
 *
 *  Created on: 2017. 8. 24.
 *      Author: jej
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <UFSystem.h>

#ifndef API_LITE
void SYS_REQ_KILL(int nCallerTaskId)
{
	UFSystem_Kill(nCallerTaskId);
}


void SYS_REQ_STOP(int nCallerTaskId)
{
	UFSystem_Stop(nCallerTaskId);
}
#endif

