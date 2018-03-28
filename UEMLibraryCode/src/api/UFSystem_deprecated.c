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

void SYS_REQ_KILL()
{
	UFSystem_Kill();
}


void SYS_REQ_STOP()
{
	UFSystem_Stop();
}


