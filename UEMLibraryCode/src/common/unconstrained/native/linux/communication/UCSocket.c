/*
 * UCSocket.c
 *
 *  Created on: 2020. 1. 23.
 *      Author: JangryulKim
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <unistd.h>

#include <uem_common.h>

uem_result UCSocket_Initialize()
{
	return ERR_UEM_NOERROR;
}

uem_result UCSocket_Finalize()
{
	return ERR_UEM_NOERROR;
}

uem_result UCSocket_Close(int nSocketfd)
{
	uem_result result = ERR_UEM_UNKNOWN;

	close(nSocketfd);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

