/*
 * UCSocket.c
 *
 *  Created on: 2020. 1. 23.
 *      Author: JangryulKim
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <winsock2.h>

uem_result UCSocket_Initialize()
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nError = 0;
	WSADATA wsadata;

	nError = WSAStartup(MAKEWORD(2, 2), &wsadata);
	if(nError != 0)
	{
		ERRASSIGNGOTO(result, ERR_UEM_INTERNAL_FAIL, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UCSocket_Finalize()
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nError = 0;

	nError = WSACleanup();
	if(nError != 0)
	{
		ERRASSIGNGOTO(result, ERR_UEM_INTERNAL_FAIL, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UCSocket_Close(int nSocketfd)
{
	uem_result result = ERR_UEM_UNKNOWN;

	closesocket(nSocketfd);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

