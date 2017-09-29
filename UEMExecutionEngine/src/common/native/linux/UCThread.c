/*
 * UCThread.c
 *
 *  Created on: 2017. 8. 10.
 *      Author: jej
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

//#undef WIN32

// For MinGW, use Windows Native API because MinGW does not provide pthread_attr_setaffinity_np function
#ifndef WIN32
#define _GNU_SOURCE
#include <pthread.h>
#else
#include <windows.h>
#endif

#include <UCBasic.h>

#include <UCThread.h>

typedef struct _SThread {
	EUemModuleId enId;
#ifndef WIN32
	pthread_t hNativeThread;
#else
	HANDLE  hNativeThread;
#endif
	FnNativeThread fnThreadFunction;
	void *pUserData;
} SUCThread;

#ifndef WIN32
static uem_result createPthread(SUCThread *pstThread)
{
	uem_result result = ERR_UEM_UNKNOWN;
	pthread_attr_t threadAttr;

	if(pthread_attr_init(&threadAttr) != 0) {
		ERRASSIGNGOTO(result, ERR_UEM_INTERNAL_FAIL, _EXIT);
	}

	if(pthread_create(&(pstThread->hNativeThread), &threadAttr, pstThread->fnThreadFunction, (void *) pstThread->pUserData) != 0) {
		ERRASSIGNGOTO(result, ERR_UEM_INTERNAL_FAIL, _EXIT);
	}
	result = ERR_UEM_NOERROR;
_EXIT:
	pthread_attr_destroy( &threadAttr );
	return result;
}

#else
static DWORD WINAPI windowsThread (void *pParam)
{
	SUCThread *pstThread = NULL;

	pstThread = (SUCThread *) pParam;

	pstThread->fnThreadFunction(pstThread->pUserData);

	return 0;
}

static uem_result createWindowsThread(SUCThread *pstThread)
{
	uem_result result = ERR_UEM_UNKNOWN;

	pstThread->hNativeThread = CreateThread(NULL, 0, windowsThread, pstThread, 0, NULL);
	if(pstThread->hNativeThread == NULL) {
		ERRASSIGNGOTO(result, ERR_UEM_INTERNAL_FAIL, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}
#endif

uem_result UCThread_Create(FnNativeThread fnThreadRoutine, void *pUserData, HThread *phThread)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SUCThread *pstThread = NULL;

	IFVARERRASSIGNGOTO(phThread, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	IFVARERRASSIGNGOTO(fnThreadRoutine, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	pstThread = (SUCThread *) UC_malloc(sizeof(SUCThread));
	ERRMEMGOTO(pstThread, result, _EXIT);

	pstThread->enId = ID_UEM_THREAD;
	pstThread->fnThreadFunction = fnThreadRoutine;
	pstThread->pUserData = pUserData;

#ifndef WIN32
	result = createPthread(pstThread);
	ERRIFGOTO(result, _EXIT);
#else
	result = createWindowsThread(pstThread);
	ERRIFGOTO(result, _EXIT);
#endif

	*phThread = pstThread;

	result = ERR_UEM_NOERROR;
_EXIT:
	if(result != ERR_UEM_NOERROR) {
		SAFEMEMFREE(pstThread);
	}

	return result;
}


uem_result UCThread_Destroy(HThread *phThread)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SUCThread *pstThread = NULL;

	IFVARERRASSIGNGOTO(phThread, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	if(IS_VALID_HANDLE(*phThread, ID_UEM_THREAD) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}

	pstThread = (SUCThread *) *phThread;

#ifndef WIN32
	if(pthread_join(pstThread->hNativeThread, NULL) != 0) {
		// free the memory even though pthread_join is failed.
		ERRASSIGNGOTO(result, ERR_UEM_INTERNAL_FAIL, _EXIT);
	}
#else
	{
		DWORD dwErrorCode;

		dwErrorCode = WaitForSingleObject(pstThread->hNativeThread, INFINITE);
		if(dwErrorCode != 0) {
			// ignore error
			// possible error cases
			// WAIT_ABANDONED, WAIT_OBJECT_0, WAIT_TIMEOUT, WAIT_FAILED
		}
	}
#endif

	SAFEMEMFREE(pstThread);

	*phThread = NULL;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UCThread_SetCPUAffinityMask(HThread hThread, uem_size nCPUSetSize, uem_cpu_set cpu_set)
{
	uem_result result = ERR_UEM_UNKNOWN;

	//pthread_attr_setaffinity_np();

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}




uem_result UCThread_GetCPUAffinityMask(HThread hThread, uem_size nCPUSetSize, uem_cpu_set cpu_set)
{
	uem_result result = ERR_UEM_UNKNOWN;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}




