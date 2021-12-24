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
#include <signal.h>
#else
#include <windows.h>
#endif

#include <UCBasic.h>
#include <UCAlloc.h>
#include <UCThread.h>
#include <UCThreadEvent.h>

typedef struct _SThread {
	EUemModuleId enId;
#ifndef WIN32
	pthread_t hNativeThread;

#else
	HANDLE  hNativeThread;
#endif
	HThreadEvent hEvent;
	FnNativeThread fnThreadFunction;
	void *pUserData;
} SUCThread;

#ifndef WIN32

static void *posixThread(void *pParam)
{
	SUCThread *pstThread = NULL;

	pstThread = (SUCThread *) pParam;

	pstThread->fnThreadFunction(pstThread->pUserData);

	UCThreadEvent_SetEvent(pstThread->hEvent);

	return NULL;
}

static uem_result createPthread(SUCThread *pstThread)
{
	uem_result result = ERR_UEM_UNKNOWN;
	pthread_attr_t threadAttr;

	if(pthread_attr_init(&threadAttr) != 0) {
		ERRASSIGNGOTO(result, ERR_UEM_INTERNAL_FAIL, _EXIT);
	}

	if(pthread_create(&(pstThread->hNativeThread), &threadAttr, posixThread, (void *) pstThread) != 0) {
		ERRASSIGNGOTO(result, ERR_UEM_INTERNAL_FAIL, _EXIT);
	}
	result = ERR_UEM_NOERROR;
_EXIT:
	pthread_attr_destroy( &threadAttr );
	return result;
}

static uem_result destroyPosixThread(SUCThread *pstThread, uem_bool bDetach, int nTimeoutInMS)
{
	uem_result result = ERR_UEM_UNKNOWN;

	if(bDetach == TRUE)
	{
		if(pthread_detach(pstThread->hNativeThread) != 0) {
			// free the memory even though pthread_detach is failed.
			// debug exit for entering this case
			ERRASSIGNGOTO(result, ERR_UEM_INTERNAL_FAIL, _EXIT);
		}
	}
	else
	{
		if(nTimeoutInMS > 0)
		{
			result = UCThreadEvent_WaitTimeEvent(pstThread->hEvent, (long long) nTimeoutInMS);
			if(result == ERR_UEM_TIME_EXPIRED)
			{
				if(pthread_cancel(pstThread->hNativeThread) != 0) {
					pthread_kill(pstThread->hNativeThread, SIGKILL);
					UEM_DEBUG_PRINT("Thread is forcedly terminated.\n");
					ERRASSIGNGOTO(result, ERR_UEM_INTERNAL_FAIL, _EXIT);
				}
				UEM_DEBUG_PRINT("Thread cancellation request is delivered.\n");
				result = ERR_UEM_NOERROR;
			}
			ERRIFGOTO(result, _EXIT);
		}
		// otherwise, wait for infinite seconds

		if(pthread_join(pstThread->hNativeThread, NULL) != 0) {
			// free the memory even though pthread_join is failed.
			// debug exit for entering this case
			ERRASSIGNGOTO(result, ERR_UEM_INTERNAL_FAIL, _EXIT);
		}
	}

	result = ERR_UEM_NOERROR;
_EXIT:
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

static uem_result destroyWindowsThread(SUCThread *pstThread, uem_bool bDetach, int nTimeoutInMS)
{
	uem_result result = ERR_UEM_UNKNOWN;

	if(bDetach == FALSE)
	{
		DWORD dwErrorCode;

		if(nTimeoutInMS > 0)
		{
			dwErrorCode = WaitForSingleObject(pstThread->hNativeThread, nTimeoutInMS);
		}
		else
		{
			dwErrorCode = WaitForSingleObject(pstThread->hNativeThread, INFINITE);
		}

		if(dwErrorCode != 0) {
			switch(dwErrorCode)
			{
			case WAIT_TIMEOUT:
				UEM_DEBUG_PRINT("Thread wait timeout.\n");
				if(TerminateThread(pstThread->hNativeThread, 0) == 0)
				{
					ERRASSIGNGOTO(result, ERR_UEM_INTERNAL_FAIL, _EXIT);
				}
				UEM_DEBUG_PRINT("Thread is forcedly terminated.\n");
				break;
			case WAIT_ABANDONED:
			case WAIT_FAILED:
				UEM_DEBUG_PRINT("Fail to wait thread termination.\n");
				ERRASSIGNGOTO(result, ERR_UEM_INTERNAL_FAIL, _EXIT);
				break;
			default:
				ERRASSIGNGOTO(result, ERR_UEM_INTERNAL_FAIL, _EXIT);
			}
		}
	}
	else
	{
		// for detach, just close the handle without waiting
	}

	if( CloseHandle(pstThread->hNativeThread) == FALSE) {
		// free the memory even though CloseHandle is failed.
		// debug exit for entering this case
		ERRASSIGNGOTO(result, ERR_UEM_INTERNAL_FAIL, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


#endif

uem_result UCThread_Create(FnNativeThread fnThreadRoutine, void *pUserData, OUT HThread *phThread)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SUCThread *pstThread = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(phThread, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	IFVARERRASSIGNGOTO(fnThreadRoutine, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	pstThread = (SUCThread *) UCAlloc_malloc(sizeof(SUCThread));
	ERRMEMGOTO(pstThread, result, _EXIT);

	pstThread->enId = ID_UEM_THREAD;
	pstThread->fnThreadFunction = fnThreadRoutine;
	pstThread->pUserData = pUserData;
	pstThread->hEvent = NULL;

#ifndef WIN32
	result = UCThreadEvent_Create(&(pstThread->hEvent));
	ERRIFGOTO(result, _EXIT);

	result = createPthread(pstThread);
	ERRIFGOTO(result, _EXIT);
#else
	result = createWindowsThread(pstThread);
	ERRIFGOTO(result, _EXIT);
#endif

	*phThread = pstThread;

	result = ERR_UEM_NOERROR;
_EXIT:
	if(result != ERR_UEM_NOERROR && pstThread != NULL) {
		if(pstThread->hEvent != NULL) {
			UCThreadEvent_Destroy(&(pstThread->hEvent));
		}
		SAFEMEMFREE(pstThread);
	}

	return result;
}


uem_result UCThread_Destroy(HThread *phThread, uem_bool bDetach, int nTimeoutInMS)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SUCThread *pstThread = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(phThread, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	if(IS_VALID_HANDLE(*phThread, ID_UEM_THREAD) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}

	if(bDetach != TRUE && bDetach != FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}

	// 0 for infinite wait
	if(nTimeoutInMS < 0) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
#endif
	pstThread = (SUCThread *) *phThread;

#ifndef WIN32
	result = destroyPosixThread(pstThread, bDetach, nTimeoutInMS);
	ERRIFGOTO(result, _EXIT);
#else
	result = destroyWindowsThread(pstThread, bDetach, nTimeoutInMS);
	ERRIFGOTO(result, _EXIT);
#endif

	if(pstThread->hEvent != NULL) {
		UCThreadEvent_Destroy(&(pstThread->hEvent));
	}

	SAFEMEMFREE(pstThread);

	*phThread = NULL;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static int getCPUSetSize()
{
#ifndef WIN32
	return sizeof(cpu_set_t);
#else
	return sizeof(DWORD_PTR);
#endif
}

// Setting CPU is not portable in POSIX
#ifndef WIN32
static uem_result setCPUInLinux(SUCThread *pstThread, int nCoreId)
{
	uem_result result = ERR_UEM_UNKNOWN;
	cpu_set_t cpuset;
	int nError = 0;

	CPU_ZERO(&cpuset);
	CPU_SET(nCoreId, &cpuset);

	nError = pthread_setaffinity_np(pstThread->hNativeThread, sizeof(cpu_set_t), &cpuset);
	if(nError != 0) {
		ERRASSIGNGOTO(result, ERR_UEM_INTERNAL_FAIL, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}
#else
static uem_result setCPUInMinGW(SUCThread *pstThread, int nCoreId)
{
	uem_result result = ERR_UEM_UNKNOWN;
	DWORD_PTR dwThreadAffinityMask;
	DWORD_PTR dwOldAffinity = 0;

	dwThreadAffinityMask = 1 << nCoreId;

	dwOldAffinity = SetThreadAffinityMask(pstThread->hNativeThread, dwThreadAffinityMask);
	if(dwOldAffinity == 0) {
		ERRASSIGNGOTO(result, ERR_UEM_INTERNAL_FAIL, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}
#endif

// Setting CPU is not portable in POSIX
#ifndef WIN32
static uem_result setPriorityInLinux(SUCThread *pstThread, int nPriority)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nError = 0;

	nError = pthread_setschedprio(pstThread->hNativeThread, nPriority);
	if(nError != 0) {
		ERRASSIGNGOTO(result, ERR_UEM_INTERNAL_FAIL, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}
#else
static uem_result setPriorityInMinGW(SUCThread *pstThread, int nPriority)
{
	uem_result result = ERR_UEM_UNKNOWN;
	DWORD_PTR dwResult = 0;

	dwResult = SetThreadPriority(pstThread->hNativeThread, nPriority);
	if(dwResult == 0) {
		ERRASSIGNGOTO(result, ERR_UEM_INTERNAL_FAIL, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}
#endif

void UCThread_Yield()
{
#ifndef WIN32
	pthread_yield();
#else
	SwitchToThread();
#endif
}


uem_result UCThread_SetMappedCPU(HThread hThread, int nCoreId)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SUCThread *pstThread = NULL;
#ifdef ARGUMENT_CHECK
	if(IS_VALID_HANDLE(hThread, ID_UEM_THREAD) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}

	if(nCoreId >= getCPUSetSize()) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
#endif
	pstThread = (SUCThread *) hThread;

#ifndef WIN32
	result = setCPUInLinux(pstThread, nCoreId);
#else
	result = setCPUInMinGW(pstThread, nCoreId);
#endif
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}
/*
uem_result UCThread_GetCPUAffinityMask(HThread hThread, unsigned long long *pnThreadAffinity)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SUCThread *pstThread = NULL;
#ifdef ARGUMENT_CHECK
	if(IS_VALID_HANDLE(hThread, ID_UEM_THREAD) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}

	IFVARERRASSIGNGOTO(pnThreadAffinity, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	pstThread = (SUCThread *) hThread;

#ifndef WIN32
	//pthread_getaffinity_np(pthread_t thread, size_t cpusetsize, cpu_set_t *cpuset);
#else
	SetThreadAffinityMask(hThread, dwThreadAffinityMask);

	SetThreadAffinityMask(hThread, dwThreadAffinityMask);
#endif

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}*/

uem_result UCThread_SetPriority(HThread hThread, int nScheduler, int nPriority)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SUCThread *pstThread = NULL;
#ifdef ARGUMENT_CHECK
	if(IS_VALID_HANDLE(hThread, ID_UEM_THREAD) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}

	if(nPriority < sched_get_priority_min(nScheduler) || nPriority > sched_get_priority_max(nScheduler)) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
#endif
	pstThread = (SUCThread *) hThread;

#ifndef WIN32
	result = setPriorityInLinux(pstThread, nPriority);
#else
	result = setPriorityInMinGW(pstThread, nPriority);
#endif
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}
