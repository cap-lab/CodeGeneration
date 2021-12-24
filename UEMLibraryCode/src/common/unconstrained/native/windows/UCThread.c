/*
 * UCThread.c
 *
 *  Created on: 2019. 11. 29.
 *      Author: jej
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

//#undef WIN32

#include <windows.h>

#include <UCBasic.h>
#include <UCAlloc.h>
#include <UCThread.h>
#include <UCThreadEvent.h>

typedef struct _SThread {
	EUemModuleId enId;
	HANDLE  hNativeThread;
	HThreadEvent hEvent;
	FnNativeThread fnThreadFunction;
	void *pUserData;
} SUCThread;


static DWORD WINAPI windowsThread (void *pParam)
{
	SUCThread *pstThread = NULL;

	pstThread = (SUCThread *) pParam;

	pstThread->fnThreadFunction(pstThread->pUserData);

	return 0;
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

	pstThread->hNativeThread = CreateThread(NULL, 0, windowsThread, pstThread, 0, NULL);
	if(pstThread->hNativeThread == NULL) {
		ERRASSIGNGOTO(result, ERR_UEM_INTERNAL_FAIL, _EXIT);
	}

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

	result = destroyWindowsThread(pstThread, bDetach, nTimeoutInMS);
	ERRIFGOTO(result, _EXIT);

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
	return sizeof(DWORD_PTR);
}


void UCThread_Yield()
{
	SwitchToThread();
}


uem_result UCThread_SetMappedCPU(HThread hThread, int nCoreId)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SUCThread *pstThread = NULL;
	DWORD_PTR dwThreadAffinityMask;
	DWORD_PTR dwOldAffinity = 0;
#ifdef ARGUMENT_CHECK
	if(IS_VALID_HANDLE(hThread, ID_UEM_THREAD) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}

	if(nCoreId >= getCPUSetSize()) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
#endif
	pstThread = (SUCThread *) hThread;

	dwThreadAffinityMask = 1 << nCoreId;

	dwOldAffinity = SetThreadAffinityMask(pstThread->hNativeThread, dwThreadAffinityMask);
	if(dwOldAffinity == 0) {
		ERRASSIGNGOTO(result, ERR_UEM_INTERNAL_FAIL, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UCThread_SetPriority(HThread hThread, int nScheduler, int nPriority)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SUCThread* pstThread = NULL;
	DWORD_PTR dwResult = 0;	
	int nWindowsPriority = nPriority - 3; // priority range: (user) 1 ~ 5 -> (system) -2 ~ 2
#ifdef ARGUMENT_CHECK
	if (IS_VALID_HANDLE(hThread, ID_UEM_THREAD) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}

	if (nWindowsPriority != THREAD_PRIORITY_IDLE && nWindowsPriority != THREAD_PRIORITY_LOWEST 
		&& nWindowsPriority != THREAD_PRIORITY_BELOW_NORMAL && nWindowsPriority != THREAD_PRIORITY_NORMAL
		&& nWindowsPriority != THREAD_PRIORITY_ABOVE_NORMAL && nWindowsPriority != THREAD_PRIORITY_HIGHEST
		&& nWindowsPriority != THREAD_PRIORITY_TIME_CRITICAL) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
#endif
	pstThread = (SUCThread*)hThread;
	
	// priority range: (user) 1 ~ 5 -> (system) -2 ~ 2
	dwResult = SetThreadPriority(pstThread->hNativeThread, -3 + nPriority);
	if (dwResult == 0) {
		dwResult = GetLastError();
		ERRASSIGNGOTO(result, ERR_UEM_INTERNAL_FAIL, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}
