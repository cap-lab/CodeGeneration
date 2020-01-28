/*
 * UCThreadEvent.c
 *
 *  Created on: 2019. 11. 29.
 *      Author: jej
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <windows.h>

#include <UCBasic.h>
#include <UCAlloc.h>
#include <UCThreadEvent.h>

typedef struct _SThreadEvent {
	EUemModuleId enId;
	uem_bool bIsStatic;
	HANDLE hEvent;
} SThreadEvent;

uem_result UCThreadEvent_Create(HThreadEvent *phEvent)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SThreadEvent *pstEvent = NULL;

#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(phEvent, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	pstEvent = (SThreadEvent *) UCAlloc_malloc(sizeof(SThreadEvent));
	ERRMEMGOTO(pstEvent, result, _EXIT);

	pstEvent->enId = ID_UEM_THREAD_EVENT;
	pstEvent->hEvent = NULL;

	pstEvent->hEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
	if(pstEvent->hEvent == NULL)
	{
		ERRASSIGNGOTO(result, ERR_UEM_INTERNAL_FAIL, _EXIT);
	}

	*phEvent = pstEvent;

	result = ERR_UEM_NOERROR;
_EXIT:
	if (result != ERR_UEM_NOERROR) {
		SAFEMEMFREE(pstEvent);
	}
	return result;
}

uem_result UCThreadEvent_SetEvent(HThreadEvent hEvent)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SThreadEvent *pstEvent = NULL;
	BOOL bSuccess;
#ifdef ARGUMENT_CHECK
	if (IS_VALID_HANDLE(hEvent, ID_UEM_THREAD_EVENT) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}
#endif
	pstEvent = (SThreadEvent *) hEvent;

	bSuccess = SetEvent(pstEvent->hEvent);
	if(bSuccess == FALSE)
	{
		ERRASSIGNGOTO(result, ERR_UEM_INTERNAL_FAIL, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UCThreadEvent_ClearEvent(HThreadEvent hEvent)
{   
    uem_result result = ERR_UEM_UNKNOWN;
    SThreadEvent *pstEvent = NULL;
    BOOL bSuccess;
#ifdef ARGUMENT_CHECK
    if (IS_VALID_HANDLE(hEvent, ID_UEM_THREAD_EVENT) == FALSE) {
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
    }
#endif
    pstEvent = (SThreadEvent *) hEvent;
    
    bSuccess = ResetEvent(pstEvent->hEvent);
	if(bSuccess == FALSE)
	{
		ERRASSIGNGOTO(result, ERR_UEM_INTERNAL_FAIL, _EXIT);
	}

    result = ERR_UEM_NOERROR;
_EXIT:
    return result;
}


uem_result UCThreadEvent_WaitEvent(HThreadEvent hEvent)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SThreadEvent *pstEvent = NULL;
	DWORD dwErrorCode = WAIT_FAILED;
#ifdef ARGUMENT_CHECK
	if (IS_VALID_HANDLE(hEvent, ID_UEM_THREAD_EVENT) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}
#endif
	pstEvent = (SThreadEvent *) hEvent;

	dwErrorCode = WaitForSingleObject(pstEvent->hEvent, INFINITE);
	if(dwErrorCode != WAIT_OBJECT_0)
	{
		ERRASSIGNGOTO(result, ERR_UEM_MUTEX_ERROR, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UCThreadEvent_WaitTimeEvent(HThreadEvent hEvent, long long llSleepTimeMs)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SThreadEvent *pstEvent = NULL;
	DWORD dwErrorCode = WAIT_FAILED;
	DWORD dwMilliseconds;
#ifdef ARGUMENT_CHECK
	if (IS_VALID_HANDLE(hEvent, ID_UEM_THREAD_EVENT) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}

	if(llSleepTimeMs <= 0) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
#endif
	pstEvent = (SThreadEvent *) hEvent;

	if(llSleepTimeMs > (long long) (unsigned long long) INFINITE )
	{
		dwMilliseconds = INFINITE;
	}
	else
	{
		dwMilliseconds = (unsigned long) llSleepTimeMs;
	}

	dwErrorCode = WaitForSingleObject(pstEvent->hEvent, dwMilliseconds);
	if(dwErrorCode != WAIT_OBJECT_0)
	{
		ERRASSIGNGOTO(result, ERR_UEM_MUTEX_ERROR, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UCThreadEvent_Destroy(HThreadEvent *phEvent)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SThreadEvent *pstEvent = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(phEvent, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	if (IS_VALID_HANDLE(*phEvent, ID_UEM_THREAD_EVENT) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}
#endif
	pstEvent = (SThreadEvent *) *phEvent;

	// send an event to release waiting tasks (ignore error)
	UCThreadEvent_SetEvent(*phEvent);

	// ignore error
	CloseHandle(pstEvent->hEvent);

	*phEvent = NULL;
	result = ERR_UEM_NOERROR;
_EXIT:
	SAFEMEMFREE(pstEvent);
	return result;
}


