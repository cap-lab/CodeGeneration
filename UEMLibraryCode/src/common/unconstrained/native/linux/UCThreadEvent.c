/*
 * UCThreadEvent.c
 *
 *  Created on: 2017. 8. 5.
 *      Author: jej
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#ifdef HAVE_PTHREAD_TIME_H
#include <pthread_time.h>
#endif

#include <errno.h>

#include <pthread.h>

#include <UCBasic.h>
#include <UCAlloc.h>
#include <UCThreadEvent.h>

typedef struct _SThreadEvent {
	EUemModuleId enId;
	uem_bool bIsSet;
	uem_bool bIsStatic;
	pthread_mutex_t hMutex;
	pthread_cond_t hCond;
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
	pstEvent->bIsSet = FALSE;

	if (pthread_mutex_init(&(pstEvent->hMutex), NULL) != 0) {
		ERRASSIGNGOTO(result, ERR_UEM_INTERNAL_FAIL, _EXIT);
	}

	if (pthread_cond_init(&(pstEvent->hCond), NULL) != 0) {
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
	uem_bool bMutexFailed = FALSE;
	int nErrorNum = 0;
#ifdef ARGUMENT_CHECK
	if (IS_VALID_HANDLE(hEvent, ID_UEM_THREAD_EVENT) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}
#endif
	pstEvent = (SThreadEvent *) hEvent;

	if (pthread_mutex_lock(&(pstEvent->hMutex)) != 0) {
		bMutexFailed = TRUE;
	}

	pstEvent->bIsSet = TRUE; // event is set
	// send a signal
	nErrorNum = pthread_cond_broadcast(&(pstEvent->hCond));

	if (bMutexFailed == FALSE && pthread_mutex_unlock(&(pstEvent->hMutex)) != 0) {
		// ignore error
	}

	if(nErrorNum != 0)
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
#ifdef ARGUMENT_CHECK
	if (IS_VALID_HANDLE(hEvent, ID_UEM_THREAD_EVENT) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}
#endif
	pstEvent = (SThreadEvent *) hEvent;

	if (pthread_mutex_lock(&(pstEvent->hMutex)) != 0) {
		ERRASSIGNGOTO(result, ERR_UEM_MUTEX_ERROR, _EXIT);
	}

	while(pstEvent->bIsSet == FALSE) {
		if (pthread_cond_wait(&(pstEvent->hCond), &(pstEvent->hMutex)) != 0) {
			ERRASSIGNGOTO(result, ERR_UEM_MUTEX_ERROR, _EXIT_LOCK);
		}
	}

	pstEvent->bIsSet = FALSE;

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
	if (pthread_mutex_unlock(&(pstEvent->hMutex)) != 0) {
		ERRASSIGNGOTO(result, ERR_UEM_MUTEX_ERROR, _EXIT);
	}
_EXIT:
	return result;
}

uem_result UCThreadEvent_WaitTimeEvent(HThreadEvent hEvent, long long llSleepTimeMs)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SThreadEvent *pstEvent = NULL;
	struct timespec stTimeOut;
	struct timespec stNow;
	int ret = 0;
#ifdef ARGUMENT_CHECK
	if (IS_VALID_HANDLE(hEvent, ID_UEM_THREAD_EVENT) == FALSE) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}

	if(llSleepTimeMs <= 0) {
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
#endif
	pstEvent = (SThreadEvent *) hEvent;

	//gettimeofday(&sNow, NULL);
	clock_gettime(CLOCK_REALTIME, &stNow);

	stTimeOut.tv_sec = stNow.tv_sec + llSleepTimeMs/1000;
	stTimeOut.tv_nsec = stNow.tv_nsec + (llSleepTimeMs % 1000)*1000000;


	if (pthread_mutex_lock(&(pstEvent->hMutex)) != 0) {
		ERRASSIGNGOTO(result, ERR_UEM_MUTEX_ERROR, _EXIT);
	}

	while(pstEvent->bIsSet == FALSE) {
		ret = pthread_cond_timedwait(&(pstEvent->hCond), &(pstEvent->hMutex), &stTimeOut);

		if(ret == 0)
		{
		    // do nothing
		}
		else if(ret == ETIMEDOUT)
		{
		    UEMASSIGNGOTO(result, ERR_UEM_TIME_EXPIRED, _EXIT_LOCK);
		}
		else
		{
			ERRASSIGNGOTO(result, ERR_UEM_MUTEX_ERROR, _EXIT_LOCK);
		}
	}

	pstEvent->bIsSet = FALSE;

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
	if (pthread_mutex_unlock(&(pstEvent->hMutex)) != 0) {
		ERRASSIGNGOTO(result, ERR_UEM_MUTEX_ERROR, _EXIT);
	}
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
	if (pthread_cond_destroy(&(pstEvent->hCond)) != 0) {
		ERRASSIGNGOTO(result, ERR_UEM_MUTEX_ERROR, _EXIT);
	}

	// ignore error
	if (pthread_mutex_destroy(&(pstEvent->hMutex)) != 0) {
		ERRASSIGNGOTO(result, ERR_UEM_MUTEX_ERROR, _EXIT);
	}

	*phEvent = NULL;
	result = ERR_UEM_NOERROR;
_EXIT:
	SAFEMEMFREE(pstEvent);
	return result;
}


