/*
 * UCThreadMutex.c
 *
 *  Created on: 2017. 8. 5.
 *      Author: jej
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif


#include <pthread.h>

#include <UCBasic.h>
#include <UCAlloc.h>

#include <UCThreadMutex.h>

typedef struct _SThreadMutex {
	EUemModuleId enId;
	uem_bool bInMutex;
	uem_bool bIsStatic;
    pthread_mutex_t hMutex;
} SThreadMutex;

uem_result UCThreadMutex_Create(HThreadMutex *phMutex)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SThreadMutex *pstMutex = NULL;

#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(phMutex, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif

	pstMutex = (SThreadMutex *) UCAlloc_malloc(sizeof(SThreadMutex));
	ERRMEMGOTO(pstMutex, result, _EXIT);

	pstMutex->enId = ID_UEM_THREAD_MUTEX;
	pstMutex->bIsStatic = FALSE;
	pstMutex->bInMutex = FALSE;

	if(pthread_mutex_init(&(pstMutex->hMutex), NULL) != 0) {
		ERRASSIGNGOTO(result, ERR_UEM_MUTEX_ERROR, _EXIT);
	}

	*phMutex = pstMutex;

	result = ERR_UEM_NOERROR;
_EXIT:
	if(result != ERR_UEM_NOERROR)
	{
		SAFEMEMFREE(pstMutex);
	}
	return result;
}


uem_result UCThreadMutex_Lock(HThreadMutex hMutex)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SThreadMutex *pstMutex = NULL;

#ifdef ARGUMENT_CHECK
	if(IS_VALID_HANDLE(hMutex, ID_UEM_THREAD_MUTEX) == FALSE)
	{
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}
#endif

	pstMutex = (SThreadMutex *) hMutex;

	if(pthread_mutex_lock(&(pstMutex->hMutex)) != 0)
	{
		ERRASSIGNGOTO(result, ERR_UEM_MUTEX_ERROR, _EXIT);
	}

	pstMutex->bInMutex = TRUE;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UCThreadMutex_Unlock(HThreadMutex hMutex)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SThreadMutex *pstMutex = NULL;

#ifdef ARGUMENT_CHECK
	if(IS_VALID_HANDLE(hMutex, ID_UEM_THREAD_MUTEX) == FALSE)
	{
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}
#endif

	pstMutex = (SThreadMutex *) hMutex;

	if(pstMutex->bInMutex == FALSE)
	{
		ERRASSIGNGOTO(result, ERR_UEM_MUTEX_ERROR, _EXIT);
	}

	// Change the lock state before unlock
	pstMutex->bInMutex = FALSE;

	if(pthread_mutex_unlock(&(pstMutex->hMutex)) != 0)
	{
		pstMutex->bInMutex = TRUE; // restore the lock state because of unlock fail
		ERRASSIGNGOTO(result, ERR_UEM_MUTEX_ERROR, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UCThreadMutex_Destroy(HThreadMutex *phMutex)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SThreadMutex *pstMutex = NULL;

#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(phMutex, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	if(IS_VALID_HANDLE(*phMutex, ID_UEM_THREAD_MUTEX) == FALSE)
	{
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}
#endif
	pstMutex = (SThreadMutex *) *phMutex;

#ifdef ARGUMENT_CHECK
	if(pstMutex->bIsStatic == TRUE)
	{
		ERRASSIGNGOTO(result, ERR_UEM_STATIC_HANDLE, _EXIT);
	}
#endif
	if(pstMutex->bInMutex == TRUE) // if the mutex is locked, unlock the mutex
	{
		// ignore error
		pthread_mutex_unlock(&(pstMutex->hMutex));
	}

	if(pthread_mutex_destroy(&(pstMutex->hMutex)) != 0)
	{
		ERRASSIGNGOTO(result, ERR_UEM_MUTEX_ERROR, _EXIT);
	}

	SAFEMEMFREE(pstMutex);

	*phMutex = NULL;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


