/*
 * UCThreadMutex.c
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

#include <UCThreadMutex.h>

typedef struct _SThreadMutex {
	EUemModuleId enId;
	uem_bool bInMutex;
	uem_bool bIsStatic;
	CRITICAL_SECTION hMutex;
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

	InitializeCriticalSection(&(pstMutex->hMutex));

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

	EnterCriticalSection(&(pstMutex->hMutex));

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

	LeaveCriticalSection(&(pstMutex->hMutex));

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
		LeaveCriticalSection(&(pstMutex->hMutex));
	}

	DeleteCriticalSection(&(pstMutex->hMutex));

	SAFEMEMFREE(pstMutex);

	*phMutex = NULL;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


