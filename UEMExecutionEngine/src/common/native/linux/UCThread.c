/*
 * UCThread.c
 *
 *  Created on: 2017. 8. 10.
 *      Author: jej
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif


#include <pthread.h>

#include <UCBasic.h>

#include <UCThread.h>

typedef struct _SThread {
	EUemModuleId enId;
	pthread_t hNativeThread;
} SUCThread;

uem_result UCThread_Create(FnNativeThread fnThreadRoutine, void *pUserData, HThread *phThread)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SUCThread *pstThread = NULL;
	pthread_attr_t threadAttr;

	if(pthread_attr_init(&threadAttr) != 0) {
		ERRASSIGNGOTO(result, ERR_UEM_INTERNAL_FAIL, _EXIT);
	}

	IFVARERRASSIGNGOTO(phThread, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	IFVARERRASSIGNGOTO(fnThreadRoutine, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT)

	pstThread = (SUCThread *) UC_malloc(sizeof(SUCThread));
	ERRMEMGOTO(pstThread, result, _EXIT);

	pstThread->enId = ID_UEM_THREAD;

	if(pthread_create(&(pstThread->hNativeThread), &threadAttr, fnThreadRoutine, (void *)pUserData) != 0) {
		ERRASSIGNGOTO(result, ERR_UEM_INTERNAL_FAIL, _EXIT);
	}

	*phThread = (HThread) pstThread;

	result = ERR_UEM_NOERROR;
_EXIT:
	if(result != ERR_UEM_NOERROR) {
		SAFEMEMFREE(pstThread);
	}
	pthread_attr_destroy( &threadAttr );

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

	if(pthread_join(pstThread->hNativeThread, NULL) != 0) {
		// free the memory even though pthread_join is failed.
		ERRASSIGNGOTO(result, ERR_UEM_INTERNAL_FAIL, _EXIT);
	}

	*phThread = NULL;


	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

