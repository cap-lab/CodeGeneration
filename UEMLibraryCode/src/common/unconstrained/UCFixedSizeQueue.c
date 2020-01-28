/*
 * UCFixedSizeQueue.c
 *
 *  Created on: 2018. 10. 11.
 *      Author: chjej202
 */


#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <UCBasic.h>
#include <UCAlloc.h>

#include <UCThreadMutex.h>
#include <UCThreadEvent.h>

#include <UCFixedSizeQueue.h>

typedef struct _SFixedSizeQueue {
	EUemModuleId enId;
	int nElementSize;
	int nMaxElementNum;
	void *pQueueData;
	HThreadEvent hReadEvent;
	HThreadEvent hWriteEvent;
	HThreadMutex hMutex;
	int nFrontIndex;
	int nRearIndex;
	int nQueueItemNum;
	uem_bool bDestroy;
} SFixedSizeQueue;


#define QUEUE_TIMEOUT (3000)

uem_result UCFixedSizeQueue_Create(int nElementSize, int nElementNum, OUT HFixedSizeQueue *phQueue)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SFixedSizeQueue *pstQueue = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(phQueue, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	if(nElementNum <= 0 || nElementSize <= 0)
	{
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
#endif
	pstQueue = UCAlloc_malloc(sizeof(struct _SFixedSizeQueue));
	ERRMEMGOTO(pstQueue, result, _EXIT);

	pstQueue->hReadEvent = NULL;
	pstQueue->hWriteEvent = NULL;
	pstQueue->hMutex = NULL;
	pstQueue->nMaxElementNum = nElementNum;
	pstQueue->nElementSize = nElementSize;
	pstQueue->nFrontIndex = 0;
	pstQueue->nQueueItemNum = 0;
	pstQueue->nRearIndex = 0;
	pstQueue->pQueueData = NULL;
	pstQueue->bDestroy = FALSE;
	pstQueue->enId = ID_UEM_FIXED_SIZE_QUEUE;

	result = UCThreadEvent_Create(&(pstQueue->hReadEvent));
	ERRIFGOTO(result, _EXIT);

	result = UCThreadEvent_Create(&(pstQueue->hWriteEvent));
	ERRIFGOTO(result, _EXIT);

	result = UCThreadMutex_Create(&(pstQueue->hMutex));
	ERRIFGOTO(result, _EXIT);

	pstQueue->pQueueData = UCAlloc_malloc(nElementNum * nElementSize);
	ERRMEMGOTO(pstQueue, result, _EXIT);

	*phQueue = pstQueue;

	result = ERR_UEM_NOERROR;
_EXIT:
	if(result != ERR_UEM_NOERROR && pstQueue != NULL)
	{
		UCFixedSizeQueue_Destroy(&pstQueue);
	}
	return result;
}


uem_result UCFixedSizeQueue_SetExit(HFixedSizeQueue hQueue)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SFixedSizeQueue *pstQueue = NULL;
#ifdef ARGUMENT_CHECK
	if(IS_VALID_HANDLE(hQueue, ID_UEM_FIXED_SIZE_QUEUE) == FALSE)
	{
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}
#endif
	pstQueue = (struct _SFixedSizeQueue *) hQueue;

	pstQueue->bDestroy = TRUE;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UCFixedSizeQueue_Destroy(IN OUT HFixedSizeQueue *phQueue)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SFixedSizeQueue *pstQueue = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(phQueue, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	if(IS_VALID_HANDLE(*phQueue, ID_UEM_FIXED_SIZE_QUEUE) == FALSE)
	{
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}
#endif
	pstQueue = *phQueue;
	pstQueue->bDestroy = TRUE;

	SAFEMEMFREE(pstQueue->pQueueData);

	if(pstQueue->hMutex != NULL)
	{
		UCThreadMutex_Destroy(&(pstQueue->hMutex));
	}

	if(pstQueue->hWriteEvent != NULL)
	{
		UCThreadEvent_Destroy(&(pstQueue->hWriteEvent));
	}

	if(pstQueue->hReadEvent != NULL)
	{
		UCThreadEvent_Destroy(&(pstQueue->hReadEvent));
	}

	SAFEMEMFREE(pstQueue);

	*phQueue = NULL;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UCFixedSizeQueue_PutItem(HFixedSizeQueue hQueue, void *pData, int nElementSize)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SFixedSizeQueue *pstQueue = NULL;
	void *pItemLocation = NULL;
#ifdef ARGUMENT_CHECK
	if(IS_VALID_HANDLE(hQueue, ID_UEM_FIXED_SIZE_QUEUE) == FALSE)
	{
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}
	IFVARERRASSIGNGOTO(pData, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	pstQueue = (struct _SFixedSizeQueue *) hQueue;
#ifdef ARGUMENT_CHECK
	if(nElementSize != pstQueue->nElementSize)
	{
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
#endif

	result = UCThreadMutex_Lock(pstQueue->hMutex);
	ERRIFGOTO(result, _EXIT);

	while(pstQueue->nQueueItemNum == pstQueue->nMaxElementNum && pstQueue->bDestroy == FALSE)
	{
		result = UCThreadMutex_Unlock(pstQueue->hMutex);
		ERRIFGOTO(result, _EXIT);

		result = UCThreadEvent_WaitTimeEvent(pstQueue->hWriteEvent, QUEUE_TIMEOUT);
		if(result == ERR_UEM_TIME_EXPIRED)
		{
			UEMASSIGNGOTO(result, ERR_UEM_TIME_EXPIRED, _EXIT);
		}
		ERRIFGOTO(result, _EXIT);

		result = UCThreadMutex_Lock(pstQueue->hMutex);
		ERRIFGOTO(result, _EXIT);
	}

	if (pstQueue->bDestroy == TRUE) {
		ERRASSIGNGOTO(result, ERR_UEM_SUSPEND, _EXIT_LOCK);
	}

	pItemLocation = (char *) pstQueue->pQueueData + pstQueue->nRearIndex * nElementSize;
	UC_memcpy(pItemLocation, pData, nElementSize);

	pstQueue->nRearIndex = (pstQueue->nRearIndex + 1) % pstQueue->nMaxElementNum;
	pstQueue->nQueueItemNum++;

	result = UCThreadEvent_SetEvent(pstQueue->hReadEvent);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
	UCThreadMutex_Unlock(pstQueue->hMutex);
_EXIT:
	return result;
}


uem_result UCFixedSizeQueue_GetItem(HFixedSizeQueue hQueue, void *pData, int *pnElementSize)
{
	uem_result result = ERR_UEM_UNKNOWN;
	struct _SFixedSizeQueue *pstQueue = NULL;
	void *pItemLocation = NULL;
#ifdef ARGUMENT_CHECK
	if(IS_VALID_HANDLE(hQueue, ID_UEM_FIXED_SIZE_QUEUE) == FALSE)
	{
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
	}
	IFVARERRASSIGNGOTO(pData, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	IFVARERRASSIGNGOTO(pnElementSize, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	pstQueue = (struct _SFixedSizeQueue *) hQueue;

	result = UCThreadMutex_Lock(pstQueue->hMutex);
	ERRIFGOTO(result, _EXIT);

	while(pstQueue->nQueueItemNum <= 0 && pstQueue->bDestroy == FALSE)
	{
		result = UCThreadMutex_Unlock(pstQueue->hMutex);
		ERRIFGOTO(result, _EXIT);

		result = UCThreadEvent_WaitTimeEvent(pstQueue->hReadEvent, QUEUE_TIMEOUT);
		if(result == ERR_UEM_TIME_EXPIRED)
		{
			UEMASSIGNGOTO(result, ERR_UEM_TIME_EXPIRED, _EXIT);
		}
		ERRIFGOTO(result, _EXIT);

		result = UCThreadMutex_Lock(pstQueue->hMutex);
		ERRIFGOTO(result, _EXIT);
	}

	if (pstQueue->bDestroy == TRUE) {
		ERRASSIGNGOTO(result, ERR_UEM_SUSPEND, _EXIT_LOCK);
	}

	pItemLocation = (char *) pstQueue->pQueueData + pstQueue->nFrontIndex * pstQueue->nElementSize;

	UC_memcpy(pData, pItemLocation, pstQueue->nElementSize);

	*pnElementSize = pstQueue->nElementSize;

	pstQueue->nFrontIndex = (pstQueue->nFrontIndex + 1) % pstQueue->nMaxElementNum;
	pstQueue->nQueueItemNum--;

	result = UCThreadEvent_SetEvent(pstQueue->hWriteEvent);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT_LOCK:
	UCThreadMutex_Unlock(pstQueue->hMutex);
_EXIT:
	return result;
}


