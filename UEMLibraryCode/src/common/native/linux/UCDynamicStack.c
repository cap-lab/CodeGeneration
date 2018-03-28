/*
 * UCDynamicStack.c
 *
 *  Created on: 2017. 10. 28.
 *      Author: jej
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif


#include <uem_common.h>

#include <UCBasic.h>

#include <UCDynamicLinkedList.h>
#include <UCDynamicStack.h>

typedef struct _SUCDynamicStack {
    EUemModuleId enId;
    HLinkedList hLinkedList;
} SUCDynamicStack;

typedef struct _SUCStackUserData {
    CbFnUCStack fnDestroyCallback;
    void *pUserData;
} SUCStackUserData;

uem_result removeStackData(IN int nOffset, IN void *pData, IN void *pUserData)
{
	uem_result result = ERR_UEM_UNKNOWN;
    SUCStackUserData *pstUserData = NULL;

    pstUserData = (SUCStackUserData *) pUserData;

    pstUserData->fnDestroyCallback(pData, pstUserData->pUserData);

    result = ERR_UEM_NOERROR;

    return result;
}


uem_result UCDynamicStack_Create(OUT HStack *phStack)
{
    uem_result result = ERR_UEM_UNKNOWN;
    SUCDynamicStack * pstStack = NULL;
#ifdef ARGUMENT_CHECK
    IFVARERRASSIGNGOTO(phStack, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
    pstStack = (SUCDynamicStack * ) UC_malloc(sizeof(SUCDynamicStack));
    ERRMEMGOTO(pstStack, result, _EXIT);

    pstStack->enId = ID_UEM_STACK;
    pstStack->hLinkedList = NULL;

    result = UCDynamicLinkedList_Create(&(pstStack->hLinkedList));
    ERRIFGOTO(result, _EXIT);

    *phStack = pstStack;

    result = ERR_UEM_NOERROR;
_EXIT:
    if(result != ERR_UEM_NOERROR && pstStack != NULL)
    {
        UCDynamicLinkedList_Destroy(&(pstStack->hLinkedList));
        SAFEMEMFREE(pstStack);
    }
    return result;
}


uem_result UCDynamicStack_Push(HStack hStack, IN void *pData)
{
    uem_result result = ERR_UEM_UNKNOWN;
    SUCDynamicStack * pstStack = NULL;
#ifdef ARGUMENT_CHECK
    if (IS_VALID_HANDLE(hStack, ID_UEM_STACK) == FALSE) {
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
    }
#endif
    pstStack = (SUCDynamicStack *) hStack;

    result = UCDynamicLinkedList_Add(pstStack->hLinkedList, LINKED_LIST_OFFSET_FIRST, 0, pData);
    ERRIFGOTO(result, _EXIT);

    result = UCDynamicLinkedList_Get(pstStack->hLinkedList, LINKED_LIST_OFFSET_FIRST, 0, &pData);
    ERRIFGOTO(result, _EXIT);

    result = ERR_UEM_NOERROR;
_EXIT:
    return result;
}


uem_result UCDynamicStack_Pop(HStack hStack, OUT void **ppData)
{
    uem_result result = ERR_UEM_UNKNOWN;
    SUCDynamicStack * pstStack = NULL;
    void *pData = NULL;
    int nLen = 0;
#ifdef ARGUMENT_CHECK
    if (IS_VALID_HANDLE(hStack, ID_UEM_STACK) == FALSE) {
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
    }

    IFVARERRASSIGNGOTO(ppData, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
    pstStack = (SUCDynamicStack *) hStack;

    result = UCDynamicLinkedList_GetLength(pstStack->hLinkedList, &nLen);
    ERRIFGOTO(result, _EXIT);

    if(nLen == 0)
    {
        ERRASSIGNGOTO(result, ERR_UEM_NO_DATA, _EXIT);
    }

    result = UCDynamicLinkedList_Get(pstStack->hLinkedList, LINKED_LIST_OFFSET_FIRST, 0, &pData);
    ERRIFGOTO(result, _EXIT);

    result = UCDynamicLinkedList_Remove(pstStack->hLinkedList, LINKED_LIST_OFFSET_FIRST, 0);
    ERRIFGOTO(result, _EXIT);

    *ppData = pData;

    result = ERR_UEM_NOERROR;
_EXIT:
    return result;
}


uem_result UCDynamicStack_Length(HStack hStack, OUT int *pnLength)
{
    uem_result result = ERR_UEM_UNKNOWN;
    SUCDynamicStack * pstStack = NULL;
    int nLen = 0;
#ifdef ARGUMENT_CHECK
    if (IS_VALID_HANDLE(hStack, ID_UEM_STACK) == FALSE) {
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
    }

    IFVARERRASSIGNGOTO(pnLength, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
    pstStack = (SUCDynamicStack *) hStack;

    result = UCDynamicLinkedList_GetLength(pstStack->hLinkedList, &nLen);
    ERRIFGOTO(result, _EXIT);

    *pnLength = nLen;

    result = ERR_UEM_NOERROR;
_EXIT:
    return result;
}

uem_result UCDynamicStack_Destroy(IN OUT HStack *phStack, IN CbFnUCStack fnDestroyCallback, IN void *pUserData)
{
    uem_result result = ERR_UEM_UNKNOWN;
    SUCDynamicStack * pstStack = NULL;
    SUCStackUserData stCbData;
#ifdef ARGUMENT_CHECK
    IFVARERRASSIGNGOTO(phStack, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

    if (IS_VALID_HANDLE(*phStack, ID_UEM_STACK) == FALSE) {
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
    }
#endif
    pstStack = (SUCDynamicStack *) *phStack;

    // ignore errors in destroy function
    if(fnDestroyCallback != NULL)
    {
        stCbData.fnDestroyCallback = fnDestroyCallback;
        stCbData.pUserData = pUserData;
        UCDynamicLinkedList_Traverse(pstStack->hLinkedList, removeStackData, &stCbData);
    }

    UCDynamicLinkedList_Destroy(&(pstStack->hLinkedList));
    SAFEMEMFREE(pstStack);

    *phStack = NULL;

    result = ERR_UEM_NOERROR;
_EXIT:
    return result;
}

