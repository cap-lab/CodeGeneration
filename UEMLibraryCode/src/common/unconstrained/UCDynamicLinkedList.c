/*
 * UCDynamicLinkedList.c
 *
 *  Created on: 2017. 9. 20.
 *      Author: jej
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>


#include <UCBasic.h>
#include <UCAlloc.h>

#include <UCDynamicLinkedList.h>

#define CURRENT_NOT_SET (-1)

typedef struct _SNode {
    void *pData;
    struct _SNode *pNext;
} SNode;

typedef struct _SUCLinkedList {
	EUemModuleId enId;
    struct _SNode *pFirst;
    struct _SNode *pLast;
    struct _SNode *pCurrent;
    int nLinkSize;
    int nCurrent;
} SUCLinkedList;

uem_result UCDynamicLinkedList_Create(OUT HLinkedList *phLinkedList)
{
    SUCLinkedList* pstLinkedList = NULL;
    uem_result result = ERR_UEM_UNKNOWN;

#ifdef ARGUMENT_CHECK
    IFVARERRASSIGNGOTO(phLinkedList, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
    pstLinkedList = (SUCLinkedList *) UCAlloc_malloc(sizeof(SUCLinkedList));
    ERRMEMGOTO(pstLinkedList, result, _EXIT);

    pstLinkedList->enId = ID_UEM_DYNAMIC_LINKED_LIST;
    pstLinkedList->pFirst = NULL;
    pstLinkedList->pLast = NULL;
    pstLinkedList->pCurrent = NULL;
    pstLinkedList->nLinkSize = 0;
    pstLinkedList->nCurrent = CURRENT_NOT_SET;

    *phLinkedList = (HLinkedList) pstLinkedList;

    result = ERR_UEM_NOERROR;
_EXIT:
    return result;
}

uem_result UCDynamicLinkedList_Add(HLinkedList hLinkedList, IN ELinkedListOffset enOffset, IN int nIndex, IN void *pData)
{
    SNode *pstParentNode = NULL;
    SNode *pstNewNode = NULL;
    uem_result result = ERR_UEM_UNKNOWN;
    SUCLinkedList* pstLinkedList = NULL;
    int nOffset = 0;
#ifdef ARGUMENT_CHECK
    if (IS_VALID_HANDLE(hLinkedList, ID_UEM_DYNAMIC_LINKED_LIST) == FALSE) {
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
    }

    if (enOffset != LINKED_LIST_OFFSET_FIRST
            && enOffset != LINKED_LIST_OFFSET_LAST
            && enOffset != LINKED_LIST_OFFSET_CURRENT)
    {
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
    }
#endif
    pstLinkedList = (SUCLinkedList*) hLinkedList;

    if (enOffset == LINKED_LIST_OFFSET_FIRST) {
        nOffset = 0;
    } else if (enOffset == LINKED_LIST_OFFSET_CURRENT) {
        nOffset = pstLinkedList->nCurrent;
    } else { // enOffset == LINKED_LIST_OFFSET_LAST || LINKED_LIST_OFFSET_DEFAULT
        nOffset = pstLinkedList->nLinkSize;
    }

    nOffset = nOffset + nIndex;
#ifdef ARGUMENT_CHECK
    if (nOffset < 0 || nOffset > pstLinkedList->nLinkSize) {
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
    }
#endif
    pstNewNode = (SNode *) UCAlloc_malloc(sizeof(SNode));
    ERRMEMGOTO(pstNewNode, result, _EXIT);

    if (pstLinkedList->nLinkSize == 0) { /* If a linked list is empty */
        pstNewNode->pData = pData;
        pstNewNode->pNext = NULL;
        pstLinkedList->pFirst = pstNewNode;
        pstLinkedList->pLast = pstLinkedList->pFirst;
        pstLinkedList->pCurrent = pstLinkedList->pFirst;
        pstLinkedList->nLinkSize++;
        pstLinkedList->nCurrent = 0;
    } else {
        if (nOffset == 0) {
            /* add first */
            pstNewNode->pData = pData;
            pstNewNode->pNext = pstLinkedList->pFirst;

            pstLinkedList->pFirst = pstNewNode;
        } else {
            result = UCDynamicLinkedList_Seek(hLinkedList, LINKED_LIST_OFFSET_FIRST,
                    nOffset - 1);
            ERRIFGOTO(result, _EXIT);

            if (nOffset == pstLinkedList->nLinkSize) {
                /* add last */
                pstParentNode = pstLinkedList->pCurrent;
                pstParentNode->pNext = pstNewNode;

                pstNewNode->pData = pData;
                pstNewNode->pNext = NULL;

                pstLinkedList->pLast = pstNewNode;
            } else {
                /* add in middle */
                pstParentNode = pstLinkedList->pCurrent;

                pstNewNode->pData = pData;
                pstNewNode->pNext = pstParentNode->pNext;

                pstParentNode->pNext = pstNewNode;
            }
        }

        pstLinkedList->pCurrent = pstNewNode;
        pstLinkedList->nCurrent = nOffset;
        pstLinkedList->nLinkSize++;
    }
    result = ERR_UEM_NOERROR;
_EXIT:
    if (result != ERR_UEM_NOERROR) {
        SAFEMEMFREE(pstNewNode);
    }
    return result;
}

uem_result UCDynamicLinkedList_Seek(HLinkedList hLinkedList, IN ELinkedListOffset enOffset, IN int nIndex) {
    SUCLinkedList *pstLinkedList = NULL;
    uem_result result = ERR_UEM_UNKNOWN;
    int nOffset = 0;
    int nLoop = 0;
#ifdef ARGUMENT_CHECK
    if (IS_VALID_HANDLE(hLinkedList, ID_UEM_DYNAMIC_LINKED_LIST) == FALSE) {
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
    }
#endif
    pstLinkedList = (SUCLinkedList *) hLinkedList;

    if (enOffset == LINKED_LIST_OFFSET_FIRST) {
        nOffset = 0;
    } else if (enOffset == LINKED_LIST_OFFSET_CURRENT) {
        nOffset = pstLinkedList->nCurrent;
    } else { // enOffset == LINKED_LIST_OFFSET_LAST || LINKED_LIST_OFFSET_DEFAULT
        nOffset = pstLinkedList->nLinkSize;
    }

    nOffset = nOffset + nIndex;
#ifdef ARGUMENT_CHECK
    if (nOffset < 0 || nOffset > pstLinkedList->nLinkSize) {
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
    }

    if (pstLinkedList->nLinkSize == 0) {
        ERRASSIGNGOTO(result, ERR_UEM_NO_DATA, _EXIT);
    }
#endif
    if (nOffset >= pstLinkedList->nCurrent) {
        for (nLoop = 0; nLoop < nOffset - pstLinkedList->nCurrent; nLoop++) {
            pstLinkedList->pCurrent = pstLinkedList->pCurrent->pNext;
        }
    } else {
        pstLinkedList->pCurrent = pstLinkedList->pFirst;

        for (nLoop = 0; nLoop < nOffset; nLoop++) {
            pstLinkedList->pCurrent = pstLinkedList->pCurrent->pNext;
        }
    }

    pstLinkedList->nCurrent = nOffset;

    result = ERR_UEM_NOERROR;
_EXIT:
    return result;
}

uem_result UCDynamicLinkedList_Get(HLinkedList hLinkedList,
        IN ELinkedListOffset enOffset, IN int nIndex, OUT void **ppData)
{
    SUCLinkedList* pstLinkedList = NULL;
    uem_result result = ERR_UEM_UNKNOWN;
#ifdef ARGUMENT_CHECK
    if (IS_VALID_HANDLE(hLinkedList, ID_UEM_DYNAMIC_LINKED_LIST) == FALSE) {
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
    }

    IFVARERRASSIGNGOTO(ppData, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
    pstLinkedList = (SUCLinkedList *) hLinkedList;

    result = UCDynamicLinkedList_Seek(pstLinkedList, enOffset, nIndex);
    ERRIFGOTO(result, _EXIT);
#ifdef ARGUMENT_CHECK
    // nIndex is not located in the linked list
    if(pstLinkedList->pCurrent == NULL) {
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
    }
#endif
    *ppData = (void *) (pstLinkedList->pCurrent->pData);

    result = ERR_UEM_NOERROR;
_EXIT: return result;
}

// set the data to existing node
// before setting the new data, use UCDynamicLinkedList_Get to get the internal data and free the data first
uem_result UCDynamicLinkedList_Set(HLinkedList hLinkedList,
        IN ELinkedListOffset enOffset, IN int nIndex, IN void *pData)
{
    SUCLinkedList* pstLinkedList = NULL;
    uem_result result = ERR_UEM_UNKNOWN;
#ifdef ARGUMENT_CHECK
    if (IS_VALID_HANDLE(hLinkedList, ID_UEM_DYNAMIC_LINKED_LIST) == FALSE)
    {
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
    }

    IFVARERRASSIGNGOTO(pData, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
    pstLinkedList = (SUCLinkedList *) hLinkedList;

    result = UCDynamicLinkedList_Seek(pstLinkedList, enOffset, nIndex);
    ERRIFGOTO(result, _EXIT);
#ifdef ARGUMENT_CHECK
    // nIndex is not located in the linked list
    if(pstLinkedList->pCurrent == NULL)
    {
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
    }
#endif
    pstLinkedList->pCurrent->pData = pData;

    result = ERR_UEM_NOERROR;
_EXIT: return result;
}

uem_result UCDynamicLinkedList_GetLength(HLinkedList hLinkedList, OUT int *pnLength)
{
    SUCLinkedList* pstLinkedList = NULL;
    uem_result result = ERR_UEM_UNKNOWN;
#ifdef ARGUMENT_CHECK
    if (IS_VALID_HANDLE(hLinkedList, ID_UEM_DYNAMIC_LINKED_LIST) == FALSE)
    {
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
    }

    IFVARERRASSIGNGOTO(pnLength, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
    pstLinkedList = (SUCLinkedList *) hLinkedList;

    *pnLength = pstLinkedList->nLinkSize;

    result = ERR_UEM_NOERROR;
_EXIT: return result;
}



uem_result UCDynamicLinkedList_Remove(HLinkedList hLinkedList,
        IN ELinkedListOffset enOffset, IN int nIndex)
{
    SNode *pstCurNode = NULL;
    SNode *pstNextNode = NULL;
    SUCLinkedList* pstLinkedList = NULL;
    uem_result result = ERR_UEM_UNKNOWN;
    int nOffset = 0;
#ifdef ARGUMENT_CHECK
    if (IS_VALID_HANDLE(hLinkedList, ID_UEM_DYNAMIC_LINKED_LIST) == FALSE)
    {
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
    }
#endif
    pstLinkedList = (SUCLinkedList *) hLinkedList;

    if (enOffset == LINKED_LIST_OFFSET_FIRST)
    {
        nOffset = 0;
    }
    else if (enOffset == LINKED_LIST_OFFSET_CURRENT)
    {
        nOffset = pstLinkedList->nCurrent;
    }
    else // enOffset == LINKED_LIST_OFFSET_LAST || LINKED_LIST_OFFSET_DEFAULT
    {
        nOffset = pstLinkedList->nLinkSize;
    }

    nOffset = nOffset + nIndex;
#ifdef ARGUMENT_CHECK
    // To remove the last node, enOffset = LINKED_LIST_OFFSET_LAST, nIndex = -1
    if (nOffset < 0 || nOffset >= pstLinkedList->nLinkSize)
    {
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
    }

    if (pstLinkedList->nLinkSize == 0)
    {
        ERRASSIGNGOTO(result, ERR_UEM_NO_DATA, _EXIT);
    }
#endif
    if (nOffset == 0)
    { /* remove first */
        pstCurNode = pstLinkedList->pFirst;
        pstLinkedList->pFirst = pstCurNode->pNext;
        pstLinkedList->pCurrent = pstLinkedList->pFirst;
        pstLinkedList->nCurrent = nOffset;
    }
    else
    {
        result = UCDynamicLinkedList_Seek(hLinkedList, enOffset, nIndex - 1);
        ERRIFGOTO(result, _EXIT);

        if (nOffset == pstLinkedList->nLinkSize - 1)
        { /* remove last */
            pstCurNode = pstLinkedList->pCurrent->pNext;

            pstLinkedList->pLast = pstLinkedList->pCurrent;
            pstLinkedList->pLast->pNext = NULL;
            pstLinkedList->pCurrent = pstLinkedList->pLast;
            pstLinkedList->nCurrent = nOffset - 1;
        }
        else
        { /* remove in middle */
            pstCurNode = pstLinkedList->pCurrent->pNext;
            pstNextNode = pstCurNode->pNext;
            pstLinkedList->pCurrent->pNext = pstNextNode;
            pstLinkedList->nCurrent = nOffset - 1;
        }
    }
    SAFEMEMFREE(pstCurNode);
    pstLinkedList->nLinkSize--;

    result = ERR_UEM_NOERROR;
_EXIT:
    return result;
}


uem_result UCDynamicLinkedList_Traverse(HLinkedList hLinkedList, IN CbFnUCDynamicLinkedList fnCallback, IN void *pUserData)
{
    uem_result result = ERR_UEM_UNKNOWN;
    SUCLinkedList* pstLinkedList = NULL;
    SNode *pstNode = NULL;
    int nOffset = 0;
#ifdef ARGUMENT_CHECK
    if (IS_VALID_HANDLE(hLinkedList, ID_UEM_DYNAMIC_LINKED_LIST) == FALSE) {
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
    }

    IFVARERRASSIGNGOTO(fnCallback, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
    pstLinkedList = (SUCLinkedList*) hLinkedList;

    for(pstNode = pstLinkedList->pFirst; pstNode != NULL ; pstNode = pstNode->pNext)
    {
        result = fnCallback(nOffset, pstNode->pData, pUserData);
        if(result == ERR_UEM_USER_CANCELED) // if a user canceled the work, just return no error
        {
            result = ERR_UEM_NOERROR;
            break;
        }
        else if(result == ERR_UEM_FOUND_DATA)
        {
            UEMASSIGNGOTO(result, ERR_UEM_FOUND_DATA, _EXIT);
        }
        ERRIFGOTO(result, _EXIT);
        nOffset++;
    }

    result = ERR_UEM_NOERROR;
_EXIT:
    return result;
}



// UCDynamicLinkedList_Destroy does not free the internal void pointer which is inserted by UCDynamicLinkedList_Add function.
// To destroy all the elements in the linked list,
// API user should use UCDynamicLinkedList_Traverse to traverse all the linked list element and destroy the internal contents.
uem_result UCDynamicLinkedList_Destroy(IN OUT HLinkedList *phLinkedList)
{
    uem_result result = ERR_UEM_UNKNOWN;
    SUCLinkedList* pstLinkedList = NULL;
    int nLoop = 0;
    int nLinkedListSize = 0;
#ifdef ARGUMENT_CHECK
    IFVARERRASSIGNGOTO(phLinkedList, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

    if (IS_VALID_HANDLE(*phLinkedList, ID_UEM_DYNAMIC_LINKED_LIST) == FALSE)
    {
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
    }
#endif
    pstLinkedList = (SUCLinkedList*) *phLinkedList;

    nLinkedListSize = pstLinkedList->nLinkSize;
    for(nLoop = 0 ; nLoop < nLinkedListSize ; nLoop++)
        //for(pstNode = pstLinkedList->pFirst; pstNode != NULL ; pstNode = pstNode->pNext)
    {
        result = UCDynamicLinkedList_Remove((HLinkedList) pstLinkedList, LINKED_LIST_OFFSET_FIRST, 0);
        // ignore error;
    }

    SAFEMEMFREE(pstLinkedList);
    *phLinkedList = NULL;

    result = ERR_UEM_NOERROR;
_EXIT:
    return result;
}


