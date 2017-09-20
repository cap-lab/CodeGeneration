/*
 * UCDynamicLinkedList.h
 *
 *  Created on: 2017. 9. 20.
 *      Author: jej
 */

#ifndef SRC_COMMON_INCLUDE_UCDYNAMICLINKEDLIST_H_
#define SRC_COMMON_INCLUDE_UCDYNAMICLINKEDLIST_H_

#include "uem_common.h";

#ifdef __cplusplus
extern "C"
{
#endif

typedef void *HLinkedList;

typedef enum _ELinkedListOffset {
    LINKED_LIST_OFFSET_FIRST,
    LINKED_LIST_OFFSET_LAST,
    LINKED_LIST_OFFSET_CURRENT,
    LINKED_LIST_OFFSET_DEFAULT = LINKED_LIST_OFFSET_LAST,
} ELinkedListOffset;

typedef uem_result (*CbFnUCDynamicLinkedList)(IN int nOffset, IN void *pData, IN void *pUserData);
typedef uem_result (*CbFnUCDynamicLinkedListDup)(IN int nOffset, IN void *pDataSrc, IN void *pUserData, OUT void **ppDataDst);


uem_result UCDynamicLinkedList_Create(int nSize);
uem_result UCDynamicLinkedList_Add(HLinkedList hLinkedList, IN ELinkedListOffset enOffset, IN int nIndex, IN void *pData);
uem_result UCDynamicLinkedList_Seek(HLinkedList hLinkedList, IN ELinkedListOffset enOffset, IN int nIndex);
uem_result UCDynamicLinkedList_Get(HLinkedList hLinkedList, IN ELinkedListOffset enOffset, IN int nIndex, OUT void **ppData);
uem_result UCDynamicLinkedList_Set(HLinkedList hLinkedList, IN ELinkedListOffset enOffset, IN int nIndex, IN void *pData);
uem_result UCDynamicLinkedList_GetLength(HLinkedList hLinkedList, OUT int *pnLength);
uem_result UCDynamicLinkedList_Remove(HLinkedList hLinkedList, IN ELinkedListOffset enOffset, IN int nIndex);
uem_result UCDynamicLinkedList_Traverse(HLinkedList hLinkedList, IN CbFnUCDynamicLinkedList fnCallback, IN void *pUserData);
uem_result UCDynamicLinkedList_Destroy(IN OUT HLinkedList *phLinkedList);

#ifdef __cplusplus
}
#endif


#endif /* SRC_COMMON_INCLUDE_UCDYNAMICLINKEDLIST_H_ */
