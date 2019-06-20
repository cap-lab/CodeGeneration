/*
 * UCDynamicLinkedList.h
 *
 *  Created on: 2017. 9. 20.
 *      Author: jej
 */

#ifndef SRC_COMMON_INCLUDE_UCDYNAMICLINKEDLIST_H_
#define SRC_COMMON_INCLUDE_UCDYNAMICLINKEDLIST_H_

#include "uem_common.h"

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct _SUCLinkedList *HLinkedList;

typedef enum _ELinkedListOffset {
    LINKED_LIST_OFFSET_FIRST,
    LINKED_LIST_OFFSET_LAST,
    LINKED_LIST_OFFSET_CURRENT,
    LINKED_LIST_OFFSET_DEFAULT = LINKED_LIST_OFFSET_LAST,
} ELinkedListOffset;

typedef uem_result (*CbFnUCDynamicLinkedList)(IN int nOffset, IN void *pData, IN void *pUserData);
typedef uem_result (*CbFnUCDynamicLinkedListDup)(IN int nOffset, IN void *pDataSrc, IN void *pUserData, OUT void **ppDataDst);

/**
 * @brief Create a dynamic linked list.
 *
 * This function creates an unbounded linked list. \n
 * Internal memory allocation is happened depending on the items to be added or removed.
 *
 * @warning UCDynamicLinkedList does not provide callback function for destroying internal data. \n
 * To destroy internal data, use @ref UCDynamicLinkedList_Traverse or Use @ref UCDynamicLinkedList_Get with for-loop. \n
 * Then, call @ref UCDynamicLinkedList_Destroy to free the whole data structure.
 *
 * @param[out] phLinkedList a handle of the linked list.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_OUT_OF_MEMORY.
 */
uem_result UCDynamicLinkedList_Create(OUT HLinkedList *phLinkedList);

/**
 * @brief Add an item into the linked list
 *
 * This function adds an item into the linked list. \n
 * By using @a enOffset and @a nIndex, an item can be inserted any specific location. \n
 * Check @ref ELinkedListOffset for base offset setting. \n
 *
 * To insert an item at head:
 *
 *     @code{.c}
 *     UCDynamicLinkedList_Add(hLinkedList, LINKED_LIST_OFFSET_FIRST, 0, pData);
 *     @endcode
 *
 *
 * To insert an item at tail:
 *
 *     @code{.c}
 *     UCDynamicLinkedList_Add(hLinkedList, LINKED_LIST_OFFSET_LAST, 0, pData);
 *     @endcode
 *
 * @param hLinkedList a linked list handle.
 * @param enOffset a base offset. @ref ELinkedListOffset.
 * @param nIndex an index from the base offset.
 * @param pData an item to add into the linked list.
 *
 * @return  @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *          Errors to be returned - @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_OUT_OF_MEMORY.
 */
uem_result UCDynamicLinkedList_Add(HLinkedList hLinkedList, IN ELinkedListOffset enOffset, IN int nIndex, IN void *pData);

/**
 * @brief Move the linked list offset.
 *
 * This function moves the offset of @ref LINKED_LIST_OFFSET_CURRENT.
 *
 * @param hLinkedList a linked list handle.
 * @param enOffset a base offset. @ref ELinkedListOffset.
 * @param nIndex an index from the base offset.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_NO_DATA. \n
 *         @ref ERR_UEM_NO_DATA can be occurred if there is no item in the linked list.
 */
uem_result UCDynamicLinkedList_Seek(HLinkedList hLinkedList, IN ELinkedListOffset enOffset, IN int nIndex);

/**
 * @brief Get a specific item in the linked list.
 *
 * This function retrieves data in the linked list. \n
 * @a enOffset and @a nIndex are used to specify the item location in the linked list.
 *
 * @param hLinkedList a linked list handle.
 * @param enOffset a base offset. @ref ELinkedListOffset.
 * @param nIndex an index from the base offset.
 * @param[out] ppData a retrieved item.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_NO_DATA. \n
 *         @ref ERR_UEM_NO_DATA can be occurred if there is no item in the linked list.
 */
uem_result UCDynamicLinkedList_Get(HLinkedList hLinkedList, IN ELinkedListOffset enOffset, IN int nIndex, OUT void **ppData);

/**
 * @brief Change an item in the linked list.
 *
 * This function changes an item of specific location of the linked list.\n
 * @a enOffset and @a nIndex are used to specify the location in the linked list.
 *
 * @param hLinkedList a linked list handle.
 * @param enOffset a base offset. @ref ELinkedListOffset.
 * @param nIndex an index from the base offset.
 * @param pData an item to set.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_NO_DATA. \n
 *         @ref ERR_UEM_NO_DATA can be occurred if there is no item in the linked list.
 */
uem_result UCDynamicLinkedList_Set(HLinkedList hLinkedList, IN ELinkedListOffset enOffset, IN int nIndex, IN void *pData);

/**
 * @brief Get size of the linked list.
 *
 * This function retrieves the number of items in the linked list.
 *
 * @param hLinkedList a linked list handle.
 * @param[out] pnLength the size of linked list.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UCDynamicLinkedList_GetLength(HLinkedList hLinkedList, OUT int *pnLength);

/**
 * @brief Remove the specific item in the linked list.
 *
 * This function removes the item specified by @a enOffset and @a nIndex.
 *
 * @warning This function does not remove internal data structure, \n
 * so internal data removal is needed before calling this function. \n
 * Use @ref UCDynamicLinkedList_Get to get and free the data. Then, call this function.
 *
 * @param hLinkedList a linked list handle.
 * @param enOffset a base offset. @ref ELinkedListOffset.
 * @param nIndex an index from the base offset.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_NO_DATA.
 *         @ref ERR_UEM_NO_DATA can be occurred if there is no item in the linked list.
 */
uem_result UCDynamicLinkedList_Remove(HLinkedList hLinkedList, IN ELinkedListOffset enOffset, IN int nIndex);

/**
 * @brief Traverse all items in the linked list.
 *
 * This function traverses all items in the linked list. For handling each item, @a fnCallback callback function is used. \n
 * By returning @ref ERR_UEM_USER_CANCELED in the callback function, user can stop traversing the linked list without an error. \n
 * By returning @ref ERR_UEM_FOUND_DATA in the callback function, the function will forward the @ref ERR_UEM_FOUND_DATA result as a return value.
 *
 * @param hLinkedList a linked list handle.
 * @param fnCallback  callback function for accessing each linked list item.
 * @param pUserData pointer to user data passing to the callback function.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INVALID_PARAM, or results from @a fnCallback function.
 */
uem_result UCDynamicLinkedList_Traverse(HLinkedList hLinkedList, IN CbFnUCDynamicLinkedList fnCallback, IN void *pUserData);

/**
 * @brief Destroy the linked list.
 *
 * This function destroys the linked list.
 *
 * @warning please read warning shown in @ref UCDynamicLinkedList_Create.
 *
 * @param[in,out] phLinkedList a linked list handle to be destroyed.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INVALID_PARAM.
 *
 */
uem_result UCDynamicLinkedList_Destroy(IN OUT HLinkedList *phLinkedList);

#ifdef __cplusplus
}
#endif


#endif /* SRC_COMMON_INCLUDE_UCDYNAMICLINKEDLIST_H_ */
