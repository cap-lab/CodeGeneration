/*
 * uem_callbacks.h
 *
 *  Created on: 2018. 8. 30.
 *      Author: chjej202
 */

#ifndef SRC_KERNEL_INCLUDE_UEM_CALLBACKS_H_
#define SRC_KERNEL_INCLUDE_UEM_CALLBACKS_H_

#ifdef __cplusplus
extern "C"
{
#endif

// Task functions
typedef void (*FnUemTaskInit)(int nTaskId);
typedef void (*FnUemTaskGo)(int nTaskId);
typedef void (*FnUemTaskWrapup)();

// Library functions
typedef void (*FnUemLibraryInit)();
typedef void (*FnUemLibraryWrapup)();

#ifdef __cplusplus
}
#endif


#endif /* SRC_KERNEL_INCLUDE_UEM_CALLBACKS_H_ */
