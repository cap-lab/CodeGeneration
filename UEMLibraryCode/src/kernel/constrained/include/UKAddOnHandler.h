/*
 * UKAddOnHandler.h
 *
 *  Created on: 2018. 10. 26.
 *      Author: jej
 */

#ifndef SRC_KERNEL_CONSTRAINED_INCLUDE_UKADDONHANDLER_H_
#define SRC_KERNEL_CONSTRAINED_INCLUDE_UKADDONHANDLER_H_

#include <uem_common.h>

#ifdef __cplusplus
extern "C"
{
#endif


typedef uem_result (*FuncAddOnFunction)();


typedef struct _SAddOnFunction {
	FuncAddOnFunction fnInit;
	FuncAddOnFunction fnRun;
	FuncAddOnFunction fnWrapup;
} SAddOnFunction;


extern SAddOnFunction g_astAddOns[];
extern int g_nAddOnNum;

/**
 * @brief Initialize addon module.
 *
 * This function executes addon module initialization function. \n
 * Addon module is used for extra jobs needed for task execution.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error.
 */
uem_result UKAddOnHandler_Init();

/**
 * @brief Perform addon module execution.
 *
 * This function executes addon module run function.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error.
 */
uem_result UKAddOnHandler_Run();

/**
 * @brief Perform addon module wrapup.
 *
 * This function executes addon module wrapup function.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error.
 */
uem_result UKAddOnHandler_Fini();


#ifdef __cplusplus
}
#endif


#endif /* SRC_KERNEL_CONSTRAINED_INCLUDE_UKADDONHANDLER_H_ */
