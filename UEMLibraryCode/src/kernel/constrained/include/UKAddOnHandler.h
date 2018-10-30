/*
 * UKAddOnHandler.h
 *
 *  Created on: 2018. 10. 26.
 *      Author: jej
 */

#ifndef SRC_KERNEL_CONSTRAINED_INCLUDE_UKADDONHANDLER_H_
#define SRC_KERNEL_CONSTRAINED_INCLUDE_UKADDONHANDLER_H_

#include <uem_common.h>

#include <UCSerial.h>

#include <uem_protocol_data.h>


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


uem_result UKAddOnHandler_Init();
uem_result UKAddOnHandler_Run();
uem_result UKAddOnHandler_Fini();


#ifdef __cplusplus
}
#endif


#endif /* SRC_KERNEL_CONSTRAINED_INCLUDE_UKADDONHANDLER_H_ */
