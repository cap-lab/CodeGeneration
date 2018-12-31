/*
 * UKModule.h
 *
 *  Created on: 2018. 6. 16.
 *      Author: jej
 */

#ifndef SRC_KERNEL_INCLUDE_UKMODULE_H_
#define SRC_KERNEL_INCLUDE_UKMODULE_H_

#include <uem_common.h>

#include <uem_data.h>

#ifdef __cplusplus
extern "C"
{
#endif

uem_result UKModule_Initialize();
uem_result UKModule_Finalize();

extern int g_nModuleNum;
extern SAddOnModule g_stModules[];

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_INCLUDE_UKMODULE_H_ */
