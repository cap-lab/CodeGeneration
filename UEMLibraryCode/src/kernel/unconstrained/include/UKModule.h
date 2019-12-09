/*
 * UKModule.h
 *
 *  Created on: 2018. 6. 16.
 *      Author: jej
 */

#ifndef SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKMODULE_H_
#define SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKMODULE_H_

#include <uem_common.h>

#include <uem_data.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief Initialize modules.
 *
 * This function initializes modules. Supported modules are located in src/module folder.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - corresponding error results from each module's fnInitialize() function.
 */
uem_result UKModule_Initialize();

/**
 * @brief Finalize modules.
 *
 * This function finalizes modules. Supported modules are located in src/module folder.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - corresponding error results from each module's fnFinalize() function.
 */
uem_result UKModule_Finalize();

extern int g_nModuleNum;
extern SAddOnModule g_stModules[];

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKMODULE_H_ */
