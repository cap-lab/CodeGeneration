/*
 * UKBluetoothModule.h
 *
 *  Created on: 2018. 10. 17.
 *      Author: chjej202
 */

#ifndef SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKBLUETOOTHMODULE_H_
#define SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKBLUETOOTHMODULE_H_

#include <uem_common.h>


#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief Initialize Bluetooth module.
 *
 * This function initialize Bluetooth aggregate clients and services. \n
 * This function establish connections with other Bluetooth devices for channel communication.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_OUT_OF_MEMORY, @ref ERR_UEM_INTERNAL_FAIL, and \n
 *         errors corresponding to @ref SVirtualCommunication fnCreate().
 */
uem_result UKBluetoothModule_Initialize();

/**
 * @brief Finalize Bluetooth module.
 *
 * This function finalizes Bluetooth aggregate clients and services. \n
 * This function destroys connections with other Bluetooth devices.
 *
 * @return This function always returns @ref ERR_UEM_NOERROR. \n
 *
 */
uem_result UKBluetoothModule_Finalize();

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKBLUETOOTHMODULE_H_ */
