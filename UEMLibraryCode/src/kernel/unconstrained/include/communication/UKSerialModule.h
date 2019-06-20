/*
 * UKSerialModule.h
 *
 *  Created on: 2019. 02. 18., modified from UKBluetoothModule.h
 *      Author: dowhan1128
 */

#ifndef SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKSERIALMODULE_H_
#define SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKSERIALMODULE_H_

#include <uem_common.h>


#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief Initialize serial port communication module.
 *
 * This function initializes serial port communication aggregate clients and services. \n
 * This function opens a serial port to communicate with other devices via wire or USB.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_OUT_OF_MEMORY, @ref ERR_UEM_INTERNAL_FAIL, and \n
 *         errors corresponding to @ref SVirtualCommunicationAPI fnCreate().
 */
uem_result UKSerialModule_Initialize();

/**
 * @brief Finalize serial port communication module.
 *
 * This function finalizes serial port communication aggregate clients and services. \n
 * This function closes serial ports to disconnect connections with other devices.
 *
 * @return This function always returns @ref ERR_UEM_NOERROR.
 */
uem_result UKSerialModule_Finalize();

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKSERIALMODULE_H_ */
