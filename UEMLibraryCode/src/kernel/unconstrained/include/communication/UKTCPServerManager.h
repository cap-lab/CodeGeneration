/*
 * UKTCPServerManager.h
 *
 *  Created on: 2018. 6. 11.
 *      Author: jej
 */

#ifndef SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKTCPSERVERMANAGER_H_
#define SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKTCPSERVERMANAGER_H_

#include <uem_common.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief Initialize TCP server.
 *
 * This function initializes a TCP server. \n
 * This function loads a server and accept clients from different devices.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_OUT_OF_MEMORY, and \n
 *         errors corresponding to @ref SVirtualCommunicationAPI fnCreate() and fnListen().
 */
uem_result UKTCPServerManager_Initialize();

/**
 * @brief Finalize TCP server.
 *
 * This function finalizes a TCP server. This function destroys a server.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_INTERNAL_FAIL, and \n
 *         errors corresponding to @ref SVirtualCommunicationAPI fnDestroy(). \n
 *         @ref ERR_UEM_INTERNAL_FAIL can be occurred when threads created by this module is not terminated properly.
 */
uem_result UKTCPServerManager_Finalize();

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKTCPSERVERMANAGER_H_ */
