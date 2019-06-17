/*
 * UKTCPCommunication.h
 *
 *  Created on: 2019. 5. 23.
 *      Author: jej
 */

#ifndef SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKTCPCOMMUNICATION_H_
#define SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKTCPCOMMUNICATION_H_

#include <uem_common.h>

#include <UKVirtualCommunication.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief Create a TCP communication socket.
 *
 * This function creates a TCP communication socket. \n
 * This is an implementation function of SVirtualCommunicationAPI's fnCreate().
 *
 * @param[out] phSocket phSocket a socket handle to be created.
 * @param pSocketInfo TCP socket options.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_OUT_OF_MEMORY, @ref ERR_UEM_NOT_SUPPORTED, \n
 *         @ref ERR_UEM_NOT_FOUND.
 */
uem_result UKTCPCommunication_Create(OUT HVirtualSocket *phSocket, void *pSocketInfo);


#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKTCPCOMMUNICATION_H_ */
