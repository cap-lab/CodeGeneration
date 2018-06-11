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

uem_result UKTCPServerManager_Initialize();
uem_result UKTCPServerManager_Finalize();

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKTCPSERVERMANAGER_H_ */
