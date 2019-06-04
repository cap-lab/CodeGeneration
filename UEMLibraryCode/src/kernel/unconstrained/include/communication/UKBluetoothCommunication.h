/*
 * UKBluetoothCommunication.h
 *
 *  Created on: 2019. 5. 23.
 *      Author: jej
 */

#ifndef SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKBLUETOOTHCOMMUNICATION_H_
#define SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKBLUETOOTHCOMMUNICATION_H_

#include <uem_common.h>

#include <UKVirtualCommunication.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief
 *
 * This function
 *
 * @param[out] phSocket
 * @param pSocketInfo
 * @return
 */
uem_result UKBluetoothCommunication_Create(OUT HVirtualSocket *phSocket, void *pSocketInfo);


#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKBLUETOOTHCOMMUNICATION_H_ */
