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

uem_result UKSerialModule_Initialize();
uem_result UKSerialModule_Finalize();

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKSERIALMODULE_H_ */
