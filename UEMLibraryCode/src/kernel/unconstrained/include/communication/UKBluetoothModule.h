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

uem_result UKBluetoothModule_Initialize();
uem_result UKBluetoothModule_Finalize();

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKBLUETOOTHMODULE_H_ */
