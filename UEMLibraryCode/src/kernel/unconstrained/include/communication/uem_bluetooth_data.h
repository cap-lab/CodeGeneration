/*
 * uem_bluetooth_data.h
 *
 *  Created on: 2018. 10. 8.
 *      Author: chjej202
 */

#ifndef SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UEM_BLUETOOTH_DATA_H_
#define SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UEM_BLUETOOTH_DATA_H_

#include <uem_common.h>

#include <UKVirtualCommunication.h>

#include <uem_remote_data.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct _SBluetoothAccessInfo {
	char *pszTargetMacAddress;
	EMasterSlavePairType enType;
} BluetoothAccessInfo;

typedef struct _SBluetoothInfo {
	SAggregateServiceInfo stAggregateInfo;
	BluetoothAccessInfo stAccessInfo;
} SBluetoothInfo;


extern SBluetoothInfo g_astBluetoothMasterInfo[];
extern int g_nBluetoothMasterNum;

extern SBluetoothInfo g_astBluetoothSlaveInfo[];
extern int g_nBluetoothSlaveNum;


#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UEM_BLUETOOTH_DATA_H_ */
