/*
 * uem_serial_data.h
 *
 *  Created on: 2019. 5. 29.
 *      Author: jej
 */

#ifndef SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UEM_SERIAL_DATA_H_
#define SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UEM_SERIAL_DATA_H_

#include <uem_common.h>

#include <UKVirtualCommunication.h>

#include <uem_remote_data.h>

#ifdef __cplusplus
extern "C"
{
#endif


typedef struct _SSerialAccessInfo {
	char *pszSerialPortPath;
	EMasterSlavePairType enType;
} SSerialAccessInfo;

typedef struct _SSerialInfo {
	SAggregateServiceInfo stAggregateInfo;
	SSerialAccessInfo stAccessInfo;
} SSerialInfo;


extern SSerialInfo g_astSerialMasterInfo[];
extern int g_nSerialMasterInfoNum;

extern SSerialInfo g_astSerialSlaveInfo[];
extern int g_nSerialSlaveInfoNum;


#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UEM_SERIAL_DATA_H_ */
