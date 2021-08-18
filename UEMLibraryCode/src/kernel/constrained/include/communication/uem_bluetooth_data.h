/*
 * uem_bluetooth_data.h
 *
 *  Created on: 2018. 10. 23.
 *      Author: jej
 */

#ifndef SRC_KERNEL_CONSTRAINED_INCLUDE_COMMUNICATION_UEM_BLUETOOTH_DATA_H_
#define SRC_KERNEL_CONSTRAINED_INCLUDE_COMMUNICATION_UEM_BLUETOOTH_DATA_H_

#include <uem_common.h>

#include <uem_enum.h>

#include <UCSerial.h>

#include <uem_channel_data.h>
#include <uem_protocol_data.h>

#ifdef __cplusplus
extern "C"
{
#endif



// sender request or received request
typedef struct _SRequestInfo {
	EMessageType enMessageType;
	int nDataToRead;
} SRequestInfo;

//SChannel *g_pastSerialChannelList[nMaxChannelRequestNum];


//SoftwareSerial BT_Serial(7, 8);


typedef struct _SSerialChannel SSerialChannel;


typedef struct _SSerialInfo {
	HSerial hSerial;
	int nMaxChannelAccessNum;
	int nSetChannelAccessNum;
	SChannel **ppastAccessChannelList;
} SSerialInfo;


typedef struct _SSerialChannel {
	SSerialInfo *pstSerialInfo;
	SRequestInfo stRequestInfo; // pointer to self-maintained request info slot
	SSharedMemoryChannel *pstInternalChannel;
} SSerialChannel;


extern SSerialInfo g_astSerialMasterInfo[];
extern int g_nSerialMasterNum;

extern SSerialInfo g_astSerialSlaveInfo[];
extern int g_nSerialSlaveNum;


#ifdef __cplusplus
}
#endif


#endif /* SRC_KERNEL_CONSTRAINED_INCLUDE_COMMUNICATION_UEM_BLUETOOTH_DATA_H_ */
