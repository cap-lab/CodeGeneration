/*
 * uem_bluetooth_data.h
 *
 *  Created on: 2018. 10. 8.
 *      Author: chjej202
 */

#ifndef SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UEM_BLUETOOTH_DATA_H_
#define SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UEM_BLUETOOTH_DATA_H_

#include <uem_common.h>

#include <UCThreadMutex.h>
#include <UCDynamicSocket.h>
#include <UCSerialPort.h>
#include <UCFixedSizeQueue.h>


#include <UKConnector.h>
#include <UKSerialCommunicationManager.h>

#include <uem_channel_data.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct _SBluetoothInfo {
	const char *pszTargetMacAddress;
	HSocket hSocket;
	HThread hHandlingThread;
	HConnector hConnector;
	int nMaxChannelAccessNum;
	HSerialCommunicationManager hManager;
	uem_bool bInitialized;
} SBluetoothInfo;

typedef struct _SSerialInfo {
	const char *pszSerialPortPath;
	HSerialPort hSerialPort;
	HThread hHandlingThread;
	HConnector hConnector;
	int nMaxChannelAccessNum;
	HSerialCommunicationManager hManager;
	uem_bool bInitialized;
} SSerialInfo;

typedef struct _SSerialWriterChannel {
	void *pConnectionInfo;
	HFixedSizeQueue hRequestQueue;
	HThread hReceivingThread; // for WRITER channel
	char *pBuffer; // temporary buffer for getting data from shared memory channel (for WRITER)
	int nBufLen; // temporary buffer length (for WRITER)
	HThreadMutex hMutex;
	uem_bool bChannelExit;
	SSharedMemoryChannel *pstInternalChannel; // for WRITER
} SSerialWriterChannel;

typedef struct _SSerialReaderChannel {
	void *pConnectionInfo;
	HFixedSizeQueue hResponseQueue;
	// SSerialCommunicationManager handle; =>
	// hResponseQueue; // single-size queue
	HThreadMutex hMutex;
	uem_bool bChannelExit;
	SGenericMemoryAccess *pstReaderAccess; // for READER channel
} SSerialReaderChannel;

extern SBluetoothInfo g_astBluetoothMasterInfo[];
extern int g_nBluetoothMasterNum;

extern SBluetoothInfo g_astBluetoothSlaveInfo[];
extern int g_nBluetoothSlaveNum;

extern SSerialInfo g_astSerialMasterInfo[];
extern int g_nSerialMasterInfoNum;

extern SSerialInfo g_astSerialSlaveInfo[];
extern int g_nSerialSlaveInfoNum;



#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UEM_BLUETOOTH_DATA_H_ */
