/*
 * UKSerialCommunicationManager.h
 *
 *  Created on: 2018. 10. 6.
 *      Author: chjej202
 */

#ifndef SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKSERIALCOMMUNICATIONMANAGER_H_
#define SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKSERIALCOMMUNICATIONMANAGER_H_

#include <uem_common.h>

#include <UKConnector.h>
#include <uem_protocol_data.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct _SSerialCommunicationManager *HSerialCommunicationManager;


typedef struct _SRequestItem {
	int nRequestDataSize;
} SRequestItem;

typedef struct _SResponseItem {
	EMessageType enRequestMessageType;
	int nReturnValue;
	int nDataSize;
	void *pData; // pointer to temporary buffer of each writer channel (does not need to be freed)
} SResponseItem;

typedef union _USendItem {
	SRequestItem stRequest;
	SResponseItem stResponse;
} USendItem;

typedef struct _SItemToSend {
	EMessageType enMessageType;
	int nChannelId;
	USendItem uDetailItem;
} SCommunicationQueueItem;


uem_result UKSerialCommunicationManager_Create(HConnector hConnector, int nMaximumSend, OUT HSerialCommunicationManager *phManager);
uem_result UKSerialCommunicationManager_Destroy(IN OUT HSerialCommunicationManager *phManager);
uem_result UKSerialCommunicationManager_Run(HSerialCommunicationManager hManager);
uem_result UKSerialCommunicationManager_PutItemToSend(HSerialCommunicationManager hManager, SCommunicationQueueItem *pstItem);
uem_result UKSerialCommunicationManager_SetChannel(HSerialCommunicationManager hManager, int nChannelId);
uem_result UKSerialCommunicationManager_ReleaseChannel(HSerialCommunicationManager hManager, int nChannelId);
uem_result UKSerialCommunicationManager_GetChannelQueue(HSerialCommunicationManager hManager, int nChannelId, HFixedSizeQueue *phQueue);
uem_result UKSerialCommunicationManager_Handshake(HSerialCommunicationManager hManager);
uem_result UKSerialCommunicationManager_AcceptHandshake(HSerialCommunicationManager hManager);


#ifdef __cplusplus
}
#endif


#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKSERIALCOMMUNICATIONMANAGER_H_ */
