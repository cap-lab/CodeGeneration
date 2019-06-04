/*
 * UKSerialCommunicationManager.h
 *
 *  Created on: 2018. 10. 6.
 *      Author: chjej202
 */

#ifndef SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKSERIALCOMMUNICATIONMANAGER_H_
#define SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKSERIALCOMMUNICATIONMANAGER_H_

#include <uem_common.h>

#include <UKVirtualCommunication.h>

#include <uem_protocol_data.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct _SSerialCommunicationManager *HSerialCommunicationManager;


typedef struct _SRequestItem {
	int nRequestDataSize;
	int nChunkIndex;
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

/**
 * @brief
 *
 * This function
 *
 * @param hSocket
 * @param pstAPI
 * @param nMaxChannelNum
 * @param[out] phManager
 *
 * @return
 */
uem_result UKSerialCommunicationManager_Create(HVirtualSocket hSocket, SVirtualCommunicationAPI *pstAPI, int nMaxChannelNum, OUT HSerialCommunicationManager *phManager);

/**
 * @brief
 *
 * This function
 *
 * @param phManager
 *
 * @return
 */
uem_result UKSerialCommunicationManager_Destroy(IN OUT HSerialCommunicationManager *phManager);

/**
 * @brief
 *
 * This function
 *
 * @param hManager
 *
 * @return
 */
uem_result UKSerialCommunicationManager_Run(HSerialCommunicationManager hManager);

/**
 * @brief
 *
 * This function
 *
 * @param hManager
 * @param pstItem
 *
 * @return
 */
uem_result UKSerialCommunicationManager_PutItemToSend(HSerialCommunicationManager hManager, SCommunicationQueueItem *pstItem);

/**
 * @brief
 *
 * This function
 *
 * @param hManager
 * @param nChannelId
 *
 * @return
 */
uem_result UKSerialCommunicationManager_SetChannel(HSerialCommunicationManager hManager, int nChannelId);

/**
 * @brief
 *
 * This function
 *
 * @param hManager
 * @param nChannelId
 *
 * @return
 */
uem_result UKSerialCommunicationManager_ReleaseChannel(HSerialCommunicationManager hManager, int nChannelId);

/**
 * @brief
 *
 * This function
 *
 * @param hManager
 * @param nChannelId
 * @param[out] phQueue
 *
 * @return
 */
uem_result UKSerialCommunicationManager_GetChannelQueue(HSerialCommunicationManager hManager, int nChannelId, OUT HFixedSizeQueue *phQueue);

/**
 * @brief
 *
 * This function
 *
 * @param hManager
 *
 * @return
 */
uem_result UKSerialCommunicationManager_Handshake(HSerialCommunicationManager hManager);

/**
 * @brief
 *
 * This function
 *
 * @param hManager
 *
 * @return
 */
uem_result UKSerialCommunicationManager_AcceptHandshake(HSerialCommunicationManager hManager);


#ifdef __cplusplus
}
#endif


#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKSERIALCOMMUNICATIONMANAGER_H_ */
