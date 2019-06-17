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
 * @brief Create a serial communication manager.
 *
 * This function creates a serial communication manager. \n
 * Serial communication manager is used for aggregate connection and is a intermediate module between remote devices and local channels.
 *
 * @param hSocket a virtual socket handle which is used for send/receive data.
 * @param pstAPI a virtual socket API which contains send/receive functions.
 * @param nMaxChannelNum the maximum number of channels which uses a serial communication manager.
 * @param[out] phManager a serial communication manager to be created.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_OUT_OF_MEMORY, @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKSerialCommunicationManager_Create(HVirtualSocket hSocket, SVirtualCommunicationAPI *pstAPI, int nMaxChannelNum, OUT HSerialCommunicationManager *phManager);

/**
 * @brief Destroy a serial communication manager.
 *
 * This function destroys a serial communication manager.
 *
 * @param[in,out] phManager a serial communication manager to be destroyed.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKSerialCommunicationManager_Destroy(IN OUT HSerialCommunicationManager *phManager);

/**
 * @brief Run the serial communication manager.
 *
 * This function executes a serial communication manager.
 *
 * @param hManager a serial communication manager handle.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INTERNAL_FAIL, @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_SUSPEND. \n
 *         @ref ERR_UEM_SUSPEND is occurred when the program is going to exit.
 */
uem_result UKSerialCommunicationManager_Run(HSerialCommunicationManager hManager);

/**
 * @brief Put an item to send.
 *
 * This function puts an item to send and it is used by remote channels. \n
 * @ref SCommunicationQueueItem is used pass data with extra information. Data may be a request or result to a remote device.
 *
 * @param hManager a serial communication manager handle.
 * @param pstItem an item to send.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_SUSPEND. \n
 *         @ref ERR_UEM_SUSPEND is occurred when the program is going to exit.
 */
uem_result UKSerialCommunicationManager_PutItemToSend(HSerialCommunicationManager hManager, SCommunicationQueueItem *pstItem);

/**
 * @brief Register a channel id to the serial communication manager.
 *
 * This function registers a channel id to the serial communication manager. \n
 * Internally, the serial communication manager prepares a queue for newly-registered channel ID. \n
 * After calling this function, @ref UKSerialCommunicationManager_GetChannelQueue function can be called to get a corresponding channel queue \n
 * to receive requests or results.
 *
 * @param hManager a serial communication manager handle.
 * @param nChannelId a channel ID to be registered.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_ILLEGAL_DATA. \n
 *         @ref ERR_UEM_ILLEGAL_DATA can be occurred when the number of calling this function exceeds internal maximum channel access number.
 *
 * @sa UKSerialCommunicationManager_GetChannelQueue, @ref UKSerialCommunicationManager_ReleaseChannel.
 */
uem_result UKSerialCommunicationManager_SetChannel(HSerialCommunicationManager hManager, int nChannelId);

/**
 * @brief Release a queue corresponding with an input channel id.
 *
 * This function releases a queue which is registered by @ref UKSerialCommunicationManager_SetChannel. \n
 * After all the channels calling this function, a serial communication manager can be destroyed.
 *
 * @param hManager a serial communication manager handle.
 * @param nChannelId a channel ID to be released.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_ILLEGAL_DATA. \n
 *         @ref ERR_UEM_ILLEGAL_DATA can be occurred when the input channel id is already released or never registered before.
 *
 * @sa UKSerialCommunicationManager_SetChannel.
 */
uem_result UKSerialCommunicationManager_ReleaseChannel(HSerialCommunicationManager hManager, int nChannelId);

/**
 * @brief Get queue for the corresponding channel.
 *
 * This function retrieves a queue for the corresponding channel. \n
 * By retrieved queue, a remote channel can get requests and results from the serial communication manager.
 *
 * @param hManager a serial communication manager handle.
 * @param nChannelId a channel ID.
 * @param[out] phQueue a handle of queue to get requests and results from serial communication manager.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_ILLEGAL_DATA. \n
 *         @ref ERR_UEM_ILLEGAL_DATA is occurred when the channel is not set yet.
 *
 * @sa UKSerialCommunicationManager_SetChannel.
 */
uem_result UKSerialCommunicationManager_GetChannelQueue(HSerialCommunicationManager hManager, int nChannelId, OUT HFixedSizeQueue *phQueue);

/**
 * @brief Handshake with remote device as a master role.
 *
 * This function performs a handshake with remote slave device.
 *
 * @param hManager a serial communication manager handle.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_ILLEGAL_DATA, \n
 *         @ref ERR_UEM_CONNECT_ERROR, and corresponding errors from fnReceive(), fnSend() of SVirtualCommunicationAPI. \n
 *         @ref ERR_UEM_CONNECT_ERROR can be happened if the remote device responses nothing. \n
 *         @ref ERR_UEM_ILLEGAL_DATA can be happened if the received data is malformed.
 */
uem_result UKSerialCommunicationManager_Handshake(HSerialCommunicationManager hManager);

/**
 * @brief Accept handshake from remote device as a slave role.
 *
 * This function accepts handshake requests from master device.
 *
 * @param hManager a serial communication manager handle.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_ILLEGAL_DATA, \n
 *         and corresponding errors from fnReceive(), fnSend() of SVirtualCommunicationAPI. \n
 *         @ref ERR_UEM_ILLEGAL_DATA can be happened if the received data is malformed.
 */
uem_result UKSerialCommunicationManager_AcceptHandshake(HSerialCommunicationManager hManager);


#ifdef __cplusplus
}
#endif


#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKSERIALCOMMUNICATIONMANAGER_H_ */
