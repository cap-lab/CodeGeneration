/*
 * UKUEMProtocol.h
 *
 *  Created on: 2018. 6. 6.
 *      Author: jej
 */

#ifndef SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKUEMPROTOCOL_H_
#define SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKUEMPROTOCOL_H_

#include <uem_common.h>

#include <UCDynamicSocket.h>

#include <uem_data.h>

#include <uem_protocol_data.h>

#include <UKVirtualCommunication.h>
#include <UKVirtualEncryption.h>

#ifdef __cplusplus
extern "C"
{
#endif


#define HANDSHAKE_DEVICE_KEY_INDEX (0)
#define HANDSHAKE_CHANNELD_ID_INDEX (1)

#define READ_QUEUE_CHANNEL_ID_INDEX (0)
#define READ_QUEUE_CHUNK_INDEX_INDEX (1)
#define READ_QUEUE_SIZE_TO_READ_INDEX (2)

#define READ_BUFFER_CHANNEL_ID_INDEX (0)
#define READ_BUFFER_CHUNK_INDEX_INDEX (1)
#define READ_BUFFER_SIZE_TO_READ_INDEX (2)

#define AVAILABLE_INDEX_CHANNEL_ID_INDEX (0)

#define AVAILABLE_DATA_CHANNEL_ID_INDEX (0)
#define AVAILABLE_DATA_CHUNK_INDEX_INDEX (1)


#define RESULT_ERROR_CODE_INDEX (0)
#define RESULT_BODY_SIZE_INDEX (1)
#define RESULT_RETURN_VALUE_INDEX (1)

typedef struct _SUEMProtocol *HUEMProtocol;

/**
 * @brief Create a UEM protocol.
 *
 * This function creates a UEM protocol which is used for TCP communication. \n
 *
 * @param[out] phProtocol a protocol handle to be created.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_OUT_OF_MEMORY.
 */
uem_result UKUEMProtocol_Create(OUT HUEMProtocol *phProtocol);

/**
 * @brief Get message parameter number based on message type.
 *
 * This function retrieves required message parameter number from message type.
 *
 * @param enMessageType message type.
 * @param[out] pnParamNum number of parameters for corresponding message type.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKUEMProtocol_GetMessageParamNumByMessageType(EMessageType enMessageType, OUT int *pnParamNum);

/**
 * @brief Set a virtual socket.
 *
 * This function sets a virtual socket and APIs to communicate with remote devices.
 *
 * @param hProtocol a protocol handle.
 * @param hSocket a virtual socket to set.
 * @param pstAPI corresponding virtual communication API set of a virtual socket.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKUEMProtocol_SetSocket(HUEMProtocol hProtocol, HVirtualSocket hSocket, SVirtualCommunicationAPI *pstAPI);

/**
 * @brief Set encryption key.
 *
 * This function sets encryption key.
 *
 * @param hProtocol a protocol handle.
 * @param pstEncKeyInfo a encryption key info.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKUEMProtocol_SetEncryptionKey(HUEMProtocol hProtocol, SEncryptionKeyInfo *pstEncKeyInfo);

/**
 * @brief Request handshake and get result from remote device.
 *
 * This function requests a handshake and get result from remote device.
 *
 * @param hProtocol a protocol handle.
 * @param unDeviceKey authentication key for handshaking.
 * @param nChannelId channel ID to perform handshake.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_INTERNAL_FAIL, \n
 *         @ref ERR_UEM_CONNECT_ERROR, and corresponding errors from fnReceive(), fnSend() of SVirtualCommunicationAPI. \n
 *         @ref ERR_UEM_CONNECT_ERROR can be happened if the remote device responses nothing. \n
 *         @ref ERR_UEM_INTERNAL_FAIL can be happened if the received data is malformed.
 */
uem_result UKUEMProtocol_HandShake(HUEMProtocol hProtocol, unsigned int unDeviceKey, int nChannelId);

/**
 * @brief Set read queue request.
 *
 * This function
 *
 * @param hProtocol a protocol handle.
 * @param nIndex chunk index to read.
 * @param nSizeToRead size of data to request.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKUEMProtocol_SetReadQueueRequest(HUEMProtocol hProtocol, int nIndex, int nSizeToRead);

/**
 * @brief Set read buffer request.
 *
 * This function
 *
 * @param hProtocol a protocol handle.
 * @param nIndex chunk index to read.
 * @param nSizeToRead size of data to request.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKUEMProtocol_SetReadBufferRequest(HUEMProtocol hProtocol, int nIndex, int nSizeToRead);

/**
 * @brief Set available index request.
 *
 * This function sets a request of an available chunk index to read.
 *
 * @param hProtocol a protocol handle.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKUEMProtocol_SetAvailableIndexRequest(HUEMProtocol hProtocol);

/**
 * @brief Set available data request.
 *
 * This function sets channel available data request which how much data placed in the target chunk index of channel.
 *
 * @param hProtocol a protocol handle.
 * @param nIndex a chunk index to check.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKUEMProtocol_SetAvailableDataRequest(HUEMProtocol hProtocol, int nIndex);

/**
 * @brief Set result message without using data area.
 *
 * This function sets a result message as a response.
 *
 * @param hProtocol a protocol handle.
 * @param enErrorCode an error code number to send.
 * @param nReturnValue result value to send.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKUEMProtocol_SetResultMessage(HUEMProtocol hProtocol, EProtocolError enErrorCode, int nReturnValue);

/**
 * @brief Set result message with data area.
 *
 * This function sets a result message as a response with actual channel data.
 *
 * @param hProtocol a protocol handle.
 * @param enErrorCode an error code number to send.
 * @param nDataSize size of result data to attach.
 * @param pData data to attach.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKUEMProtocol_SetResultMessageWithBuffer(HUEMProtocol hProtocol, EProtocolError enErrorCode, int nDataSize, void *pData);

/**
 * @brief Send data which are set.
 *
 * This function sends data which are set by @ref UKUEMProtocol_SetReadQueueRequest, @ref UKUEMProtocol_SetReadBufferRequest, \n
 * @ref UKUEMProtocol_SetAvailableIndexRequest, @ref UKUEMProtocol_SetAvailableDataRequest, @ref UKUEMProtocol_SetResultMessage, and \n
 * @ref UKUEMProtocol_SetResultMessageWithBuffer.
 *
 * @param hProtocol a protocol handle.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_INVALID_SOCKET, @ref ERR_UEM_ILLEGAL_DATA, \n
 *         @ref ERR_UEM_ALREADY_DONE, @ref ERR_UEM_OUT_OF_MEMORY, and corresponding results from SVirtualCommunicationAPI's fnSend(). \n
 *         @ref ERR_UEM_ILLEGAL_DATA is happened when data cannot be made as a packet. \n
 *         @ref ERR_UEM_ALREADY_DONE is occurred when the function is called without setting the request or response.
 */
uem_result UKUEMProtocol_Send(HUEMProtocol hProtocol);

/**
 * @brief Receive data.
 *
 * This function receives data. Data can be a request or result from remote device.
 *
 * @param hProtocol a protocol handle.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_INVALID_SOCKET, @ref ERR_UEM_ILLEGAL_DATA, \n
 *         @ref ERR_UEM_ILLEGAL_CONTROL, @ref ERR_UEM_OUT_OF_MEMORY, and corresponding results from SVirtualCommunicationAPI's fnReceive(). \n
 *         @ref ERR_UEM_ILLEGAL_CONTROL is occurred when the previously-received result is not handled yet.
 *         @ref ERR_UEM_ILLEGAL_DATA is occurred if the received data violates the protocol format.
 */
uem_result UKUEMProtocol_Receive(HUEMProtocol hProtocol);

/**
 * @brief Set protocol channel ID.
 *
 * This function sets a channel ID of a protocol handle.
 * This function is used for server-side channels which accept handshakes from clients.
 *
 * @param hProtocol a protocol handle.
 * @param nChannelId channel ID to set.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKUEMProtocol_SetChannelId(HUEMProtocol hProtocol, int nChannelId);

/**
 * @brief Get request information from received data.
 *
 * This function gets header information from received data.
 *
 * @param hProtocol a protocol handle.
 * @param[out] penMessageType message type (@ref EMessageType).
 * @param[out] pnParamNum number of parameters of @a ppanParam.
 * @param[out] ppanParam received parameters in the header.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_NO_DATA. \n
 *         @ref ERR_UEM_NO_DATA is occurred if this function is called before receiving data.
 */
uem_result UKUEMProtocol_GetRequestFromReceivedData(HUEMProtocol hProtocol, OUT EMessageType *penMessageType, OUT int *pnParamNum, OUT int **ppanParam);

/**
 * @brief Get result information from received data.
 *
 * This function gets result information from received data.
 *
 * @param hProtocol a protocol handle.
 * @param[out] penErrorCode an error code sent from remote device.
 * @param[out] pnReturnValue value sent from remote device.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_NO_DATA. \n
 *         @ref ERR_UEM_NO_DATA is occurred if this function is called before receiving data.
 */
uem_result UKUEMProtocol_GetResultFromReceivedData(HUEMProtocol hProtocol, OUT EProtocolError *penErrorCode, OUT int *pnReturnValue);

/**
 * @brief Get body information from received data.
 *
 * This function gets body information from received data. In body, actual channel buffer data is stored.
 *
 * @param hProtocol a protocol handle.
 * @param[out] pnBodySize size of body data.
 * @param[out] ppBody pointer to body data.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_NO_DATA. \n
 *         @ref ERR_UEM_NO_DATA is occurred if this function is called before receiving data.
 */
uem_result UKUEMProtocol_GetBodyDataFromReceivedData(HUEMProtocol hProtocol, OUT int *pnBodySize, OUT void **ppBody);

/**
 * @brief Destroy a UEM protocol.
 *
 * This function destroys a UEM protocol.
 *
 * @param[in,out] phProtocol a protocol handle to be destroyed.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKUEMProtocol_Destroy(IN OUT HUEMProtocol *phProtocol);

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKUEMPROTOCOL_H_ */
