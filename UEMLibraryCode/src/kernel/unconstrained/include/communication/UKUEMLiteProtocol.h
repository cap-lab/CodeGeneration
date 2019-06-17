/*
 * UKUEMLiteProtocol.h
 *
 *  Created on: 2018. 10. 5.
 *      Author: chjej202
 */

#ifndef SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKUEMLITEPROTOCOL_H_
#define SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKUEMLITEPROTOCOL_H_

#include <uem_common.h>

#include <UKVirtualCommunication.h>
#include <uem_protocol_data.h>

#include <uem_lite_protocol_data.h>

#ifdef __cplusplus
extern "C"
{
#endif


typedef struct _SUEMLiteProtocol *HUEMLiteProtocol;

/**
 * @brief Create a UEM lite protocol.
 *
 * This function creates a UEM lite protocol which is used for Bluetooth or serial communication. \n
 * Compared to UEM Protocol, this protocol is a light-weight protocol but has restriction related to array queue. \n
 * With this protocol, loop task features cannot be used.
 *
 * @param[out] phProtocol a protocol handle to be created.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_OUT_OF_MEMORY.
 */
uem_result UKUEMLiteProtocol_Create(OUT HUEMLiteProtocol *phProtocol);

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
 */
uem_result UKUEMLiteProtocol_SetVirtualSocket(HUEMLiteProtocol hProtocol, HVirtualSocket hSocket, SVirtualCommunicationAPI *pstAPI);

/**
 * @brief Request handshake and get result from remote device.
 *
 * This function requests a handshake and get result from remote device.
 *
 * @param hProtocol a protocol handle for sending a handshake request.
 * @param hReceiveProtocol a protocol handle for receiving a handshake result.
 * @param unDeviceKey authentication key for handshaking.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_ILLEGAL_DATA, \n
 *         @ref ERR_UEM_CONNECT_ERROR, and corresponding errors from fnReceive(), fnSend() of SVirtualCommunicationAPI. \n
 *         @ref ERR_UEM_CONNECT_ERROR can be happened if the remote device responses nothing. \n
 *         @ref ERR_UEM_ILLEGAL_DATA can be happened if the received data is malformed.
 */
uem_result UKUEMLiteProtocol_HandShake(HUEMLiteProtocol hProtocol, HUEMLiteProtocol hReceiveProtocol, unsigned int unDeviceKey);

/**
 * @brief Set read queue request.
 *
 * This function sets channel read queue request.
 *
 * @param hProtocol a protocol handle.
 * @param nChannelId a target channel ID to set a request.
 * @param nSizeToRead size of data to request.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKUEMLiteProtocol_SetReadQueueRequest(HUEMLiteProtocol hProtocol, int nChannelId, int nSizeToRead);

/**
 * @brief Set read buffer request.
 *
 * This function sets channel read buffer request.
 *
 * @param hProtocol a protocol handle.
 * @param nChannelId a target channel ID to set a request.
 * @param nSizeToRead size of data to request.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKUEMLiteProtocol_SetReadBufferRequest(HUEMLiteProtocol hProtocol, int nChannelId, int nSizeToRead);

/**
 * @brief Set available data request.
 *
 * This function sets channel available data request.
 *
 * @param hProtocol a protocol handle.
 * @param nChannelId a target channel ID to set a request.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKUEMLiteProtocol_SetAvailableDataRequest(HUEMLiteProtocol hProtocol, int nChannelId);

/**
 * @brief Set result message without using data area.
 *
 * This function sets a result message as a response.
 *
 * @param hProtocol a protocol handle.
 * @param enRequestType a corresponding request to result message.
 * @param nChannelId a target channel ID to set a result.
 * @param enErrorCode an error code number to send.
 * @param nReturnValue result value to send.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKUEMLiteProtocol_SetResultMessage(HUEMLiteProtocol hProtocol, EMessageType enRequestType, int nChannelId, EProtocolError enErrorCode, int nReturnValue);

/**
 * @brief Set result message with data area.
 *
 * This function sets a result message as a response with actual channel data.
 *
 * @param hProtocol a protocol handle.
 * @param enRequestType a corresponding request to result message.
 * @param nChannelId a target channel ID to set a result.
 * @param enErrorCode an error code number to send.
 * @param nDataSize size of result data to attach.
 * @param pData result data to attach.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKUEMLiteProtocol_SetResultMessageUsingBuffer(HUEMLiteProtocol hProtocol, EMessageType enRequestType, int nChannelId, EProtocolError enErrorCode, int nDataSize, void *pData);

/**
 * @brief Send data which are set.
 *
 * This function sends data which are set by @ref UKUEMLiteProtocol_SetReadQueueRequest, @ref UKUEMLiteProtocol_SetReadBufferRequest, \n
 * @ref UKUEMLiteProtocol_SetAvailableDataRequest, @ref UKUEMLiteProtocol_SetResultMessage, \n
 * or @ref UKUEMLiteProtocol_SetResultMessageHeaderUsingBuffer.
 *
 * @param hProtocol a protocol handle.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_INVALID_SOCKET, @ref ERR_UEM_ILLEGAL_DATA, \n
 *         @ref ERR_UEM_ALREADY_DONE, and corresponding results from SVirtualCommunicationAPI's fnSend(). \n
 *         @ref ERR_UEM_ILLEGAL_DATA is happened when data cannot be made as a packet. \n
 *         @ref ERR_UEM_ALREADY_DONE is occurred when the function is called without setting the request or response.
 */
uem_result UKUEMLiteProtocol_Send(HUEMLiteProtocol hProtocol);

/**
 * @brief Receive data.
 *
 * This function receives data. Data can be a request or result from remote device.
 *
 * @param hProtocol a protocol handle.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_INVALID_SOCKET, @ref ERR_UEM_ILLEGAL_DATA, \n
 *         @ref ERR_UEM_ILLEGAL_CONTROL, and corresponding results from SVirtualCommunicationAPI's fnReceive(). \n
 *         @ref ERR_UEM_ILLEGAL_CONTROL is occurred when the previously-received result is not handled yet.
 *         @ref ERR_UEM_ILLEGAL_DATA is occurred if the received data violates the protocol format.
 */
uem_result UKUEMLiteProtocol_Receive(HUEMLiteProtocol hProtocol);

/**
 * @brief Get header information from received data.
 *
 * This function gets header information from received data.
 *
 * @param hProtocol a protocol handle.
 * @param[out] pnChannelId a target channel ID received.
 * @param[out] penMessageType message type (@ref EMessageType).
 * @param[out] pnParamNum number of parameters of @a ppasParam.
 * @param[out] ppasParam received parameters in the header.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_NO_DATA, @ref ERR_UEM_ILLEGAL_DATA. \n
 *         @ref ERR_UEM_NO_DATA is occurred if this function is called before receiving data. \n
 *         @ref ERR_UEM_ILLEGAL_DATA is occurred if the received message type is invalid.
 *
 */
uem_result UKUEMLiteProtocol_GetHeaderFromReceivedData(HUEMLiteProtocol hProtocol, OUT int *pnChannelId, OUT EMessageType *penMessageType, OUT int *pnParamNum, OUT short **ppasParam);

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
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKUEMLiteProtocol_GetBodyDataFromReceivedData(HUEMLiteProtocol hProtocol, OUT int *pnBodySize, OUT void **ppBody);

/**
 * @brief Destroy a UEM lite protocol.
 *
 * This function destroys a UEM lite protocol.
 *
 * @param[in,out] phProtocol a protocol handle to be destroyed.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKUEMLiteProtocol_Destroy(IN OUT HUEMLiteProtocol *phProtocol);

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKUEMLITEPROTOCOL_H_ */
