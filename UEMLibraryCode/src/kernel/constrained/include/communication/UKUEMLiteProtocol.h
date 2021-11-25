/*
 * UKUEMLiteProtocol.h
 *
 *  Created on: 2018. 10. 25.
 *      Author: jej
 */

#ifndef SRC_KERNEL_CONSTRAINED_INCLUDE_COMMUNICATION_UKUEMLITEPROTOCOL_H_
#define SRC_KERNEL_CONSTRAINED_INCLUDE_COMMUNICATION_UKUEMLITEPROTOCOL_H_

#include <uem_common.h>

#include <UCSerial.h>

#include <uem_protocol_data.h>

#include <UKVirtualEncryption.h>


#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief Set encryption key.
 *
 * This function sets encryption key.
 *
 * @param pstEncKeyInfo a encryption key info.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKUEMLiteProtocol_SetEncryptionKey(SEncryptionKeyInfo *pstEncKeyInfo);

/**
 * @brief Receive data.
 *
 * This function receives data. Data can be a request or result from remote device.
 *
 * @param hSerial a Serial handle.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_ILLEGAL_DATA,@ref ERR_UEM_NOT_SUPPORTED.
 */
uem_result UKUEMLiteProtocol_Receive(HSerial hSerial);

/**
 * @brief Set hanshake request.
 *
 * This function sets handshake request to begin a communication.
 *
 * @param unDeviceKey authentication key for handshaking.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKUEMLiteProtocol_SetHandShakeRequest(unsigned int unDeviceKey);

/**
 * @brief Set read queue request.
 *
 * This function sets channel read queue request.
 *
 * @param nChannelId a target channel ID to set a request.
 * @param nSizeToRead size of data to request.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKUEMLiteProtocol_SetReadQueueRequest(int nChannelId, int nSizeToRead);

/**
 * @brief Set read buffer request.
 *
 * This function sets channel read buffer request.
 *
 * @param nChannelId a target channel ID to set a request.
 * @param nSizeToRead size of data to request.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKUEMLiteProtocol_SetReadBufferRequest(int nChannelId, int nSizeToRead);

/**
 * @brief Set available data request.
 *
 * This function sets channel available data request.
 *
 * @param nChannelId a target channel ID to set a request.
 *
  * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKUEMLiteProtocol_SetAvailableDataRequest(int nChannelId);

/**
 * @brief Set result message without using data area.
 *
 * This function sets a result message as a response.
 *
 * @param enRequestType a corresponding request to result message.
 * @param nChannelId a target channel ID to set a result.
 * @param enErrorCode an error code number to send.
 * @param nReturnValue result value to send.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * 			Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKUEMLiteProtocol_SetResultMessage(EMessageType enRequestType, int nChannelId, EProtocolError enErrorCode, int nReturnValue);

/**
 * @brief Set result message with data area.
 *
 * This function sets a result message as a response with actual channel data.
 *
 * @param enRequestType a corresponding request to result message.
 * @param nChannelId a target channel ID to set a result.
 * @param enErrorCode an error code number to send.
 * @param nDataSize size of result data to attach.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKUEMLiteProtocol_SetResultMessageHeaderUsingBuffer(EMessageType enRequestType, int nChannelId, EProtocolError enErrorCode, int nDataSize);

/**
 * @brief Get result buffer to be sent.
 *
 * This function gets result buffer to be sent.
 *
 * @param ppbyBuffer pointer to buffer containing data to be sent.
 * @param[out] pnBufferSize size of result data.
 *
 * @return  @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *			Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKUEMLiteProtocol_GetResultBufferToSend(OUT unsigned char **ppbyBuffer, OUT int *pnBufferSize);

/**
 * @brief Send data which are set.
 *
 * This function sends data which are set by @ref UKUEMLiteProtocol_SetReadQueueRequest, @ref UKUEMLiteProtocol_SetReadBufferRequest, \n
 * @ref UKUEMLiteProtocol_SetAvailableDataRequest, @ref UKUEMLiteProtocol_SetResultMessage, \n
 * or @ref UKUEMLiteProtocol_SetResultMessageHeaderUsingBuffer.
 *
 * @param hSerial a Serial handle.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_NET_SEND_ERROR, @ref ERR_UEM_ILLEGAL_DATA.
 */
uem_result UKUEMLiteProtocol_Send(HSerial hSerial);

/**
 * @brief Get header information from received data.
 *
 * This function gets header information from received data.
 *
 * @param[out] pnChannelId a target channel ID received.
 * @param[out] penMessageType message type (@ref EMessageType).
 * @param[out] pnParamNum number of parameters of @a ppasParam.
 * @param[out] ppanParam received parameters in the header.
 *
 * @return  @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_ILLEGAL_DATA.
 */
uem_result UKUEMLiteProtocol_GetHeaderFromReceivedData(OUT int *pnChannelId, OUT EMessageType *penMessageType, OUT int *pnParamNum, OUT short **ppanParam);

/**
 * @brief Get body information from received data.
 *
 * This function gets body information from received data. In body, actual channel buffer data is stored.
 *
 * @param[out] pnBodySize size of body data.
 * @param[out] ppBody pointer to body data.
 *
 * @return It always returns @ref ERR_UEM_NOERROR.
 */
uem_result UKUEMLiteProtocol_GetBodyDataFromReceivedData(OUT int *pnBodySize, OUT void **ppBody);


#ifdef __cplusplus
}
#endif


#endif /* SRC_KERNEL_CONSTRAINED_INCLUDE_COMMUNICATION_UKUEMLITEPROTOCOL_H_ */
