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


#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief
 *
 * This function
 *
 * @param hSerial
 *
 * @return
 */
uem_result UKUEMLiteProtocol_Receive(HSerial hSerial);

/**
 * @brief
 *
 * This function
 *
 * @param nChannelId
 * @param nSizeToRead
 *
 * @return
 */
uem_result UKUEMLiteProtocol_SetReadQueueRequest(int nChannelId, int nSizeToRead);

/**
 * @brief
 *
 * This function
 *
 * @param nChannelId
 * @param nSizeToRead
 *
 * @return
 */
uem_result UKUEMLiteProtocol_SetReadBufferRequest(int nChannelId, int nSizeToRead);

/**
 * @brief
 *
 * This function
 *
 * @param nChannelId
 *
 * @return
 */
uem_result UKUEMLiteProtocol_SetAvailableDataRequest(int nChannelId);

/**
 * @brief
 *
 * This function
 *
 * @param enRequestType
 * @param nChannelId
 * @param enErrorCode
 * @param nReturnValue
 *
 * @return
 */
uem_result UKUEMLiteProtocol_SetResultMessage(EMessageType enRequestType, int nChannelId, EProtocolError enErrorCode, int nReturnValue);

/**
 * @brief
 *
 * This function
 *
 * @param enRequestType
 * @param nChannelId
 * @param enErrorCode
 * @param nDataSize
 *
 * @return
 */
uem_result UKUEMLiteProtocol_SetResultMessageHeaderUsingBuffer(EMessageType enRequestType, int nChannelId, EProtocolError enErrorCode, int nDataSize);

/**
 * @brief
 *
 * This function
 *
 * @param ppbyBuffer
 * @param[out] pnBufferSize
 *
 * @return
 */
uem_result UKUEMLiteProtocol_GetResultBufferToSend(OUT unsigned char **ppbyBuffer, OUT int *pnBufferSize);

/**
 * @brief
 *
 * This function
 *
 * @param hSerial
 *
 * @return
 */
uem_result UKUEMLiteProtocol_Send(HSerial hSerial);

/**
 * @brief
 *
 * This function
 *
 * @param[out] pnChannelId
 * @param[out] penMessageType
 * @param[out] pnParamNum
 * @param[out] ppanParam
 * @return
 */
uem_result UKUEMLiteProtocol_GetHeaderFromReceivedData(OUT int *pnChannelId, OUT EMessageType *penMessageType, OUT int *pnParamNum, OUT short **ppanParam);

/**
 * @brief
 *
 * This function
 *
 * @param[out] pnBodySize
 * @param[out] ppBody
 *
 * @return
 */
uem_result UKUEMLiteProtocol_GetBodyDataFromReceivedData(OUT int *pnBodySize, OUT void **ppBody);


#ifdef __cplusplus
}
#endif


#endif /* SRC_KERNEL_CONSTRAINED_INCLUDE_COMMUNICATION_UKUEMLITEPROTOCOL_H_ */
