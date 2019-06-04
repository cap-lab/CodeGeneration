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
 * @brief
 *
 * This function
 *
 * @param[out] phProtocol
 *
 * @return
 */
uem_result UKUEMLiteProtocol_Create(OUT HUEMLiteProtocol *phProtocol);

/**
 * @brief
 *
 * This function
 *
 * @param hProtocol
 * @param hSocket
 * @param pstAPI
 *
 * @return
 */
uem_result UKUEMLiteProtocol_SetVirtualSocket(HUEMLiteProtocol hProtocol, HVirtualSocket hSocket, SVirtualCommunicationAPI *pstAPI);

/**
 * @brief
 *
 * This function
 *
 * @param hProtocol
 * @param hReceiveProtocol
 * @param unDeviceKey
 *
 * @return
 */
uem_result UKUEMLiteProtocol_HandShake(HUEMLiteProtocol hProtocol, HUEMLiteProtocol hReceiveProtocol, unsigned int unDeviceKey);

/**
 * @brief
 *
 * This function
 *
 * @param hProtocol
 * @param nChannelId
 * @param nSizeToRead
 *
 * @return
 */
uem_result UKUEMLiteProtocol_SetReadQueueRequest(HUEMLiteProtocol hProtocol, int nChannelId, int nSizeToRead);

/**
 * @brief
 *
 * This function
 *
 * @param hProtocol
 * @param nChannelId
 * @param nSizeToRead
 *
 * @return
 */
uem_result UKUEMLiteProtocol_SetReadBufferRequest(HUEMLiteProtocol hProtocol, int nChannelId, int nSizeToRead);

/**
 * @brief
 *
 * This function
 *
 * @param hProtocol
 * @param nChannelId
 *
 * @return
 */
uem_result UKUEMLiteProtocol_SetAvailableDataRequest(HUEMLiteProtocol hProtocol, int nChannelId);

/**
 * @brief
 *
 * This function
 *
 * @param hProtocol
 * @param enRequestType
 * @param nChannelId
 * @param enErrorCode
 * @param nReturnValue
 *
 * @return
 */
uem_result UKUEMLiteProtocol_SetResultMessage(HUEMLiteProtocol hProtocol, EMessageType enRequestType, int nChannelId, EProtocolError enErrorCode, int nReturnValue);

/**
 * @brief
 *
 * This function
 *
 * @param hProtocol
 * @param enRequestType
 * @param nChannelId
 * @param enErrorCode
 * @param nDataSize
 * @param pData
 *
 * @return
 */
uem_result UKUEMLiteProtocol_SetResultMessageHeaderUsingBuffer(HUEMLiteProtocol hProtocol, EMessageType enRequestType, int nChannelId, EProtocolError enErrorCode, int nDataSize, void *pData);

/**
 * @brief
 *
 * This function
 *
 * @param hProtocol
 *
 * @return
 */
uem_result UKUEMLiteProtocol_Send(HUEMLiteProtocol hProtocol);

/**
 * @brief
 *
 * This function
 *
 * @param hProtocol
 *
 * @return
 */
uem_result UKUEMLiteProtocol_Receive(HUEMLiteProtocol hProtocol);

/**
 * @brief
 *
 * This function
 *
 * @param hProtocol
 * @param[out] pnChannelId
 * @param[out] penMessageType
 * @param[out] pnParamNum
 * @param[out] ppasParam
 * @return
 */
uem_result UKUEMLiteProtocol_GetHeaderFromReceivedData(HUEMLiteProtocol hProtocol, OUT int *pnChannelId, OUT EMessageType *penMessageType, OUT int *pnParamNum, OUT short **ppasParam);

/**
 * @brief
 *
 * This function
 *
 * @param hProtocol
 * @param[out] pnBodySize
 * @param[out] ppBody
 *
 * @return
 */
uem_result UKUEMLiteProtocol_GetBodyDataFromReceivedData(HUEMLiteProtocol hProtocol, OUT int *pnBodySize, OUT void **ppBody);

/**
 * @brief
 *
 * This function
 *
 * @param phProtocol
 *
 * @return
 */
uem_result UKUEMLiteProtocol_Destroy(IN OUT HUEMLiteProtocol *phProtocol);

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKUEMLITEPROTOCOL_H_ */
