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
 * @brief
 *
 * This function
 *
 * @param[out] phProtocol
 *
 * @return
 */
uem_result UKUEMProtocol_Create(OUT HUEMProtocol *phProtocol);

/**
 * @brief
 *
 * This function
 *
 * @param enMessageType
 * @param[out] pnParamNum
 *
 * @return
 */
uem_result UKUEMProtocol_GetMessageParamNumByMessageType(EMessageType enMessageType, OUT int *pnParamNum);

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
uem_result UKUEMProtocol_SetSocket(HUEMProtocol hProtocol, HVirtualSocket hSocket, SVirtualCommunicationAPI *pstAPI);

/**
 * @brief
 *
 * This function
 *
 * @param hProtocol
 * @param unDeviceKey
 * @param nChannelId
 *
 * @return
 */
uem_result UKUEMProtocol_HandShake(HUEMProtocol hProtocol, unsigned int unDeviceKey, int nChannelId);

/**
 * @brief
 *
 * This function
 *
 * @param hProtocol
 * @param nIndex
 * @param nSizeToRead
 *
 * @return
 */
uem_result UKUEMProtocol_SetReadQueueRequest(HUEMProtocol hProtocol, int nIndex, int nSizeToRead);

/**
 * @brief
 *
 * This function
 *
 * @param hProtocol
 * @param nIndex
 * @param nSizeToRead
 *
 * @return
 */
uem_result UKUEMProtocol_SetReadBufferRequest(HUEMProtocol hProtocol, int nIndex, int nSizeToRead);

/**
 * @brief
 *
 * This function
 *
 * @param hProtocol
 *
 * @return
 */
uem_result UKUEMProtocol_SetAvailableIndexRequest(HUEMProtocol hProtocol);

/**
 * @brief
 *
 * This function
 *
 * @param hProtocol
 * @param nIndex
 *
 * @return
 */
uem_result UKUEMProtocol_SetAvailableDataRequest(HUEMProtocol hProtocol, int nIndex);

/**
 * @brief
 *
 * This function
 *
 * @param hProtocol
 * @param enErrorCode
 * @param nReturnValue
 *
 * @return
 */
uem_result UKUEMProtocol_SetResultMessage(HUEMProtocol hProtocol, EProtocolError enErrorCode, int nReturnValue);

/**
 * @brief
 *
 * This function
 *
 * @param hProtocol
 * @param enErrorCode
 * @param nDataSize
 * @param pData
 *
 * @return
 */
uem_result UKUEMProtocol_SetResultMessageWithBuffer(HUEMProtocol hProtocol, EProtocolError enErrorCode, int nDataSize, void *pData);

/**
 * @brief
 *
 * This function
 *
 * @param hProtocol
 *
 * @return
 */
uem_result UKUEMProtocol_Send(HUEMProtocol hProtocol);

/**
 * @brief
 *
 * This function
 *
 * @param hProtocol
 *
 * @return
 */
uem_result UKUEMProtocol_Receive(HUEMProtocol hProtocol);

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
uem_result UKUEMProtocol_SetChannelId(HUEMProtocol hProtocol, int nChannelId);

/**
 * @brief
 *
 * This function
 *
 * @param hProtocol
 * @param[out] penMessageType
 * @param[out] pnParamNum
 * @param[out] ppanParam
 *
 * @return
 */
uem_result UKUEMProtocol_GetRequestFromReceivedData(HUEMProtocol hProtocol, OUT EMessageType *penMessageType, OUT int *pnParamNum, OUT int **ppanParam);

/**
 * @brief
 *
 * This function
 *
 * @param hProtocol
 * @param[out] penErrorCode
 * @param[out] pnReturnValue
 *
 * @return
 */
uem_result UKUEMProtocol_GetResultFromReceivedData(HUEMProtocol hProtocol, OUT EProtocolError *penErrorCode, OUT int *pnReturnValue);

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
uem_result UKUEMProtocol_GetBodyDataFromReceivedData(HUEMProtocol hProtocol, OUT int *pnBodySize, OUT void **ppBody);

/**
 * @brief
 *
 * This function
 *
 * @param[in,out] phProtocol
 *
 * @return
 */
uem_result UKUEMProtocol_Destroy(IN OUT HUEMProtocol *phProtocol);

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKUEMPROTOCOL_H_ */
