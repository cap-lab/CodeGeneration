/*
 * UKUEMLiteProtocol.h
 *
 *  Created on: 2018. 10. 5.
 *      Author: chjej202
 */

#ifndef SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKUEMLITEPROTOCOL_H_
#define SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKUEMLITEPROTOCOL_H_

#include <uem_common.h>

#include <uem_protocol_data.h>

#include <UKConnector.h>

#ifdef __cplusplus
extern "C"
{
#endif


#define HANDSHAKE_DEVICE_KEY_LSB_INDEX (0)
#define HANDSHAKE_DEVICE_KEY_MSB_INDEX (0)

#define READ_QUEUE_CHANNEL_ID_INDEX (0)
#define READ_QUEUE_SIZE_TO_READ_INDEX (1)

#define READ_BUFFER_CHANNEL_ID_INDEX (0)
#define READ_BUFFER_SIZE_TO_READ_INDEX (1)

#define AVAILABLE_DATA_CHANNEL_ID_INDEX (0)

#define RESULT_CHANNEL_ID_INDEX (0)
#define RESULT_REQUEST_PACKET_INDEX (1)
#define RESULT_ERROR_CODE_INDEX (2)
#define RESULT_BODY_SIZE_INDEX (3)
#define RESULT_RETURN_VALUE_INDEX (3)


typedef struct _SUEMLiteProtocol *HUEMLiteProtocol;

uem_result UKUEMLiteProtocol_Create(OUT HUEMLiteProtocol *phProtocol);
uem_result UKUEMLiteProtocol_GetMessageParamNumByMessageType(EMessageType enMessageType, int *pnParamNum);
uem_result UKUEMLiteProtocol_SetConnector(HUEMLiteProtocol hProtocol, HConnector hConnector);
uem_result UKUEMLiteProtocol_HandShake(HUEMLiteProtocol hProtocol, HUEMLiteProtocol hReceiveProtocol, unsigned int unDeviceKey);
uem_result UKUEMLiteProtocol_SetReadQueueRequest(HUEMLiteProtocol hProtocol, int nChannelId, int nSizeToRead);
uem_result UKUEMLiteProtocol_SetReadBufferRequest(HUEMLiteProtocol hProtocol, int nChannelId, int nSizeToRead);
uem_result UKUEMLiteProtocol_SetAvailableDataRequest(HUEMLiteProtocol hProtocol, int nChannelId);
uem_result UKUEMLiteProtocol_SetResultMessage(HUEMLiteProtocol hProtocol, EMessageType enRequestType, int nChannelId, EProtocolError enErrorCode, int nReturnValue);
uem_result UKUEMLiteProtocol_SetResultMessageWithBuffer(HUEMLiteProtocol hProtocol, EMessageType enRequestType, int nChannelId, EProtocolError enErrorCode, int nDataSize, void *pData);
uem_result UKUEMLiteProtocol_Send(HUEMLiteProtocol hProtocol);
uem_result UKUEMLiteProtocol_Receive(HUEMLiteProtocol hProtocol);
uem_result UKUEMLiteProtocol_GetHeaderFromReceivedData(HUEMLiteProtocol hProtocol, OUT int *pnChannelId, OUT EMessageType *penMessageType, OUT int *pnParamNum, OUT short **ppanParam);
uem_result UKUEMLiteProtocol_GetBodyDataFromReceivedData(HUEMLiteProtocol hProtocol, OUT int *pnBodySize, OUT void **ppBody);
uem_result UKUEMLiteProtocol_Destroy(IN OUT HUEMLiteProtocol *phProtocol);

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKUEMLITEPROTOCOL_H_ */
