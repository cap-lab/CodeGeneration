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

uem_result UKUEMLiteProtocol_Create(OUT HUEMLiteProtocol *phProtocol);
uem_result UKUEMLiteProtocol_SetVirtualSocket(HUEMLiteProtocol hProtocol, HVirtualSocket hSocket, SVirtualCommunicationAPI *pstAPI);
uem_result UKUEMLiteProtocol_HandShake(HUEMLiteProtocol hProtocol, HUEMLiteProtocol hReceiveProtocol, unsigned int unDeviceKey);
uem_result UKUEMLiteProtocol_SetReadQueueRequest(HUEMLiteProtocol hProtocol, int nChannelId, int nSizeToRead);
uem_result UKUEMLiteProtocol_SetReadBufferRequest(HUEMLiteProtocol hProtocol, int nChannelId, int nSizeToRead);
uem_result UKUEMLiteProtocol_SetAvailableDataRequest(HUEMLiteProtocol hProtocol, int nChannelId);
uem_result UKUEMLiteProtocol_SetResultMessage(HUEMLiteProtocol hProtocol, EMessageType enRequestType, int nChannelId, EProtocolError enErrorCode, int nReturnValue);
uem_result UKUEMLiteProtocol_SetResultMessageHeaderUsingBuffer(HUEMLiteProtocol hProtocol, EMessageType enRequestType, int nChannelId, EProtocolError enErrorCode, int nDataSize, void *pData);
uem_result UKUEMLiteProtocol_Send(HUEMLiteProtocol hProtocol);
uem_result UKUEMLiteProtocol_Receive(HUEMLiteProtocol hProtocol);
uem_result UKUEMLiteProtocol_GetHeaderFromReceivedData(HUEMLiteProtocol hProtocol, OUT int *pnChannelId, OUT EMessageType *penMessageType, OUT int *pnParamNum, OUT short **ppasParam);
uem_result UKUEMLiteProtocol_GetBodyDataFromReceivedData(HUEMLiteProtocol hProtocol, OUT int *pnBodySize, OUT void **ppBody);
uem_result UKUEMLiteProtocol_Destroy(IN OUT HUEMLiteProtocol *phProtocol);

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKUEMLITEPROTOCOL_H_ */
