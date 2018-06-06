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

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct _SUEMProtocol *HUEMProtocol;

uem_result UKUEMProtocol_Create(OUT HUEMProtocol *phProtocol);
uem_result UKUEMProtocol_SetSocket(HUEMProtocol hProtocol, HSocket hSocket);
uem_result UKUEMProtocol_HandShake(HUEMProtocol hProtocol, unsigned int unDeviceKey, int nChannelId);
uem_result UKUEMProtocol_SetReadQueueRequest(HUEMProtocol hProtocol, int nIndex, int nSizeToRead);
uem_result UKUEMProtocol_SetReadBufferRequest(HUEMProtocol hProtocol, int nIndex, int nSizeToRead);
uem_result UKUEMProtocol_SetAvailableIndexRequest(HUEMProtocol hProtocol);
uem_result UKUEMProtocol_SetAvailableDataRequest(HUEMProtocol hProtocol, int nIndex);
uem_result UKUEMProtocol_SetResultMessage(HUEMProtocol hProtocol, EProtocolError enErrorCode, int nReturnValue);
uem_result UKUEMProtocol_SetResultMessageWithBuffer(HUEMProtocol hProtocol, EProtocolError enErrorCode, int nDataSize, void *pData);
uem_result UKUEMProtocol_Send(HUEMProtocol hProtocol);
uem_result UKUEMProtocol_Receive(HUEMProtocol hProtocol);
uem_result UKUEMProtocol_GetResultFromReceivedData(HUEMProtocol hProtocol, OUT EProtocolError *penErrorCode, OUT int *pnReturnValue);
uem_result UKUEMProtocol_GetBodyDataFromReceivedData(HUEMProtocol hProtocol, OUT int *pnBodySize, OUT void **ppBody);
uem_result UKUEMProtocol_Destroy(IN OUT HUEMProtocol *phProtocol);

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKUEMPROTOCOL_H_ */
