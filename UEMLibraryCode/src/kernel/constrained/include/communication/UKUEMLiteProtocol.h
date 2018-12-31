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


uem_result UKUEMLiteProtocol_Receive(HSerial hSerial);
uem_result UKUEMLiteProtocol_SetReadQueueRequest(int nChannelId, int nSizeToRead);
uem_result UKUEMLiteProtocol_SetReadBufferRequest(int nChannelId, int nSizeToRead);
uem_result UKUEMLiteProtocol_SetAvailableDataRequest(int nChannelId);
uem_result UKUEMLiteProtocol_SetResultMessage(EMessageType enRequestType, int nChannelId, EProtocolError enErrorCode, int nReturnValue);
uem_result UKUEMLiteProtocol_SetResultMessageHeaderUsingBuffer(EMessageType enRequestType, int nChannelId, EProtocolError enErrorCode, int nDataSize);
uem_result UKUEMLiteProtocol_GetResultBufferToSend(OUT unsigned char **ppbyBuffer, OUT int *pnBufferSize);
uem_result UKUEMLiteProtocol_Send(HSerial hSerial);
uem_result UKUEMLiteProtocol_GetHeaderFromReceivedData(OUT int *pnChannelId, OUT EMessageType *penMessageType, OUT int *pnParamNum, OUT short **ppanParam);
uem_result UKUEMLiteProtocol_GetBodyDataFromReceivedData(OUT int *pnBodySize, OUT void **ppBody);


#ifdef __cplusplus
}
#endif


#endif /* SRC_KERNEL_CONSTRAINED_INCLUDE_COMMUNICATION_UKUEMLITEPROTOCOL_H_ */
