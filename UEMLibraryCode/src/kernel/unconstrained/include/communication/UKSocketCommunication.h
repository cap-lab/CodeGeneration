/*
 * UKSocketCommunication.h
 *
 *  Created on: 2019. 5. 23.
 *      Author: jej
 */

#ifndef SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKSOCKETCOMMUNICATION_H_
#define SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKSOCKETCOMMUNICATION_H_

#include <uem_common.h>

#include <UKVirtualCommunication.h>


#ifdef __cplusplus
extern "C"
{
#endif

uem_result UKSocketCommunication_Destroy(HVirtualSocket *phSocket);
uem_result UKSocketCommunication_Connect(HVirtualSocket hSocket, int nTimeout);
uem_result UKSocketCommunication_Disconnect(HVirtualSocket hSocket);
uem_result UKSocketCommunication_Listen(HVirtualSocket hSocket);
uem_result UKSocketCommunication_Accept(HVirtualSocket hSocket, int nTimeout, IN OUT HVirtualSocket hAcceptedSocket);
uem_result UKSocketCommunication_Send(HVirtualSocket hSocket, IN int nTimeout, IN char *pData, IN int nDataLen, OUT int *pnSentSize);
uem_result UKSocketCommunication_Receive(HVirtualSocket hSocket, IN int nTimeout, IN OUT char *pBuffer, IN int nBufferLen, OUT int *pnReceivedSize);

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKSOCKETCOMMUNICATION_H_ */
