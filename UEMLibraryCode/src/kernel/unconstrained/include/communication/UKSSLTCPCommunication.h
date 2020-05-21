/*
 * UKTCPCommunication.h
 *
 *  Created on: 2019. 5. 23.
 *      Author: jej
 */

#ifndef SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKSSLTCPCOMMUNICATION_H_
#define SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKSSLTCPCOMMUNICATION_H_

#include <uem_common.h>

#include <UKVirtualCommunication.h>

#ifdef __cplusplus
extern "C"
{
#endif

uem_result UKSSLTCPCommunication_Create(OUT HVirtualSocket *phSocket, void *pSocketInfo);

uem_result UKSSLTCPCommunication_Destroy(HVirtualSocket *phSocket);

uem_result UKSSLTCPCommunication_Connect(HVirtualSocket hSocket, int nTimeout);

uem_result UKSSLTCPCommunication_Disconnect(HVirtualSocket hSocket);

uem_result UKSSLTCPCommunication_Listen(HVirtualSocket hSocket);

uem_result UKSSLTCPCommunication_Accept(HVirtualSocket hSocket, int nTimeout, IN OUT HVirtualSocket hAcceptedSocket);

uem_result UKSSLTCPCommunication_Send(HVirtualSocket hSocket, IN int nTimeout, IN char *pData, IN int nDataLen, OUT int *pnSentSize);

uem_result UKSSLTCPCommunication_Receive(HVirtualSocket hSocket, IN int nTimeout, IN OUT char *pBuffer, IN int nBufferLen, OUT int *pnReceivedSize);

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKSSLTCPCOMMUNICATION_H_ */
