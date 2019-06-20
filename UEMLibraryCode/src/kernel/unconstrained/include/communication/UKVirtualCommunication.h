/*
 * UKVirtualCommunication.h
 *
 *  Created on: 2019. 5. 23.
 *      Author: jej
 */

#ifndef SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKVIRTUALCOMMUNICATION_H_
#define SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKVIRTUALCOMMUNICATION_H_

#include <uem_common.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef void *HVirtualSocket;

typedef uem_result (*FnVirtualCommunicationCreate)(OUT HVirtualSocket *phSocket, void *pSocketInfo);
typedef uem_result (*FnVirtualCommunicationDestroy)(HVirtualSocket *phSocket);
typedef uem_result (*FnVirtualCommunicationConnect)(HVirtualSocket hSocket, int nTimeout);
typedef uem_result (*FnVirtualCommunicationDisconnect)(HVirtualSocket hSocket);
typedef uem_result (*FnVirtualCommunicationListen)(HVirtualSocket hSocket);
typedef uem_result (*FnVirtualCommunicationAccept)(HVirtualSocket hSocket, int nTimeout, IN OUT HVirtualSocket hAcceptedSocket);
typedef uem_result (*FnVirtualCommunicationSend)(HVirtualSocket hSocket, IN int nTimeout, IN char *pData, IN int nDataLen, OUT int *pnSentSize);
typedef uem_result (*FnVirtualCommunicationReceive)(HVirtualSocket hSocket, IN int nTimeout, IN OUT char *pBuffer, IN int nBufferLen, OUT int *pnReceivedSize);


typedef struct _SVirtualCommunicationAPI {
	FnVirtualCommunicationCreate fnCreate;
	FnVirtualCommunicationDestroy fnDestroy;
	FnVirtualCommunicationConnect fnConnect;
	FnVirtualCommunicationDisconnect fnDisconnect;
	FnVirtualCommunicationListen fnListen;
	FnVirtualCommunicationAccept fnAccept;
	FnVirtualCommunicationSend fnSend;
	FnVirtualCommunicationReceive fnReceive;
} SVirtualCommunicationAPI;


#ifdef __cplusplus
}
#endif


#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKVIRTUALCOMMUNICATION_H_ */
