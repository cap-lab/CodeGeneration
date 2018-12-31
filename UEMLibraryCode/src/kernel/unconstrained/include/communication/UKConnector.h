/*
 * UKConnector.h
 *
 *  Created on: 2018. 10. 8.
 *      Author: chjej202
 */

#ifndef SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKCONNECTOR_H_
#define SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKCONNECTOR_H_

#include <uem_common.h>



#ifdef __cplusplus
extern "C"
{
#endif

typedef struct _SConnector *HConnector;

typedef void *HUserConnector;

typedef uem_result (*FnCallbackConnectorSend)(HConnector hConnector, HUserConnector hUserHandle, IN int nTimeout, IN char *pData, IN int nDataLen, OUT int *pnSentSize);
typedef uem_result (*FnCallbackConnectorReceive)(HConnector hConnector, HUserConnector hUserHandle, IN int nTimeout, IN OUT char *pBuffer, IN int nBufferLen, OUT int *pnReceivedSize);

uem_result UKConnector_Create(OUT HConnector *phConnector);
uem_result UKConnector_Destroy(IN OUT HConnector *phConnector);
uem_result UKConnector_SetConnector(HConnector hConnector, HUserConnector hUserHandle, FnCallbackConnectorSend fnSend, FnCallbackConnectorReceive fnReceive);
uem_result UKConnector_Send(HConnector hConnector, IN int nTimeout, IN char *pData, IN int nDataLen, OUT int *pnSentSize);
uem_result UKConnector_Receive(HConnector hConnector, IN int nTimeout, IN OUT char *pBuffer, IN int nBufferLen, OUT int *pnReceivedSize);

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKCONNECTOR_H_ */
