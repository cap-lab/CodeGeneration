/*
 * UCUDPSocket.h
 *
 *  Created on: 2019. 6. 20.
 *      Author: wecracy
 */

#ifndef SRC_COMMON_UNCONSTRAINED_INCLUDE_COMMUNICATION_UCUDPSOCKET_H_
#define SRC_COMMON_UNCONSTRAINED_INCLUDE_COMMUNICATION_UCUDPSOCKET_H_

#include <uem_common.h>

#include <UCDynamicSocket.h>

#ifdef __cplusplus
extern "C"
{
#endif

uem_result UCUDPSocket_Bind(HSocket hSocket);
uem_result UCUDPSocket_Create(HSocket hSocket, SSocketInfo *pstSocketInfo, uem_bool bIsServer);
uem_result UCUDPSocket_Sendto(HSocket hSocket, IN char *unClientAddress, IN int nTimeout, IN unsigned char *pData, IN int nDataLen, OUT int *pnSentSize);
uem_result UCUDPSocket_RecvFrom(HSocket hSocket, IN char *pszClientAddress,  IN int nTimeout, IN int nBufferLen, OUT char *pBuffer, OUT int *pnRecvSize);

#ifdef __cplusplus
}
#endif

#endif /* SRC_COMMON_UNCONSTRAINED_INCLUDE_COMMUNICATION_UCUDPSOCKET_H_ */
