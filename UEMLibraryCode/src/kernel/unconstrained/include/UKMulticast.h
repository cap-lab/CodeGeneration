/*
 * UKMulticast.h
 *
 *  Created on: 2019. 6. 20.
 *      Author: wecracy
 */

#ifndef SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKMULTICAST_H_
#define SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKMULTICAST_H_

#include <uem_common.h>
#include <uem_multicast_data.h>

#ifdef __cplusplus
extern "C"
{
#endif

uem_result UKMulticast_Initialize();
uem_result UKMulticast_GetMulticastGroupIdByTaskAndPortName(IN int nTaskId, IN const char *szPortName, OUT int *pnMulticastGroupId);
uem_result UKMulticast_GetMulticastPortIdByTaskAndPortName(IN int nTaskId, IN const char *szPortName, OUT int *pnMulticastPortId);
uem_result UKMulticast_GetMulticastGroupSize(IN int nMulticastGroupId, OUT int *pnMulticastGroupSize);
uem_result UKMulticast_GetCommunication(IN SMulticastCommunication *astCommunicationList, IN int nCommunicationTypeNum, IN EMulticastCommunicationType enCommunicationType, OUT SMulticastCommunication **pstCommunication);
uem_result UKMulticast_WriteToBuffer(IN int nMulticastGroupId, IN int nMulticastPortId, IN unsigned char *pBuffer, IN int nDataToWrite, OUT int *pnDataWritten);
uem_result UKMulticast_ReadFromBuffer(IN int nMulticastGroupId, IN int nMulticastPortId, IN OUT unsigned char *pBuffer, IN int nDataToRead, OUT int *pnDataRead);
uem_result UKMulticast_Finalize();


#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKMULTICAST_H_ */
