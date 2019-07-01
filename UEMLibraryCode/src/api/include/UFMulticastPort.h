/*
 * UFMulticastPort.h
 *
 *  Created on: 2019. 6. 20.
 *      Author: wecracy
 */

#ifndef SRC_API_INCLUDE_UFMULTICASTPORT_H_
#define SRC_API_INCLUDE_UFMULTICASTPORT_H_


#include <uem_common.h>

#ifdef __cplusplus
extern "C"
{
#endif

uem_result UFMulticastPort_Initialize(IN int nTaskId, IN char *szPortName, OUT int *pnMulticastGroupId, OUT int *pnMulticastPortId);
uem_result UFMulticastPort_ReadFromBuffer (IN int nMulticastGroupId, IN int nMulticastPortId, IN OUT unsigned char *pBuffer, IN int nDataToRead, OUT int *pnDataRead);
uem_result UFMulticastPort_WriteToBuffer (IN int nMulticastGroupId, IN int nMulticastPortId, IN unsigned char *pBuffer, IN int nDataToWrite, OUT int *pnDataWritten);
uem_result UFMulticastPort_GetMulticastSize(IN int nMulticastGroupId, OUT int *pnMulticastGroupSize);

#ifdef __cplusplus
}
#endif

#endif /* SRC_API_INCLUDE_UFMULTICASTPORT_H_ */
