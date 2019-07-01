/*
 * UKSharedMemoryMulticast.h
 *
 *  Created on: 2019. 11. 9.
 *      Author: wecracy
 */

#ifndef SRC_KERNEL_INCLUDE_UKSHAREDMEMORYMULTICAST_H_
#define SRC_KERNEL_INCLUDE_UKSHAREDMEMORYMULTICAST_H_

#include <uem_common.h>

#include <uem_data.h>

#ifdef __cplusplus
extern "C"
{
#endif

uem_result UKSharedMemoryMulticast_Initialize(SMulticastGroup *pstMulticastGroup);
uem_result UKSharedMemoryMulticast_ReadFromBuffer(SMulticastGroup *pstMulticastGroup, IN OUT unsigned char *pBuffer, IN int nDataToRead, OUT int *pnDataRead);
uem_result UKSharedMemoryMulticast_WriteToBuffer (SMulticastGroup *pstMulticastGroup, IN unsigned char *pBuffer, IN int nDataToWrite, OUT int *pnDataWritten);
uem_result UKSharedMemoryMulticast_Clear(SMulticastGroup *pstMulticastGroup);
uem_result UKSharedMemoryMulticast_Finalize(SMulticastGroup *pstMulticastGroup);

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_INCLUDE_UKSHAREDMEMORYMULTICAST_H_ */
