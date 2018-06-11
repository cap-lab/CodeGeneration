/*
 * UKTCPSocketChannel.h
 *
 *  Created on: 2018. 6. 11.
 *      Author: jej
 */

#ifndef SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKTCPSOCKETCHANNEL_H_
#define SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKTCPSOCKETCHANNEL_H_

#include <uem_common.h>

#include <UCDynamicSocket.h>

#include <uem_data.h>
#include <uem_tcp_data.h>

#ifdef __cplusplus
extern "C"
{
#endif

uem_result UKTCPSocketChannel_Clear(SChannel *pstChannel);
uem_result UKTCPSocketChannel_Initialize(SChannel *pstChannel);
uem_result UKTCPSocketChannel_ReadFromQueue(SChannel *pstChannel, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead);
uem_result UKTCPSocketChannel_ReadFromBuffer(SChannel *pstChannel, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead);
uem_result UKTCPSocketChannel_GetAvailableChunk (SChannel *pstChannel, OUT int *pnChunkIndex);
uem_result UKTCPSocketChannel_GetNumOfAvailableData (SChannel *pstChannel, IN int nChunkIndex, OUT int *pnDataNum);
uem_result UKTCPSocketChannel_WriteToQueue (SChannel *pstChannel, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten);
uem_result UKTCPSocketChannel_WriteToBuffer (SChannel *pstChannel, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten);
uem_result UKTCPSocketChannel_SetExit(SChannel *pstChannel, int nExitFlag);
uem_result UKTCPSocketChannel_ClearExit(SChannel *pstChannel, int nExitFlag);
uem_result UKTCPSocketChannel_Finalize(SChannel *pstChannel);

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKTCPSOCKETCHANNEL_H_ */
