/*
 * UKSerialChannel.h
 *
 *  Created on: 2018. 10. 26.
 *      Author: jej
 */

#ifndef SRC_KERNEL_CONSTRAINED_INCLUDE_COMMUNICATION_UKSERIALCHANNEL_H_
#define SRC_KERNEL_CONSTRAINED_INCLUDE_COMMUNICATION_UKSERIALCHANNEL_H_

#include <uem_common.h>

#include <uem_channel_data.h>


#ifdef __cplusplus
extern "C"
{
#endif

uem_result UKSerialChannel_Clear(SChannel *pstChannel);
uem_result UKSerialChannel_Initialize(SChannel *pstChannel);
uem_result UKSerialChannel_ReadFromQueue(SChannel *pstChannel, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead);
uem_result UKSerialChannel_ReadFromBuffer(SChannel *pstChannel, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead);
uem_result UKSerialChannel_GetAvailableChunk (SChannel *pstChannel, OUT int *pnChunkIndex);
uem_result UKSerialChannel_GetNumOfAvailableData (SChannel *pstChannel, IN int nChunkIndex, OUT int *pnDataNum);
uem_result UKSerialChannel_WriteToQueue (SChannel *pstChannel, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten);
uem_result UKSerialChannel_FillInitialData(SChannel *pstChannel);
uem_result UKSerialChannel_WriteToBuffer (SChannel *pstChannel, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten);

uem_result UKSerialChannel_SetExit(SChannel *pstChannel, int nExitFlag);
uem_result UKSerialChannel_ClearExit(SChannel *pstChannel, int nExitFlag);
uem_result UKSerialChannel_Finalize(SChannel *pstChannel);


uem_result UKSerialChannel_HandleRequest(SChannel *pstChannel);
uem_result UKSerialChannel_PendReadQueueRequest(SChannel *pstChannel, int nDataToRead);
uem_result UKSerialChannel_PendReadBufferRequest(SChannel *pstChannel, int nDataToRead);
uem_result UKSerialChannel_PendGetAvailableDataRequest(SChannel *pstChannel);
uem_result UKSerialChannel_ClearRequest(SChannel *pstChannel);

#ifdef __cplusplus
}
#endif


#endif /* SRC_KERNEL_CONSTRAINED_INCLUDE_COMMUNICATION_UKSERIALCHANNEL_H_ */
