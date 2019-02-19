/*
 * UKSerialChannel.h
 *
 Created on: 2019. 02. 18., modified from UKBluetoothChannel.h
 *      Author: dowhan1128
 */

#ifndef SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKSERIALCHANNEL_H_
#define SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKSERIALCHANNEL_H_

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

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKSERIALCHANNEL_H_ */
