/*
 * UKBluetoothChannel.h
 *
 *  Created on: 2018. 10. 17.
 *      Author: chjej202
 */

#ifndef SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKBLUETOOTHCHANNEL_H_
#define SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKBLUETOOTHCHANNEL_H_

#include <uem_common.h>

#include <uem_channel_data.h>

#ifdef __cplusplus
extern "C"
{
#endif

uem_result UKBluetoothChannel_Clear(SChannel *pstChannel);
uem_result UKBluetoothChannel_Initialize(SChannel *pstChannel);
uem_result UKBluetoothChannel_ReadFromQueue(SChannel *pstChannel, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead);
uem_result UKBluetoothChannel_ReadFromBuffer(SChannel *pstChannel, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead);
uem_result UKBluetoothChannel_GetAvailableChunk (SChannel *pstChannel, OUT int *pnChunkIndex);
uem_result UKBluetoothChannel_GetNumOfAvailableData (SChannel *pstChannel, IN int nChunkIndex, OUT int *pnDataNum);
uem_result UKBluetoothChannel_WriteToQueue (SChannel *pstChannel, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten);
uem_result UKBluetoothChannel_FillInitialData(SChannel *pstChannel);
uem_result UKBluetoothChannel_WriteToBuffer (SChannel *pstChannel, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten);
uem_result UKBluetoothChannel_SetExit(SChannel *pstChannel, int nExitFlag);
uem_result UKBluetoothChannel_ClearExit(SChannel *pstChannel, int nExitFlag);
uem_result UKBluetoothChannel_Finalize(SChannel *pstChannel);

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKBLUETOOTHCHANNEL_H_ */
