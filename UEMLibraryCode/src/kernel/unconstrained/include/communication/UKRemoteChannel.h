/*
 * UKRemoteChannel.h
 *
 *  Created on: 2019. 5. 29.
 *      Author: jej
 */

#ifndef SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKREMOTECHANNEL_H_
#define SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKREMOTECHANNEL_H_

#include <uem_common.h>

#include <uem_channel_data.h>

#ifdef __cplusplus
extern "C"
{
#endif

uem_result UKRemoteChannel_Clear(SChannel *pstChannel);
uem_result UKRemoteChannel_Initialize(SChannel *pstChannel);
uem_result UKRemoteChannel_ReadFromQueue(SChannel *pstChannel, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead);
uem_result UKRemoteChannel_ReadFromBuffer(SChannel *pstChannel, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead);
uem_result UKRemoteChannel_GetAvailableChunk (SChannel *pstChannel, OUT int *pnChunkIndex);
uem_result UKRemoteChannel_GetNumOfAvailableData (SChannel *pstChannel, IN int nChunkIndex, OUT int *pnDataNum);
uem_result UKRemoteChannel_WriteToQueue (SChannel *pstChannel, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten);
uem_result UKRemoteChannel_FillInitialData(SChannel *pstChannel);
uem_result UKRemoteChannel_WriteToBuffer (SChannel *pstChannel, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten);
uem_result UKRemoteChannel_SetExit(SChannel *pstChannel, int nExitFlag);
uem_result UKRemoteChannel_ClearExit(SChannel *pstChannel, int nExitFlag);
uem_result UKRemoteChannel_Finalize(SChannel *pstChannel);

uem_result UKRemoteChannel_APIInitialize();
uem_result UKRemoteChannel_APIFinalize();

extern int g_nRemoteCommunicationModuleNum;
extern FnChannelAPIInitialize g_aFnRemoteCommunicationModuleIntializeList[];
extern FnChannelAPIFinalize g_aFnRemoteCommunicationModuleFinalizeList[];

#ifdef __cplusplus
}
#endif


#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKREMOTECHANNEL_H_ */
