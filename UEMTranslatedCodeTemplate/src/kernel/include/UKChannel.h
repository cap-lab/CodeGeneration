/*
 * UKChannel.h
 *
 *  Created on: 2017. 8. 28.
 *      Author: jej
 */

#ifndef SRC_KERNEL_INCLUDE_UKCHANNEL_H_
#define SRC_KERNEL_INCLUDE_UKCHANNEL_H_

#include <uem_common.h>

#ifdef __cplusplus
extern "C"
{
#endif

uem_result UKChannel_Initialize();
int UKChannel_GetChannelIdByTaskAndPortName(int nTaskId, char *szPortName);
uem_result UKChannel_WriteToBuffer(int nChannelId, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten);
uem_result UKChannel_WriteToQueue(int nChannelId, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten);
uem_result UKChannel_ReadFromQueue(int nChannelId, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead);
uem_result UKChannel_ReadFromBuffer(int nChannelId, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead);
uem_result UKChannel_GetNumOfAvailableData (IN int nChannelId, IN int nChunkIndex, OUT int *pnDataNum);
uem_result UKChannel_GetAvailableIndex (IN int nChannelId, OUT int *pnChunkIndex);
uem_result UKChannel_Clear(IN int nChannelId);
uem_result UKChannel_SetExit();
uem_result UKChannel_Finalize();


#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_INCLUDE_UKCHANNEL_H_ */
