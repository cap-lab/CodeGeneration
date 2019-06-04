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

/**
 * @brief
 *
 * This function
 *
 * @param pstChannel
 *
 * @return
 */
uem_result UKSerialChannel_Clear(SChannel *pstChannel);

/**
 * @brief
 *
 * This function
 *
 * @param pstChannel
 *
 * @return
 */
uem_result UKSerialChannel_Initialize(SChannel *pstChannel);

/**
 * @brief
 *
 * This function
 *
 * @param pstChannel
 * @param pBuffer
 * @param nDataToRead
 * @param nChunkIndex
 * @param[out] pnDataRead
 *
 * @return
 */
uem_result UKSerialChannel_ReadFromQueue(SChannel *pstChannel, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead);

/**
 * @brief
 *
 * This function
 *
 * @param pstChannel
 * @param pBuffer
 * @param nDataToRead
 * @param nChunkIndex
 * @param[out] pnDataRead
 *
 * @return
 */
uem_result UKSerialChannel_ReadFromBuffer(SChannel *pstChannel, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead);

/**
 * @brief
 *
 * This function
 *
 * @param pstChannel
 * @param[out] pnChunkIndex
 *
 * @return
 */
uem_result UKSerialChannel_GetAvailableChunk (SChannel *pstChannel, OUT int *pnChunkIndex);

/**
 * @brief
 *
 * This function
 *
 * @param pstChannel
 * @param nChunkIndex
 * @param[out] pnDataNum
 *
 * @return
 */
uem_result UKSerialChannel_GetNumOfAvailableData (SChannel *pstChannel, IN int nChunkIndex, OUT int *pnDataNum);

/**
 * @brief
 *
 * This function
 *
 * @param pstChannel
 * @param pBuffer
 * @param nDataToWrite
 * @param nChunkIndex
 * @param[out] pnDataWritten
 *
 * @return
 */
uem_result UKSerialChannel_WriteToQueue (SChannel *pstChannel, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten);

/**
 * @brief
 *
 * This function
 *
 * @param pstChannel
 *
 * @return
 */
uem_result UKSerialChannel_FillInitialData(SChannel *pstChannel);

/**
 * @brief
 *
 * This function
 *
 * @param pstChannel
 * @param pBuffer
 * @param nDataToWrite
 * @param nChunkIndex
 * @param[out] pnDataWritten
 *
 * @return
 */
uem_result UKSerialChannel_WriteToBuffer (SChannel *pstChannel, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten);

/**
 * @brief
 *
 * This function
 *
 * @param pstChannel
 * @param nExitFlag
 *
 * @return
 */
uem_result UKSerialChannel_SetExit(SChannel *pstChannel, int nExitFlag);

/**
 * @brief
 *
 * This function
 *
 * @param pstChannel
 * @param nExitFlag
 *
 * @return
 */
uem_result UKSerialChannel_ClearExit(SChannel *pstChannel, int nExitFlag);

/**
 * @brief
 *
 * This function
 *
 * @param pstChannel
 *
 * @return
 */
uem_result UKSerialChannel_Finalize(SChannel *pstChannel);

/**
 * @brief
 *
 * This function
 *
 * @param pstChannel
 *
 * @return
 */
uem_result UKSerialChannel_HandleRequest(SChannel *pstChannel);

/**
 * @brief
 *
 * This function
 *
 * @param pstChannel
 * @param nDataToRead
 *
 * @return
 */
uem_result UKSerialChannel_PendReadQueueRequest(SChannel *pstChannel, int nDataToRead);

/**
 * @brief
 *
 * This function
 *
 * @param pstChannel
 * @param nDataToRead
 *
 * @return
 */
uem_result UKSerialChannel_PendReadBufferRequest(SChannel *pstChannel, int nDataToRead);

/**
 * @brief
 *
 * This function
 *
 * @param pstChannel
 *
 * @return
 */
uem_result UKSerialChannel_PendGetAvailableDataRequest(SChannel *pstChannel);

/**
 * @brief
 *
 * This function
 *
 * @param pstChannel
 *
 * @return
 */
uem_result UKSerialChannel_ClearRequest(SChannel *pstChannel);

#ifdef __cplusplus
}
#endif


#endif /* SRC_KERNEL_CONSTRAINED_INCLUDE_COMMUNICATION_UKSERIALCHANNEL_H_ */
