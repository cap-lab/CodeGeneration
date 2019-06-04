/*
 * UKChannelMemory.h
 *
 *  Created on: 2018. 5. 26.
 *      Author: jej
 */

#ifndef SRC_KERNEL_INCLUDE_UKCHANNELMEMORY_H_
#define SRC_KERNEL_INCLUDE_UKCHANNELMEMORY_H_

#include <uem_common.h>

#include <uem_data.h>

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
 * @param pstSharedMemoryChannel
 *
 * @return
 */
uem_result UKChannelMemory_Initialize(SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel);

/**
 * @brief
 *
 * This function
 *
 * @param pstChannel
 * @param pstSharedMemoryChannel
 * @param pBuffer
 * @param nDataToRead
 * @param nChunkIndex
 * @param[out] pnDataRead
 *
 * @return
 */
uem_result UKChannelMemory_ReadFromQueue(SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead);

/**
 * @brief
 *
 * This function
 *
 * @param pstChannel
 * @param pstSharedMemoryChannel
 * @param pBuffer
 * @param nDataToRead
 * @param nChunkIndex
 * @param[out] pnDataRead
 *
 * @return
 */
uem_result UKChannelMemory_ReadFromBuffer(SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead);

/**
 * @brief
 *
 * This function
 *
 * @param pstChannel
 * @param pstSharedMemoryChannel
 * @param pBuffer
 * @param nDataToWrite
 * @param nChunkIndex
 * @param[out] pnDataWritten
 *
 * @return
 */
uem_result UKChannelMemory_WriteToBuffer (SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten);

/**
 * @brief
 *
 * This function
 *
 * @param pstChannel
 * @param pstSharedMemoryChannel
 * @param pBuffer
 * @param nDataToWrite
 * @param nChunkIndex
 * @param[out] pnDataWritten
 *
 * @return
 */
uem_result UKChannelMemory_WriteToQueue (SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten);

/**
 * @brief
 *
 * This function
 *
 * @param pstChannel
 * @param pstSharedMemoryChannel
 * @param[out] pnChunkIndex
 *
 * @return
 */
uem_result UKChannelMemory_GetAvailableChunk (SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel, OUT int *pnChunkIndex);

/**
 * @brief
 *
 * This function
 *
 * @param pstChannel
 * @param pstSharedMemoryChannel
 * @param nChunkIndex
 * @param[out] pnDataNum
 *
 * @return
 */
uem_result UKChannelMemory_GetNumOfAvailableData (SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel, IN int nChunkIndex, OUT int *pnDataNum);

/**
 * @brief
 *
 * This function
 *
 * @param pstChannel
 * @param pstSharedMemoryChannel
 *
 * @return
 */
uem_result UKChannelMemory_Clear(SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel);

/**
 * @brief
 *
 * This function
 *
 * @param pstChannel
 * @param pstSharedMemoryChannel
 * @param nExitFlag
 *
 * @return
 */
uem_result UKChannelMemory_SetExit(SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel, int nExitFlag);

/**
 * @brief
 *
 * This function
 *
 * @param pstChannel
 * @param pstSharedMemoryChannel
 * @param nExitFlag
 *
 * @return
 */
uem_result UKChannelMemory_ClearExit(SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel, int nExitFlag);

/**
 * @brief
 *
 * This function
 *
 * @param pstChannel
 * @param pstSharedMemoryChannel
 *
 * @return
 */
uem_result UKChannelMemory_FillInitialData(SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel);

/**
 * @brief
 *
 * This function
 *
 * @param pstChannel
 * @param pstSharedMemoryChannel
 *
 * @return
 */
uem_result UKChannelMemory_Finalize(SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel);

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_INCLUDE_UKCHANNELMEMORY_H_ */
