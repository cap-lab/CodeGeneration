/*
 * UKSharedMemoryChannel.h
 *
 *  Created on: 2017. 11. 9.
 *      Author: jej
 */

#ifndef SRC_KERNEL_INCLUDE_UKSHAREDMEMORYCHANNEL_H_
#define SRC_KERNEL_INCLUDE_UKSHAREDMEMORYCHANNEL_H_

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
 *
 * @return
 */
uem_result UKSharedMemoryChannel_Initialize(SChannel *pstChannel);

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
uem_result UKSharedMemoryChannel_ReadFromQueue(SChannel *pstChannel, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead);

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
uem_result UKSharedMemoryChannel_ReadFromBuffer(SChannel *pstChannel, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead);

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
uem_result UKSharedMemoryChannel_WriteToBuffer (SChannel *pstChannel, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten);

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
uem_result UKSharedMemoryChannel_WriteToQueue (SChannel *pstChannel, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten);

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
uem_result UKSharedMemoryChannel_GetAvailableChunk (SChannel *pstChannel, OUT int *pnChunkIndex);

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
uem_result UKSharedMemoryChannel_GetNumOfAvailableData (SChannel *pstChannel, IN int nChunkIndex, OUT int *pnDataNum);

/**
 * @brief
 *
 * This function
 *
 * @param pstChannel
 *
 * @return
 */
uem_result UKSharedMemoryChannel_Clear(SChannel *pstChannel);

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
uem_result UKSharedMemoryChannel_SetExit(SChannel *pstChannel, int nExitFlag);

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
uem_result UKSharedMemoryChannel_ClearExit(SChannel *pstChannel, int nExitFlag);

/**
 * @brief
 *
 * This function
 *
 * @param pstChannel
 *
 * @return
 */
uem_result UKSharedMemoryChannel_FillInitialData(SChannel *pstChannel);

/**
 * @brief
 *
 * This function
 *
 * @param pstChannel
 *
 * @return
 */
uem_result UKSharedMemoryChannel_Finalize(SChannel *pstChannel);

#ifdef __cplusplus
}
#endif




#endif /* SRC_KERNEL_INCLUDE_UKSHAREDMEMORYCHANNEL_H_ */
