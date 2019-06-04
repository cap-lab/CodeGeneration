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

/**
 * @brief
 *
 * This function
 *
 * @return
 */
uem_result UKChannel_Initialize();

/**
 * @brief
 *
 * This function
 *
 * @param nTaskId
 * @param szPortName
 *
 * @return
 */
int UKChannel_GetChannelIdByTaskAndPortName(int nTaskId, char *szPortName);

/**
 * @brief
 *
 * This function
 *
 * @param nChannelId
 * @param pBuffer
 * @param nDataToWrite
 * @param nChunkIndex
 * @param[out] pnDataWritten
 *
 * @return
 */
uem_result UKChannel_WriteToBuffer(int nChannelId, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten);

/**
 * @brief
 *
 * This function
 *
 * @param nChannelId
 * @param pBuffer
 * @param nDataToWrite
 * @param nChunkIndex
 * @param[out] pnDataWritten
 *
 * @return
 */
uem_result UKChannel_WriteToQueue(int nChannelId, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten);

/**
 * @brief
 *
 * This function
 *
 * @param nChannelId
 * @param pBuffer
 * @param nDataToRead
 * @param nChunkIndex
 * @param[out] pnDataRead
 *
 * @return
 */
uem_result UKChannel_ReadFromQueue(int nChannelId, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead);

/**
 * @brief
 *
 * This function
 *
 * @param nChannelId
 * @param pBuffer
 * @param nDataToRead
 * @param nChunkIndex
 * @param[out] pnDataRead
 *
 * @return
 */
uem_result UKChannel_ReadFromBuffer(int nChannelId, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead);

/**
 * @brief
 *
 * This function
 *
 * @param nChannelId
 * @param nChunkIndex
 * @param[out] pnDataNum
 *
 * @return
 */
uem_result UKChannel_GetNumOfAvailableData (IN int nChannelId, IN int nChunkIndex, OUT int *pnDataNum);

/**
 * @brief
 *
 * This function
 *
 * @param nChannelId
 * @param[out] pnChunkIndex
 *
 * @return
 */
uem_result UKChannel_GetAvailableIndex (IN int nChannelId, OUT int *pnChunkIndex);

/**
 * @brief
 *
 * This function
 *
 * @param nChannelId
 *
 * @return
 */
uem_result UKChannel_Clear(IN int nChannelId);

/**
 * @brief
 *
 * This function
 *
 * @param nChannelId
 * @param[out] pnChannelSize
 *
 * @return
 */
uem_result UKChannel_GetChannelSize(IN int nChannelId, OUT int *pnChannelSize);

/**
 * @brief
 *
 * This function
 *
 * @return
 */
uem_result UKChannel_Finalize();

// used inside kernel layer
/**
 * @brief
 *
 * This function
 *
 * @param nParentTaskId
 *
 * @return
 */
uem_result UKChannel_ClearChannelInSubgraph(int nParentTaskId);


#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_INCLUDE_UKCHANNEL_H_ */
