/*
 * UFPort.h
 *
 *  Created on: 2017. 8. 11.
 *      Author: jej
 */

#ifndef SRC_API_INCLUDE_UFPORT_H_
#define SRC_API_INCLUDE_UFPORT_H_


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
 * @param nTaskId
 * @param szPortName
 * @param pnChannelId
 *
 * @return
 */
uem_result UFPort_Initialize(IN int nTaskId, IN char *szPortName, OUT int *pnChannelId);

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
uem_result UFPort_ReadFromQueue (IN int nChannelId, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead);

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
uem_result UFPort_ReadFromBuffer (IN int nChannelId, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead);

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
uem_result UFPort_WriteToQueue (IN int nChannelId, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten);

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
uem_result UFPort_WriteToBuffer (IN int nChannelId, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten);

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
uem_result UFPort_GetNumOfAvailableData (IN int nChannelId, IN int nChunkIndex, OUT int *pnDataNum);

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
uem_result UFPort_GetAvailableIndex (IN int nChannelId, OUT int *pnChunkIndex);

/**
 * @brief
 *
 * This function
 *
 * @param nChannelId
 *
 * @return
 */
int UFPort_GetChannelSize(IN int nChannelId);

#ifdef __cplusplus
}
#endif

#endif /* SRC_API_INCLUDE_UFPORT_H_ */
