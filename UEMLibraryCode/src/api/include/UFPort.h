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
 * @brief Return channel ID corresponding to the task and port name.
 *
 * This function is called by the TASK_INIT function where the first argument should be TASK_ID macro. TASK_ID means the ID of the current task.
 *
 *
 * @param nTaskId id of task.
 * @param szPortName port name.
 * @param pnChannelId returned channel id. -1 for invalid channel id.
 *
 * @return
 *
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_INVALID_PARAM for invalid szPortName, pnChannelId arguments.
 */
uem_result UFPort_Initialize(IN int nTaskId, IN char *szPortName, OUT int *pnChannelId);

/**
 * @brief Receive data from a specific chunk index on a port whose type is queue.
 *
 * blocked if no data in that chunk index exists.
 *
 * @param nChannelId ID of channel to receive data
 * @param pBuffer buffer to receive data
 * @param nDataToRead buffer size
 * @param nChunkIndex The index of a particular chunk
 * @param[out] pnDataRead size of received data
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_INVALID_PARAM for invalid channel id, nDataToRead, pBuffer, pnDataRead, nChunkIndex arguments. \n
 * @ref ERR_UEM_ILLEGAL_CONTROL if Channel API does not have readFromQueue handler.
 *
 * \n
 *  Functions that may propagate error results : \n
 *  ChannelAPI_GetAPIStructureFromCommunicationType \n
 *  readfromQueue handler generated by Translator \n
 *  \n
 */
uem_result UFPort_ReadFromQueue (IN int nChannelId, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead);

/**
 * @brief Receive data from a specific chunk index on a port whose type is buffer.
 *
 * blocked if no data in that chunk index exists.
 *
 * @param nChannelId ID of channel to receive data.
 * @param pBuffer buffer to receive data.
 * @param nDataToRead buffer size.
 * @param nChunkIndex The index of a particular chunk.
 * @param[out] pnDataRead size of received data.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_INVALID_PARAM for invalid channel id, nDataToRead, pBuffer, pnDataRead, nChunkIndex arguments. \n
 * @ref ERR_UEM_ILLEGAL_CONTROL if Channel API does not have readFromBuffer handler. \n
 *
 * \n
 *  Functions that may propagate error results : \n
 *  ChannelAPI_GetAPIStructureFromCommunicationType \n
 *  ReadFromBuffer handler generated by Translator \n
 *  \n
 */

uem_result UFPort_ReadFromBuffer (IN int nChannelId, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead);

/**
 * @brief Send data to a specific chunk index on a port whose type is queue.
 * blocked If the data in that chunk index is not read yet.
 *
 * @param nChannelId ID of channel to send data.
 * @param pBuffer buffer to send data.
 * @param nDataToWrite buffer size.
 * @param nChunkIndex The index of a particular chunk.
 * @param[out] pnDataWritten sent data size.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_INVALID_PARAM for invalid  channel id, nDataToWrite, pBuffer, pnDataWritten, nChunkIndex arguments. \n
 * @ref ERR_UEM_ILLEGAL_CONTROL if Channel API does not have writeToQueue handler. \n
 *
 *  \n
 *  Functions that may propagate error results \n
 *  ChannelAPI_GetAPIStructureFromCommunicationType \n
 *  writeToQueue handler generated by Translator \n
 *  \n
 */
uem_result UFPort_WriteToQueue (IN int nChannelId, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten);

/**
 * @brief Send data to a specific chunk index on a port whose type is buffer.
 *
 * @param nChannelId ID of channel to send data.
 * @param pBuffer buffer to send data.
 * @param nDataToWrite buffer size.
 * @param nChunkIndex The index of a particular chunk.
 * @param[out] pnDataWritten sent data size.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_INVALID_PARAM for invalid channel id, nDataToWrite, pBuffer, pnDataWritten, nChunkIndex arguments. \n
 * @ref ERR_UEM_ILLEGAL_CONTROL if Channel API does not have writeToBuffer handler .\n
 *
 *  \n
 *  Functions that may propagate error results : \n
 *  ChannelAPI_GetAPIStructureFromCommunicationType \n
 *  writeToBuffer handler generated by Translator \n
 *  \n
 */
uem_result UFPort_WriteToBuffer (IN int nChannelId, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten);

/**
 * @brief Return the number of available data on the channel with chunk index.
 *
 * @param nChannelId ID of channel to check available data.
 * @param nChunkIndex The index of a particular chunk.
 * @param[out] pnDataNum data size that could be received.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_INVALID_PARAM for invalid channel id, pnDataNum, nChunkIndex argument. \n
 * @ref ERR_UEM_ILLEGAL_CONTROL if Channel API does not have getNumOfAvailableData handler. \n
 *
 *  \n
 *  Functions that may propagate error results : \n
 *  ChannelAPI_GetAPIStructureFromCommunicationType, \n
 *  getNumOfAvailableData handler generated by Translator. \n
 *  \n
 */
uem_result UFPort_GetNumOfAvailableData (IN int nChannelId, IN int nChunkIndex, OUT int *pnDataNum);

/**
 * @brief Return chunk index that can receive the current data on the port.
 *
 * If no data is available for all chunk index, blocked until new data is received.
 *
 * @param nChannelId ID of channel to check available data.
 * @param[out] pnChunkIndex Index of a specific chunk having data to be received.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_INVALID_PARAM for invalid channel id, pnChunkIndex argument. \n
 * @ref ERR_UEM_ILLEGAL_CONTROL if Channel API does not have getAvailableChunk handler. \n
 *
 * \n
 *  Functions that may propagate error results : \n
 * ChannelAPI_GetAPIStructureFromCommunicationType, \n
 *  getAvailableChunk handler generated by Translator. \n
 *  \n
 */
uem_result UFPort_GetAvailableIndex (IN int nChannelId, OUT int *pnChunkIndex);

/**
 * @brief Return the size of the channel buffer associated with this port.
 *
 * @param nChannelId ID of channel to check buffer size.
 *
 * @return
 * channel buffer size. if channel not exists,  return zero.
 *
 */
int UFPort_GetChannelSize(IN int nChannelId);

#ifdef __cplusplus
}
#endif

#endif /* SRC_API_INCLUDE_UFPORT_H_ */
