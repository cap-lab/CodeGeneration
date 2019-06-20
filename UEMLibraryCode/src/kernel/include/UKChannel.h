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
 * @brief Initialize a channel.
 *
 * This function performs channel initialization. \n
 * This function at first initialize socket APIs if exists, \n
 * then initialize communication modules if exists, \n
 * and initialize each channel with matching channelAPI initialization function.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * error could be propagated from  ChannelAPI_GetAPIStructureFromCommunicationType.
 */
uem_result UKChannel_Initialize();

/**
 * @brief Get channel id by task and port name.
 *
 * This function returns channel id by given task name and port name.
 *
 * @param nTaskId task id.
 * @param szPortName port name.
 *
 * @return return channel index. \n
 * INVALID_CHANNEL_ID (== -1) returned for invalid @a nTaskId or @a szPortName.
 */
int UKChannel_GetChannelIdByTaskAndPortName(int nTaskId, char *szPortName);

/**
 * @brief Send data to a specific chunk index on channel whose type is buffer.
 *
 * This function sends data to a specific chunk index on channel whose type is buffer. \n
 * This function calls the writeToBuffer function specified for each channelAPI allocated for each channel. \n
 * When a single port is connected to multiple channels, it calls all corresponding channelAPIs.
 *
 * @param nChannelId ID of channel to send data.
 * @param pBuffer buffer to send data.
 * @param nDataToWrite buffer size.
 * @param nChunkIndex The index of a particular chunk (not used).
 * @param[out] pnDataWritten sent data size.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_INVALID_PARAM for invalid @a nChannelId, @a pBuffer, @a nChunkIndex. \n
 * @ref ERR_UEM_ILLEGAL_CONTROL if ChannelAPI found by ChannelAPI_GetAPIStructureFromCommunicationType function has invalid writeToBuffer function. \n
 * error could be propagated from fnWriteToBuffer function of channelAPI or from ChannelAPI_GetAPIStructureFromCommunicationType.
 *
 */
uem_result UKChannel_WriteToBuffer(int nChannelId, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten);

/**
 * @brief Send data to a specific chunk index on channel whose type is queue.
 *
 * This function sends data to a specific chunk index on channel whose type is queue. \n
 * This function calls the writeToQueue function specified for each channelAPI allocated for each channel. \n
 * When a single port is connected to multiple channels, it calls all corresponding channelAPIs.
 *
 * @param nChannelId ID of channel to send data.
 * @param pBuffer buffer to send data.
 * @param nDataToWrite buffer size.
 * @param nChunkIndex The index of a particular chunk.
 * @param[out] pnDataWritten sent data size.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_INVALID_PARAM for invalid @a nChannelId, @a pBuffer, @a nChunkIndex. \n
 * @ref ERR_UEM_ILLEGAL_CONTROL if ChannelAPI found by ChannelAPI_GetAPIStructureFromCommunicationType function has invalid writeToQueue function.
 * error could be propagated from fnWriteToQueue function of channelAPI or from ChannelAPI_GetAPIStructureFromCommunicationType.
 */
uem_result UKChannel_WriteToQueue(int nChannelId, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten);

/**
 * @brief Receive data from a specific chunk index on a port whose type is queue.
 *
 * This function receives data to a specific chunk index on channel whose type is queue. \n
 * This function calls the readFromqueue function specified for each channelAPI allocated for each channel. \n
 *
 * @param nChannelId ID of channel to receive data.
 * @param pBuffer buffer to receive data.
 * @param nDataToRead buffer size.
 * @param nChunkIndex The index of a particular chunk.
 * @param[out] pnDataRead received data size.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_INVALID_PARAM for invalid @a nChannelId, @a pBuffer, @a nChunkIndex. \n
 * @ref ERR_UEM_ILLEGAL_CONTROL if ChannelAPI found by ChannelAPI_GetAPIStructureFromCommunicationType function has invalid readFromqueue function. \n
  * error could be propagated from fnReadFromqueue function of channelAPI or from ChannelAPI_GetAPIStructureFromCommunicationType.
 */
uem_result UKChannel_ReadFromQueue(int nChannelId, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead);

/**
 * @brief Receive data from a specific chunk index on a port whose type is buffer.
 *
 * This function receives data to a specific chunk index on channel whose type is buffer. \n
 * This function calls the readFromBuffer function specified for each channelAPI allocated for each channel. \n
 *
 * @param nChannelId ID of channel to receive data.
 * @param pBuffer buffer to receive data.
 * @param nDataToRead buffer size.
 * @param nChunkIndex The index of a particular chunk (not used).
 * @param[out] pnDataRead received data size.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_INVALID_PARAM for invalid @a nChannelId, @a pBuffer, @a nChunkIndex. \n
 * @ref ERR_UEM_ILLEGAL_CONTROL if ChannelAPI found by ChannelAPI_GetAPIStructureFromCommunicationType function has invalid readFromBuffer function. \n
 * error could be propagated from fnReadFromBuffer function of channelAPI or from ChannelAPI_GetAPIStructureFromCommunicationType.
 */
uem_result UKChannel_ReadFromBuffer(int nChannelId, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead);

/**
 * @brief Return the number of available data on the channel with chunk index.
 *
 * This function returns the number of available data that can be fetched from the specified channel with chunk index.
 * This function calls the getNumOfAvailableData function specified for each channelAPI allocated for each channel. \n
 *
 * @param nChannelId ID of channel to check available data.
 * @param nChunkIndex The index of a particular chunk.
 * @param[out] pnDataNum data size that could be received.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_INVALID_PARAM for invalid @a nChannelId, @a pBuffer, @a nChunkIndex. \n
 * @ref ERR_UEM_ILLEGAL_CONTROL if ChannelAPI found by ChannelAPI_GetAPIStructureFromCommunicationType function has invalid getNumOfAvailableData function. \n
 * error could be propagated from fnGetNumOfAvailableData function of channelAPI or from ChannelAPI_GetAPIStructureFromCommunicationType.
 */
uem_result UKChannel_GetNumOfAvailableData (IN int nChannelId, IN int nChunkIndex, OUT int *pnDataNum);

/**
 * @brief Return chunk index that can receive the current data on the channel.
 *
 * This function returns chunk index that can receive the current data on the channel.
 * This function calls the getAvailableChunk function specified for each channelAPI allocated for each channel. \n
 *
 * @param nChannelId ID of channel to check available data.
 * @param[out] pnChunkIndex Index of a specific chunk having data to be received.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_INVALID_PARAM for invalid @a nChannelId, @a nChunkIndex. \n
 * @ref ERR_UEM_ILLEGAL_CONTROL if ChannelAPI found by ChannelAPI_GetAPIStructureFromCommunicationType function has invalid getAvailableChunk function. \n
 * error could be propagated from fnGetAvailableChunk function of channelAPI or from ChannelAPI_GetAPIStructureFromCommunicationType.
 */
uem_result UKChannel_GetAvailableIndex (IN int nChannelId, OUT int *pnChunkIndex);

/**
 * @brief Perform channel clearing.
 *
 * This function performs channel clearing jobs.
 * This function calls the clear function specified for each channelAPI allocated for each channel. \n
 *
 * @param nChannelId ID of channel to be cleared.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_INVALID_PARAM for invalid @a nChannelId. \n
 * @ref ERR_UEM_ILLEGAL_CONTROL if ChannelAPI found by ChannelAPI_GetAPIStructureFromCommunicationType function has invalid clear function. \n
 * error could be propagated from fnClear function of channelAPI or from ChannelAPI_GetAPIStructureFromCommunicationType.
 */
uem_result UKChannel_Clear(IN int nChannelId);

/**
 * @brief Return the size of the channel buffer.
 *
 * This function returns the size of the channel buffer.
 *
 * @param nChannelId ID of channel to check buffer size.
 * @param[out] pnChannelSize checked channel buffer size.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_INVALID_PARAM for invalid @a nChannelId and @a pnChannelSize. \n
 */
uem_result UKChannel_GetChannelSize(IN int nChannelId, OUT int *pnChannelSize);

/**
 * @brief Perform channel finalizing.
 * This function performs channel finalizing jobs.
 * This function calls the finalize function specified for each channelAPI allocated for each channel. \n
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * error could be propagated from fnFinalize function of channelAPI or from ChannelAPI_GetAPIStructureFromCommunicationType.
 */
uem_result UKChannel_Finalize();

// used inside kernel layer
/**
 * @brief Perform channel clearing inside hierarchical Task.
 *
 * This function performs channel clearing jobs.
 * This function calls clear functions For all channels in the hierarchy task. \n
 *
 * @param nParentTaskId id of hierarchical task to be cleared.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * error could be propagated from fnClear function of channelAPI or from ChannelAPI_GetAPIStructureFromCommunicationType.
 */
uem_result UKChannel_ClearChannelInSubgraph(int nParentTaskId);


#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_INCLUDE_UKCHANNEL_H_ */
