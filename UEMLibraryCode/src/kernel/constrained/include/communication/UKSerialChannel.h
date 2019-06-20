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
 * @brief Clear internal serial channel data.
 *
 * This function clears internal serial channel data.
 * This function is a part of @ref SChannelAPI.
 *
 * @param pstChannel a single channel structure.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_ILLEGAL_DATA.
 */
uem_result UKSerialChannel_Clear(SChannel *pstChannel);

/**
 * @brief Initialize a serial channel.
 *
 * This function initializes a serial channel. \n
 * this function initializes internal channel and sets channel to SerialInfo channelList.
 * This function is a part of @ref SChannelAPI.
 * *
 * @param pstChannel a single channel structure.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_ILLEGAL_DATA.
 *
 */
uem_result UKSerialChannel_Initialize(SChannel *pstChannel);

/**
 * @brief Read data from a serial channel as a queue.
 *
 * This function reads data from a serial channel. \n
 * Because it reads from queue, data in the channel is removed after reading the data. \n
 * This function reads data from internal channel when data is available, or adds pending request message. \n
 * This function is a part of @ref SChannelAPI.
 *
 * @param pstChannel a single channel structure.
 * @param[in,out] pBuffer buffer to store read data.
 * @param nDataToRead size of data to read.
 * @param nChunkIndex chunk index to read (only applicable for individual connection).
 * @param[out] pnDataRead size of data read.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_READ_BLOCK if internal channel data is not enough to read. \n
 * Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKSerialChannel_ReadFromQueue(SChannel *pstChannel, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead);

/**
 * @brief Read data from a serial channel as a buffer.
 *
 * This function reads data from a serial channel. \n
 * Because it reads from buffer, data in the channel is preserved after reading the data. \n
 * This function reads data from internal channel when data is available, or adds pending request message. \n
 * This function is a part of @ref SChannelAPI.
 *
 * @param pstChannel a single channel structure.
 * @param[in,out] pBuffer buffer to store read data.
 * @param nDataToRead size of data to read.
 * @param nChunkIndex (not used).
 * @param[out] pnDataRead size of data read.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKSerialChannel_ReadFromBuffer(SChannel *pstChannel, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead);

/**
 * @brief (not used) Get readable chunk index from queue.
 *
 * (not used) This function gets readable chunk index from queue.
 * currently it just returns 0.
 *
 * @param pstChannel a single channel structure.
 * @param[out] pnChunkIndex an available chunk index to be read.
 *
 * @return This function always returns @ref ERR_UEM_NOERROR.
 */
uem_result UKSerialChannel_GetAvailableChunk (SChannel *pstChannel, OUT int *pnChunkIndex);

/**
 * @brief Get number of data in serial channel.
 *
 * This function retrieves the number of data in channel in specific chunk index. \n
 * This function gets number of available data from internal channel, and adds pending request message of type @ref MESSAGE_TYPE_AVAILABLE_DATA. \n
 * This function is a part of @ref SChannelAPI.
 *
 * @param pstChannel a single channel structure.
 * @param nChunkIndex (not used).
 * @param[out] pnDataNum amount of data in the channel.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_ILLEGAL_DATA.
 */
uem_result UKSerialChannel_GetNumOfAvailableData (SChannel *pstChannel, IN int nChunkIndex, OUT int *pnDataNum);

/**
 * @brief write data to serial channel as a queue.
 *
 * This function writes data to the channel as a queue. \n
 * Because it writes to queue, the write block happened if the channel data is full. \n
 * This function is same to shared memory channel write @ref UKSharedMemoryChannel_WriteToQueue. \n
 * This function is a part of @ref SChannelAPI.
 *
 * @param pstChannel a single channel structure.
 * @param pBuffer data to write.
 * @param nDataToWrite amount of data to write.
 * @param nChunkIndex chunk index to write.
 * @param[out] pnDataWritten amount of data written.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_WRITE_BLOCK if remaining channel buffer size is not enough to write. \n
 * Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKSerialChannel_WriteToQueue (SChannel *pstChannel, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten);

/**
 * @brief Fill initial data in channel.
 *
 * This function fills initial data in channel.
 *
 * @param pstChannel a single channel structure.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * Errors to be returned - @ref ERR_UEM_INVALID_PARAM. \n
 */
uem_result UKSerialChannel_FillInitialData(SChannel *pstChannel);

/**
 * @brief Write data to shared memory channel as a queue.
 *
 * This function writes data to the channel as a queue. \n
 * Because it writes to queue, the write block happened if the channel data is full. \n
 * This function is same to shared memory channel write @ref UKSharedMemoryChannel_WriteToBuffer. \n
 * This function is a part of @ref SChannelAPI.
 *
 * @param pstChannel a single channel structure.
 * @param pBuffer data to write.
 * @param nDataToWrite amount of data to write.
 * @param nChunkIndex chunk index to write.
 * @param[out] pnDataWritten amount of data written.
 *
 *  @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKSerialChannel_WriteToBuffer (SChannel *pstChannel, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten);

/**
 * @brief (not used) Exit block to terminate channel.
 *
 * (not used) This function set exit flags to escape from blocking state of the channel. \n
 * currently this function does nothing.
 *
 * @param pstChannel a single channel structure.
 * @param nExitFlag an exit flag.
 *
 * @return This function always returns @ref ERR_UEM_NOERROR.
 */
uem_result UKSerialChannel_SetExit(SChannel *pstChannel, int nExitFlag);

/**
 * @brief (not used) Clear exit flag to restart channel.
 *
 * (not used) This function clears exit flags to reuse channel. \n
 * currently this function does nothing.
 *
 * @param pstChannel a single channel structure.
 * @param nExitFlag an exit flag to clear.
 *
 * @return This function always returns @ref ERR_UEM_NOERROR.
 */
uem_result UKSerialChannel_ClearExit(SChannel *pstChannel, int nExitFlag);

/**
 * @brief (not used) Finalize a serial channel.
 *
 * This function finalizes a serial channel. \n
 * currently this function does nothing.
 *
 * @param pstChannel a single channel structure.
 *
 * @return This function always returns @ref ERR_UEM_NOERROR.
 */
uem_result UKSerialChannel_Finalize(SChannel *pstChannel);

/**
 * @brief Handle request to be sent.
 *
 * This function handles request to be sent. \n
 * Requesting message type is different depending on channel communication type. \n
 * Data request is sent when channel type is @ref COMMUNICATION_TYPE_REMOTE_READER, \n
 * Result is sent when channel type is @ref COMMUNICATION_TYPE_REMOTE_WRITER.
 *
 * @param pstChannel a single channel structure.
 *
 *  @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *  @ref ERR_UEM_NOT_REACHED_YET if result is not made yet for ReadFromQueue request
 *  (when communication type is @ref COMMUNICATION_TYPE_REMOTE_WRITER). \n
 * Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_ILLEGAL_DATA.
 */
uem_result UKSerialChannel_HandleRequest(SChannel *pstChannel);

/**
  * @brief Make pending read queue request.
 *
 * This function makes pending read queue request. \n
 *
 * @param pstChannel a single channel structure.
 * @param nDataToRead size of data to read.
 *
 * @return This function always returns @ref ERR_UEM_NOERROR.
 */
uem_result UKSerialChannel_PendReadQueueRequest(SChannel *pstChannel, int nDataToRead);

/**
 * @brief Make pending read buffer request.
 *
 * This function makes pending read buffer request. \n
 *
 * @param pstChannel a single channel structure.
 * @param nDataToRead size of data to read.
 *
 * @return This function always returns @ref ERR_UEM_NOERROR.
 */
uem_result UKSerialChannel_PendReadBufferRequest(SChannel *pstChannel, int nDataToRead);

/**
 * @brief Make pending get available data number request.
 *
 * This function makes pending get available data number request.
 *
 * @param pstChannel a single channel structure.
 *
 * @return This function always returns @ref ERR_UEM_NOERROR.
 */
uem_result UKSerialChannel_PendGetAvailableDataRequest(SChannel *pstChannel);

/**
 * @brief Clear current pending request.
 *
 * This function clears current pending request.
 *
 * @param pstChannel a single channel structure.
 *
 * @return This function always returns @ref ERR_UEM_NOERROR.
 */
uem_result UKSerialChannel_ClearRequest(SChannel *pstChannel);

#ifdef __cplusplus
}
#endif


#endif /* SRC_KERNEL_CONSTRAINED_INCLUDE_COMMUNICATION_UKSERIALCHANNEL_H_ */
