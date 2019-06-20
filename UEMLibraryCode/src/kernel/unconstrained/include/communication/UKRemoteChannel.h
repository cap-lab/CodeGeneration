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

/**
 * @brief Clear internal remote channel data.
 *
 * This function clears internal remote channel data. \n
 * This function is a part of @ref SChannelAPI.
 *
 * @param pstChannel a single channel structure.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_ILLEGAL_DATA.
 *         Errors can be occurred when the channel structure data are corrupted.
 */
uem_result UKRemoteChannel_Clear(SChannel *pstChannel);

/**
 * @brief Initialize a remote channel.
 *
 * This function initializes a remote channel. \n
 * If the channel is a writer, it initializes internal shared memory channel (UKChannelMemory). \n
 * Depending on the connection methods (aggregate or individual), this function connects to aggregate service (UKSerialCommunicationManager) \n
 * or directly connects to remote devices. \n
 * This function is a part of @ref SChannelAPI.
 *
 * @param pstChannel a single channel structure.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_INTERNAL_FAIL, @ref ERR_UEM_MUTEX_ERROR, \n
 *         @ref ERR_UEM_NET_SEND_ERROR, @ref ERR_UEM_NET_TIMEOUT, @ref ERR_UEM_ILLEGAL_DATA, @ref ERR_UEM_ILLEGAL_CONTROL, \n
 *         @ref ERR_UEM_SUSPEND, and corresponding errors from fnCreate(), fnConnect(), fnReceive(), fnSend() of SVirtualCommunicationAPI.
 *
 * @sa UKRemoteChannel_Finalize.
 */
uem_result UKRemoteChannel_Initialize(SChannel *pstChannel);

/**
 * @brief Read data from a remote channel as a queue.
 *
 * This function reads data from a remote channel. \n
 * Because it reads from queue, data in the channel is removed after reading the data. \n
 * For aggregate connection, chunk index value is not used. \n
 * This function is a part of @ref SChannelAPI.
 *
 * @param pstChannel a single channel structure.
 * @param[in,out] pBuffer buffer to store read data.
 * @param nDataToRead size of data to read.
 * @param nChunkIndex chunk index to read (only applicable for individual connection).
 * @param[out] pnDataRead size of data read.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_SUSPEND, @ref ERR_UEM_INTERNAL_FAIL, \n
 *         @ref ERR_UEM_ILLEGAL_DATA, @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_TIME_EXPIRED, and \n
 *         internal virtual functions' results. \n
 *         @ref ERR_UEM_ILLEGAL_DATA is happened when the received result is invalid. \n
 *         @ref ERR_UEM_TIME_EXPIRED can be happened if the request sending time takes too long. \n
 *         @ref ERR_UEM_SUSPEND is occurred when the program is going to exit.
 */
uem_result UKRemoteChannel_ReadFromQueue(SChannel *pstChannel, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead);

/**
 * @brief Read data from a remote channel as a buffer.
 *
 * This function reads data from a remote channel. \n
 * Because it reads from buffer, data in the channel is preserved after reading the data. \n
 * This function is a part of @ref SChannelAPI.
 *
 * @param pstChannel a single channel structure.
 * @param[in,out] pBuffer buffer to store read data.
 * @param nDataToRead size of data to read.
 * @param nChunkIndex (not used).
 * @param[out] pnDataRead size of data read.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_SUSPEND, @ref ERR_UEM_INTERNAL_FAIL, \n
 *         @ref ERR_UEM_ILLEGAL_DATA, @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_TIME_EXPIRED, and \n
 *         internal virtual functions' results. \n
 *         @ref ERR_UEM_ILLEGAL_DATA is happened when the received result is invalid. \n
 *         @ref ERR_UEM_TIME_EXPIRED can be happened if the request sending time takes too long. \n
 *         @ref ERR_UEM_SUSPEND is occurred when the program is going to exit.
 */
uem_result UKRemoteChannel_ReadFromBuffer(SChannel *pstChannel, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead);

/**
 * @brief Get readable chunk index from queue.
 *
 * This function gets readable chunk index from queue. This function is only used when the channel is an input of Loop task. \n
 * This function is a part of @ref SChannelAPI.
 *
 * @warning This function is only working when the connection method is individual \n
 * because UEMLiteProtocol does not support array channel which is not supported for aggregate connection. \n
 * For aggregate connection, this function always return 0 as a chunk index number.
 *
 * @param pstChannel a single channel structure.
 * @param[out] pnChunkIndex an available chunk index to be read.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_INTERNAL_FAIL, @ref ERR_UEM_SUSPEND. \n
 *         @ref ERR_UEM_SUSPEND is occurred when the program is going to exit. \n
 *         @ref ERR_UEM_INTERNAL_FAIL is occurred when the remote device sends an error.
 */
uem_result UKRemoteChannel_GetAvailableChunk (SChannel *pstChannel, OUT int *pnChunkIndex);

/**
 * @brief Get number of data in remote channel.
 *
 * This function retrieves the number of data in channel in specific chunk index. \n
 * For aggregate connection, chunk index value is not used. \n
 * This function is a part of @ref SChannelAPI.
 *
 * @param pstChannel a single channel structure.
 * @param nChunkIndex chunk index (only applicable for individual connection).
 * @param[out] pnDataNum amount of data in the chunk.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_INTERNAL_FAIL, @ref ERR_UEM_SUSPEND, \n
 *         @ref ERR_UEM_ILLEGAL_DATA. \n
 *         @ref ERR_UEM_SUSPEND is occurred when the program is going to exit. \n
 *         @ref ERR_UEM_INTERNAL_FAIL is occurred when the remote device sends an error. \n
 *         @ref ERR_UEM_ILLEGAL_DATA is occurred when received data is corrupted.
 */
uem_result UKRemoteChannel_GetNumOfAvailableData (SChannel *pstChannel, IN int nChunkIndex, OUT int *pnDataNum);

/**
 * @brief write data to local channel as a queue.
 *
 * This function writes data to the channel as a queue. \n
 * Because it writes to queue, the write block happened if the channel data is full. \n
 * This function is a part of @ref SChannelAPI.
 *
 * @param pstChannel a single channel structure.
 * @param pBuffer data to write.
 * @param nDataToWrite amount of data to write.
 * @param nChunkIndex chunk index to write.
 * @param[out] pnDataWritten amount of data written.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_SUSPEND. \n
 *         @ref ERR_UEM_SUSPEND is occurred when the program is going to exit.
 */
uem_result UKRemoteChannel_WriteToQueue (SChannel *pstChannel, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten);

/**
 * @brief Fill initial data in channel.
 *
 * This function fills initial data in channel.
 * This function is a part of @ref SChannelAPI.
 *
 * @param pstChannel a single channel structure.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKRemoteChannel_FillInitialData(SChannel *pstChannel);

/**
 * @brief write data to local channel as a buffer.
 *
 * This function writes data to the channel as a buffer. \n
 * Because it write to buffer, it overwrites the data in the channel. \n
 * This function is a part of @ref SChannelAPI.
 *
 * @param pstChannel a single channel structure.
 * @param pBuffer data to write.
 * @param nDataToWrite  amount of data to write.
 * @param nChunkIndex (not used).
 * @param[out] pnDataWritten amount of data written.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_SUSPEND. \n
 *         @ref ERR_UEM_SUSPEND is occurred when the program is going to exit.
 */
uem_result UKRemoteChannel_WriteToBuffer (SChannel *pstChannel, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten);

/**
 * @brief Exit block to terminate channel.
 *
 * This function set exit flags to escape from blocking state of the channel. \n
 * @ref EXIT_FLAG_READ and @ref EXIT_FLAG_WRITE can be used as a bit flag. \n
 * This function is a part of @ref SChannelAPI.
 *
 * @param pstChannel a single channel structure.
 * @param nExitFlag an exit flag (@ref EXIT_FLAG_READ, @ref EXIT_FLAG_WRITE).
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKRemoteChannel_SetExit(SChannel *pstChannel, int nExitFlag);

/**
 * @brief Clear exit flag to restart channel.
 *
 * This function clears exit flags to reuse channel. \n
 * This function can be called for rerunning the task graphs.
 * This function is a part of @ref SChannelAPI.
 *
 * @param pstChannel a single channel structure.
 * @param nExitFlag an exit flag to clear (@ref EXIT_FLAG_READ, @ref EXIT_FLAG_WRITE).
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKRemoteChannel_ClearExit(SChannel *pstChannel, int nExitFlag);

/**
 * @brief Finalize a remote channel.
 *
 * This function finalizes a remote channel. \n
 * If the channel is a writer, it finalizes internal shared memory channel (UKChannelMemory). \n
 * Depending on the connection methods (aggregate or individual), this function disconnects from aggregate service (UKSerialCommunicationManager) \n
 * or directly disconnects from remote devices. \n
 * This function is a part of @ref SChannelAPI.
 *
 * @param pstChannel a single channel structure.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 *
 * @sa UKRemoteChannel_Initialize.
 */
uem_result UKRemoteChannel_Finalize(SChannel *pstChannel);

/**
 * @brief Initialize remote channel APIs.
 *
 * This function initializes remote channel APIs. \n
 * This function executes remote communication module (ex. TCP Server, establish aggregate Bluetooth connection). \n
 * This function is a part of @ref SChannelAPI.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - error results from corresponding FnChannelAPIInitialize() functions.
 *
 * @sa UKRemoteChannel_APIFinalize.
 */
uem_result UKRemoteChannel_APIInitialize();

/**
 * @brief Finalize remote channel APIs.
 *
 * This function finalizes remote channel APIs. \n
 * This function terminates remote communication modules. \n
 * This function is a part of @ref SChannelAPI.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - error results from corresponding FnChannelAPIFinalize() functions.
 *
 * @sa UKRemoteChannel_APIInitialize.
 */
uem_result UKRemoteChannel_APIFinalize();

extern int g_nRemoteCommunicationModuleNum;
extern FnChannelAPIInitialize g_aFnRemoteCommunicationModuleIntializeList[];
extern FnChannelAPIFinalize g_aFnRemoteCommunicationModuleFinalizeList[];

#ifdef __cplusplus
}
#endif


#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKREMOTECHANNEL_H_ */
