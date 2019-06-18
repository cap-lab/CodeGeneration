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
 * @brief Initialize a channel memory.
 *
 * This function initializes a channel memory.
 *
 * @param pstChannel a single channel structure.
 * @param pstSharedMemoryChannel shared memory channel structure.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * (unconstrained device) \n
 * Errors to be returned - @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_MUTEX_ERROR, @ref ERR_UEM_INTERNAL_FAIL \n
 *
 * @sa UKChannelMemory_Finalize.
 *
 */
uem_result UKChannelMemory_Initialize(SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel);

/**
 * @brief Read data from a channel memory as a queue.
 *
 * This function reads data from a channel memory. \n
 * Because it reads from queue, data in the channel is removed after reading the data. \n
 * for unconstrained device, read method is different depending on channel type(array queue or general queue) \n
 *
 * @param pstChannel a single channel structure.
 * @param pstSharedMemoryChannel shared memory channel structure.
 * @param[in,out] pBuffer buffer to store read data.
 * @param nDataToRead size of data to read.
 * @param nChunkIndex chunk index to read.
 * @param[out] pnDataRead size of data read.
 *
 * @return
 *  @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *  @ref ERR_UEM_INVALID_PARAM if channel's pChannelStruct is NULL. \n
 *  (constrained device) \n
 *  @ref ERR_UEM_READ_BLOCK if channel data is not enough to read or not initialized. \n
 *  (unconstrained device, array queue) \n
 *  @ref ERR_UEM_SUSPEND if bReadExit is true. \n
 *  @ref ERR_UEM_ILLEGAL_DATA if target chunk data size is not equal to @a nDataToRead. \n
 *  Errors to be returned - @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_MUTEX_ERROR, @ref ERR_UEM_INTERNAL_FAIL \n
 *  (unconstrained device, general queue) \n
 *  @ref ERR_UEM_SUSPEND if bReadExit is true.
 *  Errors to be returned - @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_MUTEX_ERROR, @ref ERR_UEM_INTERNAL_FAIL \n
 */
uem_result UKChannelMemory_ReadFromQueue(SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead);

/**
 * @brief Read data from a channel memory as a buffer.
 *
 * This function reads data from a channel memory. \n
 * Because it reads from buffer, data in the channel is preserved after reading the data. \n
 *
 * @param pstChannel a single channel structure.
 * @param pstSharedMemoryChannel shared memory channel structure.
 * @param[in,out] pBuffer buffer to store read data.
 * @param nDataToRead size of data to read.
 * @param nChunkIndex (not used).
 * @param[out] pnDataRead size of data read.
 *
 * @return
 *  @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *  @ref ERR_UEM_INVALID_PARAM if channel's pChannelStruct is NULL. \n
 * (unconstrained device) \n
 *  Errors to be returned - @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_MUTEX_ERROR, @ref ERR_UEM_INTERNAL_FAIL \n
 */
uem_result UKChannelMemory_ReadFromBuffer(SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead);

/**
 * @brief write data to channel memory as a buffer.
 *
 * This function writes data to the channel as a buffer. \n
 * Because it write to buffer, it overwrites the data in the channel. \n
 *
 * @param pstChannel a single channel structure.
 * @param pstSharedMemoryChannel shared memory channel structure.
 * @param pBuffer data to write.
 * @param nDataToWrite amount of data to write.
 * @param nChunkIndex (not used).
 * @param[out] pnDataWritten amount of data written.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_INVALID_PARAM if channel's pChannelStruct is NULL. \n
 * (unconstrained device) \n
 * Errors to be returned - @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_MUTEX_ERROR, @ref ERR_UEM_INTERNAL_FAIL \n
 */
uem_result UKChannelMemory_WriteToBuffer (SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten);
/**
 * @brief write data to channel memory as a queue.
 *
 * This function writes data to the channel as a queue. \n
 * Because it writes to queue, the write block happened if the channel data is full. \n
 *
 * @param pstChannel a single channel structure.
 * @param pstSharedMemoryChannel shared memory channel structure.
 * @param pBuffer data to write.
 * @param nDataToWrite amount of data to write.
 * @param nChunkIndex chunk index to write.
 * @param[out] pnDataWritten amount of data written.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_INVALID_PARAM if channel's pChannelStruct is NULL. \n
 * (constrained device) \n
 * @ref ERR_UEM_WRITE_BLOCK if remaining channel buffer size is not enough to write. \n
 * (unconstrained device)
 * Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_SUSPEND, @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_MUTEX_ERROR, @ref ERR_UEM_INTERNAL_FAIL \n
 * @ref ERR_UEM_SUSPEND is occurred when the program is going to be exit.
 */
uem_result UKChannelMemory_WriteToQueue (SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten);

/**
 * @brief Get readable chunk index from queue.
 *
 * This function gets readable chunk index from queue.
 * for unconstrained device, getting chunk method is different depending on channel type(array queue or general queue). \n
 *
 * @param pstChannel a single channel structure.
 * @param pstSharedMemoryChannel shared memory channel structure.
 * @param[out] pnChunkIndex an available chunk index to be read.
 *
 * @return  @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *  @ref ERR_UEM_INVALID_PARAM if channel's pChannelStruct is NULL. \n
 *  (constrained device) \n
 *  @ref ERR_UEM_READ_BLOCK if channel data is not enough to read or not initialized. \n
 *  (unconstrained device) \n
 *  Errors to be returned - @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_MUTEX_ERROR, @ref ERR_UEM_INTERNAL_FAIL \n
 *  @ref ERR_UEM_SUSPEND if bReadExit is true. \n
 */
uem_result UKChannelMemory_GetAvailableChunk (SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel, OUT int *pnChunkIndex);

/**
 * @brief Get number of data in channel memory.
 *
 * This function retrieves the number of data in channel in specific chunk index. \n
 *
 * @param pstChannel a single channel structure.
 * @param pstSharedMemoryChannel shared memory channel structure.
 * @param nChunkIndex chunk index.
 * @param[out] pnDataNum amount of data in the chunk.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *  @ref ERR_UEM_INVALID_PARAM if channel's pChannelStruct is NULL. \n
 *  (unconstrained device) \n
 *  Errors to be returned - @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_MUTEX_ERROR, @ref ERR_UEM_INTERNAL_FAIL \n
 */
uem_result UKChannelMemory_GetNumOfAvailableData (SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel, IN int nChunkIndex, OUT int *pnDataNum);

/**
 * @brief Clear channel memory data.
 *
  * This function clears channel memory data. \n
 *
 * @param pstChannel a single channel structure.
 * @param pstSharedMemoryChannel shared memory channel structure.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_INVALID_PARAM if channel's pChannelStruct is NULL. \n
 */
uem_result UKChannelMemory_Clear(SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel);

/**
 * @brief Exit block to terminate channel.
 *
 * This function set exit flags to escape from blocking state of the channel. \n
 * @ref EXIT_FLAG_READ and @ref EXIT_FLAG_WRITE can be used as a bit flag. \n
 * This function is meaningful only for unconstrained device. \n
 *
 * @param pstChannel a single channel structure.
 * @param pstSharedMemoryChannel shared memory channel structure.
 * @param nExitFlag an exit flag (@ref EXIT_FLAG_READ, @ref EXIT_FLAG_WRITE).
 *
 * @return  @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_MUTEX_ERROR, @ref ERR_UEM_INTERNAL_FAIL \n
 */
uem_result UKChannelMemory_SetExit(SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel, int nExitFlag);

/**
 * @brief Clear exit flag to restart channel.
 *
 * This function clears exit flags to reuse channel. \n
 * This function can be called for rerunning the task graphs. \n
 * This function is meaningful only for unconstrained device. \n
 *
 * @param pstChannel a single channel structure.
 * @param pstSharedMemoryChannel shared memory channel structure.
 * @param nExitFlag an exit flag to clear (@ref EXIT_FLAG_READ, @ref EXIT_FLAG_WRITE).
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_MUTEX_ERROR, @ref ERR_UEM_INTERNAL_FAIL \n
 */
uem_result UKChannelMemory_ClearExit(SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel, int nExitFlag);

/**
 * @brief Fill initial data in channel.
 *
 * This function fills initial data in channel.
 *
 * @param pstChannel a single channel structure.
 * @param pstSharedMemoryChannel shared memory channel structure.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * (unconstrained device) \n
 * Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_MUTEX_ERROR, @ref ERR_UEM_INTERNAL_FAIL \n
 */
uem_result UKChannelMemory_FillInitialData(SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel);

/**
 * @brief Finalize a channel memory.
 *
 * This function finalizes a channel memory. \n
 * This function is meaningful only for unconstrained device. \n
 *
 * @param pstChannel a single channel structure.
 * @param pstSharedMemoryChannel shared memory channel structure.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * Errors to be returned - @ref ERR_UEM_INVALID_PARAM. \n
 * (unconstrained device) \n
 * Errors to be returned - @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_MUTEX_ERROR, @ref ERR_UEM_INTERNAL_FAIL \n
 *
 * @sa UKChannelMemory_Initialize.
 */
uem_result UKChannelMemory_Finalize(SChannel *pstChannel, SSharedMemoryChannel *pstSharedMemoryChannel);

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_INCLUDE_UKCHANNELMEMORY_H_ */
