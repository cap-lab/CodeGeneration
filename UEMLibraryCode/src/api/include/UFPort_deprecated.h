/*
 * UFPort_deprecated.h
 *
 *  Created on: 2017. 8. 13.
 *      Author: jej
 */

#ifndef SRC_API_INCLUDE_UFPORT_DEPRECATED_H_
#define SRC_API_INCLUDE_UFPORT_DEPRECATED_H_

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief (Deprecated) returns channel ID corresponding to the task and port name
 *
 * returned nChannelId value is INVALID_CHANNEL_ID(== -1) if channel not exists.
 *
 * @param nTaskId id of task
 * @param pszPortName port name
 *
 * @return corresponding channel id
 **
 */

int PORT_INITIALIZE(int nTaskId, const char *pszPortName);

/**
 * @brief (Deprecated) receive data from Message Queue..
 *
 * blocked if no data could be read.
 *
 * @param nChannelId ID of channel to receive data.
 * @param pBuffer buffer to receive data.
 * @param nLen buffer size.
 *
 * @return size of received data.
 */
int MQ_RECEIVE(int nChannelId, unsigned char *pBuffer, int nLen);

/**
 * @brief (Deprecated) send data to Message Queue.
 *
 * @param nChannelId ID of channel to send data.
 * @param pBuffer buffer to send data.
 * @param nLen buffer size.
 *
 * @return sent data size.
 */
int MQ_SEND(int nChannelId, unsigned char *pBuffer, int nLen);

/**
 * @brief (Deprecated) Check if available data exists on the port.
 *
 * @param nChannelId ID of channel to check available data.
 *
 * @return data size that could be received.
 */
int MQ_AVAILABLE(int nChannelId);

/**
@brief (Deprecated) receive data from buffer
 *
 * blocked if no data could be read.
 *
 * @param nChannelId ID of channel to receive data.
 * @param pBuffer buffer to receive data.
 * @param nLen buffer size.
 *
 * @return size of received data.
 */
int BUF_RECEIVE(int nChannelId, unsigned char *pBuffer, int nLen);

/**
 * @brief (Deprecated) send data to buffer.
 *
 * @param nChannelId ID of channel to send data.
 * @param pBuffer buffer to send data.
 * @param nLen buffer size.
 *
 * @return sent data size.
 */
int BUF_SEND(int nChannelId, unsigned char *pBuffer, int nLen);

#ifndef API_LITE
/**
 * @brief (Deprecated) receives data from specific index of the array channel.
 *
 * blocked if no data in that chunk index exists.
 *
 * @param nChannelId ID of channel to receive data.
 * @param pBuffer buffer to receive data.
 * @param nLen buffer size.
 * @param nIndex array channel index.
 *
 * @return size of received data
 */
int AC_RECEIVE(int nChannelId, unsigned char *pBuffer, int nLen, int nIndex);

/**
 * @brief (Deprecated) send data to a specific index of the array channel.
 *
 * blocked If the data in that chunk index is not read yet.
 *
 * @param nChannelId ID of channel to send data.
 * @param pBuffer buffer to send data.
 * @param nLen buffer size.
 * @param nIndex  array channel index.
 *
 * @return sent data size.
 */
int AC_SEND(int nChannelId, unsigned char *pBuffer, int nLen, int nIndex);

/**
 * @brief (Deprecated)  return an index that can receive the current data from the array channel.
 *
 * @param nChannelId ID of channel to check available data.
 *
 * @return Index of a specific chunk having data to be received.
 */
int AC_CHECK(int nChannelId);
#endif

#ifdef __cplusplus
}
#endif

#endif /* SRC_API_INCLUDE_UFPORT_DEPRECATED_H_ */
