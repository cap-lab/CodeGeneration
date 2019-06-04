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
 * @brief (Deprecated)
 *
 * This function
 *
 * @param nTaskId
 * @param pszPortName
 *
 * @return
 */
int PORT_INITIALIZE(int nTaskId, const char *pszPortName);

/**
 * @brief (Deprecated)
 *
 * This function
 *
 * @param nChannelId
 * @param pBuffer
 * @param nLen
 *
 * @return
 */
int MQ_RECEIVE(int nChannelId, unsigned char *pBuffer, int nLen);

/**
 * @brief (Deprecated)
 *
 * This function
 *
 * @param nChannelId
 * @param pBuffer
 * @param nLen
 *
 * @return
 */
int MQ_SEND(int nChannelId, unsigned char *pBuffer, int nLen);

/**
 * @brief (Deprecated)
 *
 * This function
 *
 * @param nChannelId
 *
 * @return
 */
int MQ_AVAILABLE(int nChannelId);

/**
 * @brief (Deprecated)
 *
 * This function
 *
 * @param nChannelId
 * @param pBuffer
 * @param nLen
 *
 * @return
 */
int BUF_RECEIVE(int nChannelId, unsigned char *pBuffer, int nLen);

/**
 * @brief (Deprecated)
 *
 * This function
 *
 * @param nChannelId
 * @param pBuffer
 * @param nLen
 *
 * @return
 */
int BUF_SEND(int nChannelId, unsigned char *pBuffer, int nLen);

#ifndef API_LITE
/**
 * @brief (Deprecated)
 *
 * This function
 *
 * @param nChannelId
 * @param pBuffer
 * @param nLen
 * @param nIndex
 *
 * @return
 */
int AC_RECEIVE(int nChannelId, unsigned char *pBuffer, int nLen, int nIndex);

/**
 * @brief (Deprecated)
 *
 * This function
 *
 * @param nChannelId
 * @param pBuffer
 * @param nLen
 * @param nIndex
 *
 * @return
 */
int AC_SEND(int nChannelId, unsigned char *pBuffer, int nLen, int nIndex);

/**
 * @brief (Deprecated)
 *
 * This function
 *
 * @param nChannelId
 *
 * @return
 */
int AC_CHECK(int nChannelId);
#endif

#ifdef __cplusplus
}
#endif

#endif /* SRC_API_INCLUDE_UFPORT_DEPRECATED_H_ */
