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

int PORT_INITIALIZE(int nTaskId, const char *pszPortName);
int MQ_RECEIVE(int nChannelId, unsigned char *pBuffer, int nLen);
int MQ_SEND(int nChannelId, unsigned char *pBuffer, int nLen);
int MQ_AVAILABLE(int nChannelId);
int BUF_RECEIVE(int nChannelId, unsigned char *pBuffer, int nLen);
int BUF_SEND(int nChannelId, unsigned char *pBuffer, int nLen);

#ifndef API_LITE
int AC_RECEIVE(int nChannelId, unsigned char *pBuffer, int nLen, int nIndex);
int AC_SEND(int nChannelId, unsigned char *pBuffer, int nLen, int nIndex);
int AC_CHECK(int nChannelId);
#endif

#ifdef __cplusplus
}
#endif

#endif /* SRC_API_INCLUDE_UFPORT_DEPRECATED_H_ */
