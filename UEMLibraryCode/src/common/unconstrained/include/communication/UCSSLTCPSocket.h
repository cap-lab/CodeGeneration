/*
 * UCTCPSocket.h
 *
 *  Created on: 2018. 10. 3.
 *      Author: chjej202
 */

#ifndef SRC_COMMON_UNCONSTRAINED_INCLUDE_COMMUNICATION_UCSSLTCPSOCKET_H_
#define SRC_COMMON_UNCONSTRAINED_INCLUDE_COMMUNICATION_UCSSLTCPSOCKET_H_

#include <openssl/ssl.h>

#include <uem_common.h>

#include <UCTCPSocket.h>
#include <UCDynamicSocket.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct _SKeyInfo 
{
	char *pszCAPublicKey;
	char *pszPublicKey;
	char *pszPrivateKey;
} SKeyInfo;

typedef struct _SSSLInfo 
{
	SSL *pstSSL;
	SSL_CTX *pstCTX;
	uem_bool bKeyLoaded;
} SSSLInfo;

typedef struct _SSSLSocketInfo 
{
	SSocketInfo *pstSocketInfo;	
	SKeyInfo *pstKeyInfo;
} SSSLSocketInfo;

typedef struct _SUCSSLSocket 
{
	EUemModuleId enID;
	HSocket hSocket;
	SSSLInfo *pstSSLInfo;
} SUCSSLSocket;

typedef struct _SUCSSLSocket *HSSLSocket;

uem_result UCSSLTCPSocket_Initialize();

uem_result UCSSLTCPSocket_Create(IN SSSLSocketInfo *pstSocketInfo, IN uem_bool bIsServer, OUT HSSLSocket *phSocket);

uem_result UCSSLTCPSocket_Destroy(IN OUT HSSLSocket *phSocket);

uem_result UCSSLTCPSocket_Bind(HSSLSocket hServerSocket);

uem_result UCSSLTCPSocket_Listen(HSSLSocket hServerSocket);

uem_result UCSSLTCPSocket_Accept(HSSLSocket hServerSocket, IN int nTimeout, IN OUT HSSLSocket hSocket);

uem_result UCSSLTCPSocket_Connect(HSSLSocket hClientSocket, IN int nTimeout);

uem_result UCSSLTCPSocket_Disconnect(HSSLSocket hClientSocket);

uem_result UCSSLTCPSocket_Send(HSSLSocket hSocket, IN int nTimeout, IN char *pData, IN int nDataLen, OUT int *pnSentSize);

uem_result UCSSLTCPSocket_Receive(HSSLSocket hSocket, IN int nTimeout, IN OUT char *pBuffer, IN int nBufferLen, OUT int *pnReceivedSize);

uem_result UCSSLTCPSocket_Bind_API(HSocket hServerSocket);

uem_result UCSSLTCPSocket_Accept_API(HSocket hServerSocket, HSocket hClientSocket);

uem_result UCSSLTCPSocket_Connect_API(HSocket hSocket, IN int nTimeout);

#ifdef __cplusplus
}
#endif

#endif /* SRC_COMMON_UNCONSTRAINED_INCLUDE_COMMUNICATION_UCSSLTCPSOCKET_H_ */
