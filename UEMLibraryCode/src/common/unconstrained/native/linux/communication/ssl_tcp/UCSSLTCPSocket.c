/*
 * UCTCPSocket.c
 *
 *  Created on: 2018. 10. 2.
 *      Author: chjej202
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#ifndef _WIN32
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <openssl/ssl.h>

#else
#include <winsock2.h>
#endif

#include <uem_common.h>

#ifndef _WIN32
	#ifdef DEBUG_PRINT
#include <errno.h>
#include <string.h>
	#endif
#endif

#include <UCBasic.h>
#include <UCAlloc.h>

#include <UCSSLTCPSocket.h>
#include <UCDynamicSocket.h>

static uem_result selectTimeout(int nSocketfd, fd_set *pstReadSet, fd_set *pstWriteSet, fd_set *pstExceptSet, int nTimeout)
{
    uem_result result = ERR_UEM_UNKNOWN;
    struct timeval stTimeVal;
    int nRet = 0;

    if(pstReadSet != NULL)
    {
        FD_ZERO(pstReadSet);
        FD_SET(nSocketfd, pstReadSet);
    }

    if(pstWriteSet != NULL)
    {
        FD_ZERO(pstWriteSet);
        FD_SET(nSocketfd, pstWriteSet);
    }

    if(pstExceptSet != NULL)
   {
       FD_ZERO(pstExceptSet);
       FD_SET(nSocketfd, pstExceptSet);
   }

    stTimeVal.tv_sec = nTimeout;
    stTimeVal.tv_usec = 0;

    nRet = select(nSocketfd+1, pstReadSet, pstWriteSet, pstExceptSet, &stTimeVal);
    if(nRet < 0)
    {
        ERRASSIGNGOTO(result, ERR_UEM_SELECT_ERROR, _EXIT);
    }
    else if(nRet == 0)
    {
        ERRASSIGNGOTO(result, ERR_UEM_NET_TIMEOUT, _EXIT);
    }
    else
    {
        result = ERR_UEM_NOERROR;
    }
_EXIT:
    return result;
}

static uem_result initializeCTX(SKeyInfo *pstKeyInfo, uem_bool bIsServer, SSL_CTX **ppstCTX) 
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSL_CTX *pstCTX = NULL;

	if(bIsServer) 
	{
		if(!(pstCTX = SSL_CTX_new(SSLv23_server_method())))
		{
			ERRASSIGNGOTO(result, ERR_UEM_SSL_ERROR, _EXIT);
		}
		if(pstKeyInfo == NULL || pstKeyInfo->pszCAPublicKey == NULL || pstKeyInfo->pszPublicKey == NULL || pstKeyInfo->pszPrivateKey == NULL)
		{
			ERRASSIGNGOTO(result, ERR_UEM_SSL_KEY_NOT_FOUND, _EXIT);
		}
	}
	else
	{
		if(!(pstCTX = SSL_CTX_new(SSLv23_client_method())))
		{
			ERRASSIGNGOTO(result, ERR_UEM_SSL_ERROR, _EXIT);
		}
	}

	if(pstKeyInfo != NULL && pstKeyInfo->pszCAPublicKey != NULL && pstKeyInfo->pszPublicKey != NULL && pstKeyInfo->pszPrivateKey != NULL)
	{
		if(SSL_CTX_load_verify_locations(pstCTX, pstKeyInfo->pszCAPublicKey, NULL) != 1)
		{
			ERRASSIGNGOTO(result, ERR_UEM_SSL_KEY_INVALID, _EXIT);
		}
		if(bIsServer)
		{
			SSL_CTX_set_client_CA_list(pstCTX, SSL_load_client_CA_file(pstKeyInfo->pszCAPublicKey));
		}
		if (SSL_CTX_use_certificate_file(pstCTX, pstKeyInfo->pszPublicKey, SSL_FILETYPE_PEM) != 1)
		{
			ERRASSIGNGOTO(result, ERR_UEM_SSL_KEY_INVALID, _EXIT);
		}
		if (SSL_CTX_use_PrivateKey_file(pstCTX, pstKeyInfo->pszPrivateKey, SSL_FILETYPE_PEM) != 1)
		{
			ERRASSIGNGOTO(result, ERR_UEM_SSL_KEY_INVALID, _EXIT);
		}
		if (SSL_CTX_check_private_key(pstCTX) != 1)
		{
			ERRASSIGNGOTO(result, ERR_UEM_SSL_KEY_INVALID, _EXIT);
		}
		SSL_CTX_set_mode(pstCTX, SSL_MODE_AUTO_RETRY);
		SSL_CTX_set_verify(pstCTX, SSL_VERIFY_PEER, NULL);
		SSL_CTX_set_verify_depth(pstCTX, 1);
	}

	*ppstCTX = pstCTX;

	result = ERR_UEM_NOERROR;
_EXIT:
	if(result != ERR_UEM_NOERROR && pstCTX != NULL) 
	{
		SSL_CTX_free(pstCTX);	
	}
	return result;
}

static uem_result initializeSSL(SSL_CTX *pstCTX, SSL **ppstSSL)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSL *pstSSL = NULL;
	pstSSL = SSL_new(pstCTX);

	if(pstSSL != NULL)
	{
		result = ERR_UEM_NOERROR;	
	}

	*ppstSSL = pstSSL;

	return result;
}


uem_result UCSSLTCPSocket_Initialize()
{
	SSL_load_error_strings();
	OpenSSL_add_ssl_algorithms();

	return ERR_UEM_NOERROR;
}

uem_result UCSSLTCPSocket_Create(IN SSSLSocketInfo *pstSSLSocketInfo, IN uem_bool bIsServer, OUT HSSLSocket *phSocket)
{
    uem_result result = ERR_UEM_UNKNOWN;
    SUCSSLSocket *pstSocket = NULL;

    pstSocket = (SUCSSLSocket *) UCAlloc_malloc(sizeof(SUCSSLSocket));
    ERRMEMGOTO(pstSocket, result, _EXIT);

	pstSocket->enID = ID_UEM_SSL_SOCKET;

	result = UCDynamicSocket_Create(pstSSLSocketInfo->pstSocketInfo, bIsServer, &(pstSocket->hSocket));
	ERRIFGOTO(result, _EXIT);

	pstSocket->pstSSLInfo = (SSSLInfo *) UCAlloc_malloc(sizeof(SSSLInfo));
	ERRMEMGOTO(pstSocket->pstSSLInfo, result, _EXIT);

	pstSocket->pstSSLInfo->pstSSL = NULL;
	pstSocket->pstSSLInfo->pstCTX = NULL;

	result = initializeCTX(pstSSLSocketInfo->pstKeyInfo, bIsServer, &(pstSocket->pstSSLInfo->pstCTX));
	ERRIFGOTO(result, _EXIT);

	*phSocket = pstSocket;

    result = ERR_UEM_NOERROR;
_EXIT:
    if(result != ERR_UEM_NOERROR && pstSocket != NULL)
    {
        UCSSLTCPSocket_Destroy((HSSLSocket *)&pstSocket);
    }
    return result;
}

uem_result UCSSLTCPSocket_Destroy(IN OUT HSSLSocket *phSocket)
{
    uem_result result = ERR_UEM_UNKNOWN;
    SUCSSLSocket *pstSocket = NULL;
	SSSLInfo *pstSSLInfo = NULL;

    pstSocket = (SUCSSLSocket *) *phSocket;
	pstSSLInfo = pstSocket->pstSSLInfo;

	if(pstSSLInfo != NULL) 
	{   
		if(pstSSLInfo->pstSSL != NULL) 
		{   
			if((result = SSL_shutdown(pstSSLInfo->pstSSL)) == 1)    
			{   
				SSL_free(pstSSLInfo->pstSSL);
				pstSSLInfo->pstSSL = NULL;

				if(pstSSLInfo->pstCTX != NULL) 
				{   
					SSL_CTX_free(pstSSLInfo->pstCTX);   
					pstSSLInfo->pstCTX = NULL;
				}   
			}   
		}   
	}   
	SAFEMEMFREE(pstSSLInfo);

	result = UCDynamicSocket_Destroy(&(pstSocket->hSocket));
	ERRIFGOTO(result, _EXIT);

    SAFEMEMFREE(pstSocket);

    *phSocket = NULL;

    result = ERR_UEM_NOERROR;
_EXIT:
    return result;
}

uem_result UCSSLTCPSocket_Bind(HSSLSocket hServerSocket)
{
    uem_result result = ERR_UEM_UNKNOWN;
    SUCSSLSocket *pstSocket = NULL;

    pstSocket = (SUCSSLSocket *) hServerSocket;

	result = UCDynamicSocket_Bind(pstSocket->hSocket);
	ERRIFGOTO(result, _EXIT);

    result = ERR_UEM_NOERROR;
_EXIT:
    return result;
}

uem_result UCSSLTCPSocket_Listen(HSSLSocket hServerSocket)
{
    uem_result result = ERR_UEM_UNKNOWN;
    SUCSSLSocket *pstSocket = NULL;

    pstSocket = (SUCSSLSocket *) hServerSocket;

	result = UCDynamicSocket_Listen(pstSocket->hSocket);
	ERRIFGOTO(result, _EXIT);

    result = ERR_UEM_NOERROR;
_EXIT:
    return result;
}

uem_result UCSSLTCPSocket_Accept(HSSLSocket hServerSocket, IN int nTimeout, IN OUT HSSLSocket hSocket)
{
    uem_result result = ERR_UEM_UNKNOWN;
    SUCSSLSocket *pstSocket = NULL;
    SUCSSLSocket *pstCliSocket = NULL;

    pstSocket = (SUCSSLSocket *) hServerSocket;
    pstCliSocket = (SUCSSLSocket *) hSocket;
	pstCliSocket->pstSSLInfo = (SSSLInfo *) UCAlloc_malloc(sizeof(SSSLInfo));

	result = UCDynamicSocket_Accept(pstSocket->hSocket, nTimeout, pstCliSocket->hSocket);
	ERRIFGOTO(result, _EXIT);

	result = initializeSSL(pstSocket->pstSSLInfo->pstCTX, &(pstCliSocket->pstSSLInfo->pstSSL));
	ERRIFGOTO(result, _EXIT);

	if((result = SSL_set_fd(pstCliSocket->pstSSLInfo->pstSSL, pstCliSocket->hSocket->nSocketfd)) != 1)
	{
		ERRASSIGNGOTO(result, ERR_UEM_SSL_ERROR, _EXIT);
	}
	if((result = SSL_accept(pstCliSocket->pstSSLInfo->pstSSL)) != 1) 
	{
		if(result != 0) 
		{
			SSL_shutdown(pstCliSocket->pstSSLInfo->pstSSL);
			pstCliSocket->pstSSLInfo->pstSSL = NULL;	
		}	
		SSL_free(pstCliSocket->pstSSLInfo->pstSSL);
		ERRASSIGNGOTO(result, ERR_UEM_SSL_ERROR, _EXIT);
	}

    result = ERR_UEM_NOERROR;
_EXIT:
    return result;
}

uem_result UCSSLTCPSocket_Connect(HSSLSocket hClientSocket, IN int nTimeout)
{
    uem_result result = ERR_UEM_UNKNOWN;
    SUCSSLSocket *pstSocket = NULL;
	SSSLInfo *pstSSLInfo = NULL;

    pstSocket = (SUCSSLSocket *) hClientSocket;
	pstSSLInfo = pstSocket->pstSSLInfo;

	result = UCDynamicSocket_Connect(pstSocket->hSocket, nTimeout);
	ERRIFGOTO(result, _EXIT);

	result = initializeSSL(pstSSLInfo->pstCTX, &(pstSSLInfo->pstSSL));
	ERRIFGOTO(result, _EXIT);

	if((result = SSL_set_fd(pstSSLInfo->pstSSL, pstSocket->hSocket->nSocketfd)) != 1)
	{
		ERRASSIGNGOTO(result, ERR_UEM_SSL_ERROR, _EXIT);
	}
	if((result = SSL_connect(pstSSLInfo->pstSSL)) != 1) 
	{
		if(result != 0) 
		{
			SSL_shutdown(pstSSLInfo->pstSSL);
			pstSSLInfo->pstSSL = NULL;	
		}	
		SSL_free(pstSSLInfo->pstSSL);
		ERRASSIGNGOTO(result, ERR_UEM_SSL_ERROR, _EXIT);
	}
	if(SSL_do_handshake(pstSSLInfo->pstSSL) != 1)
	{
		ERRASSIGNGOTO(result, ERR_UEM_SSL_ERROR, _EXIT);
	}
	if(SSL_get_verify_result(pstSSLInfo->pstSSL) != X509_V_OK) 
	{
		ERRASSIGNGOTO(result, ERR_UEM_SSL_ERROR, _EXIT);
	}

    result = ERR_UEM_NOERROR;
_EXIT:
    return result;
}

uem_result UCSSLTCPSocket_Disconnect(HSSLSocket hClientSocket)
{
	uem_result result = ERR_UEM_UNKNOWN;
    SUCSSLSocket *pstSocket = NULL;
	SSSLInfo *pstSSLInfo = NULL;

    pstSocket = (SUCSSLSocket *) hClientSocket;
	pstSSLInfo = pstSocket->pstSSLInfo;

	result = UCDynamicSocket_Disconnect(pstSocket->hSocket);
	ERRIFGOTO(result, _EXIT);

	if(pstSSLInfo != NULL) 
	{   
		if(pstSSLInfo->pstSSL != NULL) 
		{   
			if((result = SSL_shutdown(pstSSLInfo->pstSSL)) == 1)    
			{   
				SSL_free(pstSSLInfo->pstSSL);
				pstSSLInfo->pstSSL = NULL;
			}   
		}   
	}   

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UCSSLTCPSocket_Send(HSSLSocket hSocket, IN int nTimeout, IN char *pData, IN int nDataLen, OUT int *pnSentSize)
{
    uem_result result = ERR_UEM_UNKNOWN;
    SUCSSLSocket *pstSocket = NULL;
	SUCSocket *pstHSocket = NULL;
    fd_set stWriteSet;
    int nDataSent = 0;
#ifdef ARGUMENT_CHECK
    IFVARERRASSIGNGOTO(pData, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

    if(IS_VALID_HANDLE(hSocket, ID_UEM_SSL_SOCKET) == FALSE)
    {
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
    }

    if(nTimeout <= 0 || nDataLen <= 0)
    {
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
    }
#endif
    pstSocket = (SUCSSLSocket *) hSocket;
	pstHSocket = pstSocket->hSocket;
#ifdef ARGUMENT_CHECK
    if(pstHSocket->bIsServer == TRUE)
    {
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_SOCKET, _EXIT);
    }
#endif
	if(SSL_pending(pstSocket->pstSSLInfo->pstSSL) == 0)
	{
		result = selectTimeout(pstHSocket->nSocketfd, NULL, &stWriteSet, NULL, nTimeout);
		ERRIFGOTO(result, _EXIT);
	}

	nDataSent = SSL_write(pstSocket->pstSSLInfo->pstSSL, pData, nDataLen);
	if(nDataSent < 0)
	{
		ERRASSIGNGOTO(result, ERR_UEM_NET_SEND_ERROR, _EXIT);
	}
	if(pnSentSize != NULL)
	{
		*pnSentSize = nDataSent;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
    return result;
}

uem_result UCSSLTCPSocket_Receive(HSSLSocket hSocket, IN int nTimeout, IN OUT char *pBuffer, IN int nBufferLen, OUT int *pnReceivedSize)
{
    uem_result result = ERR_UEM_UNKNOWN;
    SUCSSLSocket *pstSocket = NULL;
	SUCSocket *pstHSocket = NULL;
    fd_set stReadSet;
    int nDataReceived = 0;
#ifdef ARGUMENT_CHECK
    IFVARERRASSIGNGOTO(pBuffer, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

    if(IS_VALID_HANDLE(hSocket, ID_UEM_SSL_SOCKET) == FALSE)
    {
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
    }

    if(nTimeout <= 0 || nBufferLen <= 0)
    {
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
    }
#endif
    pstSocket = (SUCSSLSocket *) hSocket;
	pstHSocket = pstSocket->hSocket;
#ifdef ARGUMENT_CHECK
    if(pstHSocket->bIsServer == TRUE)
    {
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_SOCKET, _EXIT);
    }
#endif
	if(SSL_pending(pstSocket->pstSSLInfo->pstSSL) == 0)
	{
		result = selectTimeout(pstHSocket->nSocketfd, &stReadSet, NULL, NULL, nTimeout);
		ERRIFGOTO(result, _EXIT);
	}

	nDataReceived = SSL_read(pstSocket->pstSSLInfo->pstSSL, pBuffer, nBufferLen);
	if(nDataReceived <= 0)
	{
		ERRASSIGNGOTO(result, ERR_UEM_NET_RECEIVE_ERROR, _EXIT);
	}
	if(pnReceivedSize != NULL)
	{
		*pnReceivedSize = nDataReceived;
	}

    result = ERR_UEM_NOERROR;
_EXIT:
    return result;
}
