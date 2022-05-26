/*
 * UCSecureTCPSocket.c
 *
 *  Created on: 2020. 5. 21.
 *      Author: jrkim
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#ifndef _WIN32
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#else
#include <winsock2.h>
#endif

#include <openssl/ssl.h>
#include <uem_common.h>

#ifdef DEBUG_PRINT
#include <errno.h>
#include <string.h>
#endif

#include <UCBasic.h>
#include <UCAlloc.h>

#include <UCSecureTCPSocket.h>
#include <UCDynamicSocket.h>

typedef struct _SSecureInfo
{
	SSL *pstSSL;
	SSL_CTX *pstCTX;
	uem_bool bKeyLoaded;
} SSecureInfo;

typedef struct _SUCSecureSocket
{
	EUemModuleId enID;
	HSocket hSocket;
	SSecureInfo stSSLInfo;
} SUCSecureSocket;


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

static uem_result initializeCTX(SSecurityKeyInfo *pstKeyInfo, uem_bool bIsServer, SSL_CTX **ppstCTX, uem_bool *pbKeyLoaded)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSL_CTX *pstCTX = NULL;
	uem_bool bKeyLoaded = FALSE;

	if(bIsServer) 
	{
		if(!(pstCTX = SSL_CTX_new(TLS_server_method())))
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
		if(!(pstCTX = SSL_CTX_new(TLS_client_method())))
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

		bKeyLoaded = TRUE;
	}
	else if(bIsServer) {
		ERRASSIGNGOTO(result, ERR_UEM_SSL_ERROR, _EXIT);
	}

	*pbKeyLoaded = bKeyLoaded;

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


uem_result UCSecureTCPSocket_Initialize()
{
	SSL_load_error_strings();
	OpenSSL_add_ssl_algorithms();

	return ERR_UEM_NOERROR;
}

uem_result UCSecureTCPSocket_Create(IN SSecureSocketInfo *pstSSLSocketInfo, IN uem_bool bIsServer, OUT HSSLSocket *phSocket)
{
    uem_result result = ERR_UEM_UNKNOWN;
    SUCSecureSocket *pstSocket = NULL;

    pstSocket = (SUCSecureSocket *) UCAlloc_malloc(sizeof(SUCSecureSocket));
    ERRMEMGOTO(pstSocket, result, _EXIT);

	pstSocket->enID = ID_UEM_SSL_SOCKET;

	result = UCDynamicSocket_Create(&(pstSSLSocketInfo->stSocketInfo), bIsServer, &(pstSocket->hSocket));
	ERRIFGOTO(result, _EXIT);

	pstSocket->stSSLInfo.pstSSL = NULL;
	pstSocket->stSSLInfo.pstCTX = NULL;
	pstSocket->stSSLInfo.bKeyLoaded = FALSE;

	result = initializeCTX(pstSSLSocketInfo->pstKeyInfo, bIsServer, &(pstSocket->stSSLInfo.pstCTX), &(pstSocket->stSSLInfo.bKeyLoaded));
	ERRIFGOTO(result, _EXIT);

	*phSocket = pstSocket;

    result = ERR_UEM_NOERROR;
_EXIT:
    if(result != ERR_UEM_NOERROR && pstSocket != NULL)
    {
        UCSecureTCPSocket_Destroy((HSSLSocket *)&pstSocket);
    }
    return result;
}

uem_result UCSecureTCPSocket_Destroy(IN OUT HSSLSocket *phSocket)
{
    uem_result result = ERR_UEM_UNKNOWN;
    SUCSecureSocket *pstSocket = NULL;

	pstSocket = (SUCSecureSocket *) *phSocket;

	if(pstSocket->stSSLInfo.pstSSL != NULL)
	{   
		if(SSL_shutdown(pstSocket->stSSLInfo.pstSSL) == 1)
		{   
			SSL_free(pstSocket->stSSLInfo.pstSSL);
			pstSocket->stSSLInfo.pstSSL = NULL;
		}

		if(pstSocket->stSSLInfo.pstCTX != NULL)
		{
			SSL_CTX_free(pstSocket->stSSLInfo.pstCTX);
			pstSocket->stSSLInfo.pstCTX = NULL;
		}
	}   

	result = UCDynamicSocket_Destroy(&(pstSocket->hSocket));
	ERRIFGOTO(result, _EXIT);

	SAFEMEMFREE(pstSocket);

    *phSocket = NULL;

    result = ERR_UEM_NOERROR;
_EXIT:
    return result;
}

uem_result UCSecureTCPSocket_Bind(HSSLSocket hServerSocket)
{
    uem_result result = ERR_UEM_UNKNOWN;
    SUCSecureSocket *pstServerSocket = NULL;

    pstServerSocket = (SUCSecureSocket *) hServerSocket;

	result = UCDynamicSocket_Bind(pstServerSocket->hSocket);
	ERRIFGOTO(result, _EXIT);

    result = ERR_UEM_NOERROR;
_EXIT:
    return result;
}

uem_result UCSecureTCPSocket_Listen(HSSLSocket hServerSocket)
{
    uem_result result = ERR_UEM_UNKNOWN;
    SUCSecureSocket *pstServerSocket = NULL;

    pstServerSocket = (SUCSecureSocket *) hServerSocket;

	result = UCDynamicSocket_Listen(pstServerSocket->hSocket);
	ERRIFGOTO(result, _EXIT);

    result = ERR_UEM_NOERROR;
_EXIT:
    return result;
}

uem_result UCSecureTCPSocket_Accept(HSSLSocket hServerSocket, IN int nTimeout, IN OUT HSSLSocket hSocket)
{
    uem_result result = ERR_UEM_UNKNOWN;
    SUCSecureSocket *pstServerSocket = NULL;
    SUCSecureSocket *pstCliSocket = NULL;

    pstServerSocket = (SUCSecureSocket *) hServerSocket;
    pstCliSocket = (SUCSecureSocket *) hSocket;

	result = UCDynamicSocket_Accept(pstServerSocket->hSocket, nTimeout, pstCliSocket->hSocket);
	ERRIFGOTO(result, _EXIT);

	result = initializeSSL(pstServerSocket->stSSLInfo.pstCTX, &(pstCliSocket->stSSLInfo.pstSSL));
	ERRIFGOTO(result, _EXIT);

	if((result = SSL_set_fd(pstCliSocket->stSSLInfo.pstSSL, pstCliSocket->hSocket->nSocketfd)) != 1)
	{
		ERRASSIGNGOTO(result, ERR_UEM_SSL_ERROR, _EXIT);
	}
	if((result = SSL_accept(pstCliSocket->stSSLInfo.pstSSL)) != 1)
	{
		if(result != 0) 
		{
			SSL_shutdown(pstCliSocket->stSSLInfo.pstSSL);
			pstCliSocket->stSSLInfo.pstSSL = NULL;
		}	
		SSL_free(pstCliSocket->stSSLInfo.pstSSL);
		ERRASSIGNGOTO(result, ERR_UEM_SSL_ERROR, _EXIT);
	}

    result = ERR_UEM_NOERROR;
_EXIT:
    return result;
}

uem_result UCSecureTCPSocket_Connect(HSSLSocket hClientSocket, IN int nTimeout)
{
    uem_result result = ERR_UEM_UNKNOWN;
    SUCSecureSocket *pstSocket = NULL;

    pstSocket = (SUCSecureSocket *) hClientSocket;

	result = UCDynamicSocket_Connect(pstSocket->hSocket, nTimeout);
	ERRIFGOTO(result, _EXIT);

	result = initializeSSL(pstSocket->stSSLInfo.pstCTX, &(pstSocket->stSSLInfo.pstSSL));
	ERRIFGOTO(result, _EXIT);

	if(SSL_set_fd(pstSocket->stSSLInfo.pstSSL, pstSocket->hSocket->nSocketfd) != 1)
	{
		ERRASSIGNGOTO(result, ERR_UEM_SSL_ERROR, _EXIT);
	}

	if(SSL_connect(pstSocket->stSSLInfo.pstSSL) != 1)
	{
		if(result != 0) 
		{
			SSL_shutdown(pstSocket->stSSLInfo.pstSSL);
			pstSocket->stSSLInfo.pstSSL = NULL;
		}	
		SSL_free(pstSocket->stSSLInfo.pstSSL);
		ERRASSIGNGOTO(result, ERR_UEM_SSL_ERROR, _EXIT);
	}

	if(pstSocket->stSSLInfo.bKeyLoaded) {
		if(SSL_get_peer_certificate(pstSocket->stSSLInfo.pstSSL) != NULL)
		{
			if(SSL_get_verify_result(pstSocket->stSSLInfo.pstSSL) != X509_V_OK)
			{
				ERRASSIGNGOTO(result, ERR_UEM_SSL_ERROR, _EXIT);
			}
		}
		else
		{
			ERRASSIGNGOTO(result, ERR_UEM_SSL_ERROR, _EXIT);
		}
	}

    result = ERR_UEM_NOERROR;
_EXIT:
    return result;
}

uem_result UCSecureTCPSocket_Disconnect(HSSLSocket hClientSocket)
{
	uem_result result = ERR_UEM_UNKNOWN;
    SUCSecureSocket *pstSocket = NULL;

    pstSocket = (SUCSecureSocket *) hClientSocket;

	result = UCDynamicSocket_Disconnect(pstSocket->hSocket);
	ERRIFGOTO(result, _EXIT);

	if(pstSocket->stSSLInfo.pstSSL != NULL)
	{   
		if((result = SSL_shutdown(pstSocket->stSSLInfo.pstSSL)) == 1)
		{   
			SSL_free(pstSocket->stSSLInfo.pstSSL);
			pstSocket->stSSLInfo.pstSSL = NULL;
		}   
	}   

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UCSecureTCPSocket_Send(HSSLSocket hSocket, IN int nTimeout, IN char *pData, IN int nDataLen, OUT int *pnSentSize)
{
    uem_result result = ERR_UEM_UNKNOWN;
    SUCSecureSocket *pstSocket = NULL;
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
    pstSocket = (SUCSecureSocket *) hSocket;
	pstHSocket = pstSocket->hSocket;
#ifdef ARGUMENT_CHECK
    if(pstHSocket->bIsServer == TRUE)
    {
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_SOCKET, _EXIT);
    }
#endif
	if(SSL_pending(pstSocket->stSSLInfo.pstSSL) == 0)
	{
		result = selectTimeout(pstHSocket->nSocketfd, NULL, &stWriteSet, NULL, nTimeout);
		ERRIFGOTO(result, _EXIT);
	}

	nDataSent = SSL_write(pstSocket->stSSLInfo.pstSSL, pData, nDataLen);
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

uem_result UCSecureTCPSocket_Receive(HSSLSocket hSocket, IN int nTimeout, IN OUT char *pBuffer, IN int nBufferLen, OUT int *pnReceivedSize)
{
    uem_result result = ERR_UEM_UNKNOWN;
    SUCSecureSocket *pstSocket = NULL;
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
    pstSocket = (SUCSecureSocket *) hSocket;
	pstHSocket = pstSocket->hSocket;
#ifdef ARGUMENT_CHECK
    if(pstHSocket->bIsServer == TRUE)
    {
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_SOCKET, _EXIT);
    }
#endif
	if(SSL_pending(pstSocket->stSSLInfo.pstSSL) == 0)
	{
		result = selectTimeout(pstHSocket->nSocketfd, &stReadSet, NULL, NULL, nTimeout);
		ERRIFGOTO(result, _EXIT);
	}

	nDataReceived = SSL_read(pstSocket->stSSLInfo.pstSSL, pBuffer, nBufferLen);
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
