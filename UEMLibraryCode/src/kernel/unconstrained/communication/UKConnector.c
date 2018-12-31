/*
 * UKConnector.c
 *
 *  Created on: 2018. 10. 8.
 *      Author: chjej202
 */


#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <UCAlloc.h>

#include <UKConnector.h>

typedef struct _SConnector {
	HUserConnector hUserHandle;
	FnCallbackConnectorSend fnUserSendFunc;
	FnCallbackConnectorReceive fnUserReceiveFunc;
} SConnector;


uem_result UKConnector_Create(OUT HConnector *phConnector)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SConnector *pstConnector = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(phConnector, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	pstConnector = UCAlloc_malloc(sizeof(SConnector));
	ERRMEMGOTO(pstConnector, result, _EXIT);

	pstConnector->hUserHandle = NULL;
	pstConnector->fnUserReceiveFunc = NULL;
	pstConnector->fnUserSendFunc = NULL;

	*phConnector = pstConnector;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKConnector_Destroy(IN OUT HConnector *phConnector)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SConnector *pstConnector = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(phConnector, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	IFVARERRASSIGNGOTO(*phConnector, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif

	pstConnector = (SConnector *) *phConnector;

	SAFEMEMFREE(pstConnector);

	*phConnector = NULL;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKConnector_SetConnector(HConnector hConnector, HUserConnector hUserHandle, FnCallbackConnectorSend fnSend, FnCallbackConnectorReceive fnReceive)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SConnector *pstConnector = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(hConnector, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	IFVARERRASSIGNGOTO(hUserHandle, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	IFVARERRASSIGNGOTO(fnReceive, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	IFVARERRASSIGNGOTO(fnSend, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif

	pstConnector = (SConnector *) hConnector;

	pstConnector->hUserHandle = hUserHandle;
	pstConnector->fnUserReceiveFunc = fnReceive;
	pstConnector->fnUserSendFunc = fnSend;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKConnector_Send(HConnector hConnector, IN int nTimeout, IN char *pData, IN int nDataLen, OUT int *pnSentSize)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SConnector *pstConnector = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(hConnector, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	pstConnector = (SConnector *) hConnector;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(pstConnector->hUserHandle, NULL, result, ERR_UEM_NO_DATA, _EXIT);
	IFVARERRASSIGNGOTO(pstConnector->fnUserSendFunc, NULL, result, ERR_UEM_NO_DATA, _EXIT);
#endif

	result = pstConnector->fnUserSendFunc(hConnector, pstConnector->hUserHandle, nTimeout, pData, nDataLen, pnSentSize);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKConnector_Receive(HConnector hConnector, IN int nTimeout, IN OUT char *pBuffer, IN int nBufferLen, OUT int *pnReceivedSize)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SConnector *pstConnector = NULL;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(hConnector, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	pstConnector = (SConnector *) hConnector;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(pstConnector->hUserHandle, NULL, result, ERR_UEM_NO_DATA, _EXIT);
	IFVARERRASSIGNGOTO(pstConnector->fnUserReceiveFunc, NULL, result, ERR_UEM_NO_DATA, _EXIT);
#endif

	result = pstConnector->fnUserReceiveFunc(hConnector, pstConnector->hUserHandle, nTimeout, pBuffer, nBufferLen, pnReceivedSize);
	ERRIFGOTO(result, _EXIT);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}





