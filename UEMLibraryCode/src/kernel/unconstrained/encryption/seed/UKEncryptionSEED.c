/*
 * UKEncryptionSEED.c
 *
 *  Created on: 2021. 9. 27.
 *      Author: jrkim
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <UCBasic.h>
#include <UCAlloc.h>
#include <UCEncryptionSEED.h>

#include <UKVirtualEncryption.h>

#include <UKEncryptionSEED.h>

#define BLOCK_SIZE 16
#define ROUNDKEY_SIZE 128

typedef struct _SSEEDKey {
	unsigned char *pszInitVec;
	unsigned char *pszRoundKey; // 128
	int nKeyLen;
} SSEEDKey;

uem_result UKEncryptionSEED_Initialize(HVirtualKey *phKey, void *pstEncKeyInfo)
{
	uem_result result = ERR_UEM_UNKNOWN;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(pstEncKeyInfo, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	SSEEDKey *pstKey;
	SEncryptionKeyInfo *pstKeyInfo = (SEncryptionKeyInfo *)pstEncKeyInfo;

	pstKey = UCAlloc_malloc(sizeof(SSEEDKey));	
	ERRMEMGOTO(pstKey, result, _EXIT);

	if(pstKey->pszInitVec == NULL) {
		pstKey->pszInitVec = UCAlloc_malloc(sizeof(unsigned char) * BLOCK_SIZE);		
		ERRMEMGOTO(pstKey->pszInitVec, result, _EXIT);

		UC_memcpy(pstKey->pszInitVec, pstKeyInfo->pszInitVec, BLOCK_SIZE);
	}
	if(pstKey->pszRoundKey == NULL) {
		pstKey->pszRoundKey = UCAlloc_malloc(sizeof(unsigned char) * ROUNDKEY_SIZE);	
		ERRMEMGOTO(pstKey->pszRoundKey, result, _EXIT);
	}
	pstKey->nKeyLen = pstKeyInfo->nKeyLen;

	result = UCEncryptionSEED_GenerateRoundKey(pstKeyInfo->pszUserKey, pstKey->pszRoundKey);
	ERRIFGOTO(result, _EXIT);

	result = UCEncryptionSEED_Encode(pstKey->pszRoundKey, pstKey->pszInitVec);
	ERRIFGOTO(result, _EXIT);

	*phKey = pstKey;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKEncryptionSEED_EncodeOnCTRMode(HVirtualKey hKey, uem_uint8 *pData, uem_uint32 nDataLen)
{
	uem_result result = ERR_UEM_UNKNOWN;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(hKey, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif

	if(nDataLen > 0)
	{
		SSEEDKey *pstKey = (SSEEDKey *)hKey;

		result = UCEncryptionSEED_EncodeOnCTRMode((uem_uint32 *)pstKey->pszRoundKey, pstKey->pszInitVec, pData, nDataLen);
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKEncryptionSEED_Finalize(HVirtualKey *phKey)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SSEEDKey *pstKey = (SSEEDKey *)(*phKey);

	SAFEMEMFREE(pstKey->pszInitVec);
	SAFEMEMFREE(pstKey->pszRoundKey);
	SAFEMEMFREE(pstKey);

	result = ERR_UEM_NOERROR;

	return result;
}
