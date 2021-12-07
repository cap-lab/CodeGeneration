/*
 * UKEncryptionHIGHT.c
 *
 *  Created on: 2021. 9. 24.
 *      Author: jrkim
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <UCBasic.h>
#include <UCAlloc.h>
#include <UCEncryptionHIGHT.h>

#include <UKVirtualEncryption.h>

#include <UKEncryptionHIGHT.h>

#define BLOCK_SIZE 8
#define ROUNDKEY_SIZE 136

typedef struct _SHIGHTKey {
	unsigned char *pszInitVec;
	unsigned char *pszRoundKey; // 136
	int nKeyLen;
} SHIGHTKey;

uem_result UKEncryptionHIGHT_Initialize(HVirtualKey *phKey, void *pstEncKeyInfo)
{
	uem_result result = ERR_UEM_UNKNOWN;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(pstEncKeyInfo, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	SHIGHTKey *pstKey;
	SEncryptionKeyInfo *pstKeyInfo = (SEncryptionKeyInfo *)pstEncKeyInfo;

	pstKey = UCAlloc_malloc(sizeof(SHIGHTKey));	
	ERRMEMGOTO(pstKey, result, _EXIT);

	pstKey->pszInitVec = NULL;
	pstKey->pszRoundKey = NULL;

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

	result = UCEncryptionHIGHT_GenerateRoundKey(pstKeyInfo->pszUserKey, pstKey->pszRoundKey);
	ERRIFGOTO(result, _EXIT);

	result = UCEncryptionHIGHT_Encode(pstKey->pszRoundKey, pstKey->pszInitVec);
	ERRIFGOTO(result, _EXIT);

	*phKey = pstKey;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKEncryptionHIGHT_EncodeOnCTRMode(HVirtualKey hKey, uem_uint8 *pData, uem_uint32 nDataLen)
{
	uem_result result = ERR_UEM_UNKNOWN;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(hKey, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif

	if(nDataLen > 0)
	{
		SHIGHTKey *pstKey = (SHIGHTKey *)hKey;

		result = UCEncryptionHIGHT_EncodeOnCTRMode((uem_uint32 *)pstKey->pszRoundKey, pstKey->pszInitVec, pData, nDataLen);
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKEncryptionHIGHT_Finalize(HVirtualKey *phKey)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SHIGHTKey *pstKey = (SHIGHTKey *)(*phKey);

	SAFEMEMFREE(pstKey->pszInitVec);
	SAFEMEMFREE(pstKey->pszRoundKey);
	SAFEMEMFREE(pstKey);

	result = ERR_UEM_NOERROR;

	return result;
}
