/*
 * UKEncryptionLEA.c
 *
 *  Created on: 2021. 9. 2.
 *      Author: jrkim
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <UCBasic.h>
#include <UCAlloc.h>
#include <UCEncryptionLEA.h>

#include <UKVirtualEncryption.h>

#include <UKEncryptionLEA.h>

#define BYTE_TO_BITS 8

typedef enum _EKeyLen {
	KEYLEN_TYPE_128 = 128,
	KEYLEN_TYPE_192 = 192,
	KEYLEN_TYPE_256 = 256,
} EKeyLen;

typedef struct _SLEAKey {
	unsigned char *pszInitVec;
	unsigned char *pszRoundKey; // 24 * 16, 28 * 24, 32 * 32
	int nKeyLen;
} SLEAKey;


static uem_uint32 getRoundNum(uem_uint8 nKeyLen)
{
	uem_uint32 nRoundNum = 0;

	switch(nKeyLen * BYTE_TO_BITS)
	{
	case KEYLEN_TYPE_128:
		nRoundNum = 24;
		break;
	case KEYLEN_TYPE_192:
		nRoundNum = 28;
		break;
	case KEYLEN_TYPE_256:
		nRoundNum = 32;
		break;
	default:
		nRoundNum = 0;
		break;
	}

	return nRoundNum;
}

uem_result UKEncryptionLEA_Initialize(HVirtualKey *phKey, void *pstEncKeyInfo)
{
	uem_result result = ERR_UEM_UNKNOWN;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(pstEncKeyInfo, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	SLEAKey *pstKey;
	SEncryptionKeyInfo *pstKeyInfo = (SEncryptionKeyInfo *)pstEncKeyInfo;
	uem_uint32 nRoundNum = 0;

	pstKey = UCAlloc_malloc(sizeof(SLEAKey));	
	ERRMEMGOTO(pstKey, result, _EXIT);

	nRoundNum = getRoundNum(pstKeyInfo->nKeyLen);

	if(pstKey->pszInitVec == NULL) {
		pstKey->pszInitVec = UCAlloc_malloc(sizeof(unsigned char) * pstKeyInfo->nKeyLen);		
		ERRMEMGOTO(pstKey->pszInitVec, result, _EXIT);

		UC_memcpy(pstKey->pszInitVec, pstKeyInfo->pszInitVec, pstKeyInfo->nKeyLen);
	}
	if(pstKey->pszRoundKey == NULL) {
		pstKey->pszRoundKey = UCAlloc_malloc(sizeof(unsigned char) * pstKeyInfo->nKeyLen * nRoundNum);	
		ERRMEMGOTO(pstKey->pszRoundKey, result, _EXIT);
	}
	pstKey->nKeyLen = pstKeyInfo->nKeyLen;

	result = UCEncryptionLEA_GenerateRoundKey(pstKeyInfo->pszUserKey, pstKey->pszRoundKey, nRoundNum);
	ERRIFGOTO(result, _EXIT);

	result = UCEncryptionLEA_Encode(pstKey->pszRoundKey, pstKey->pszInitVec, nRoundNum);
	ERRIFGOTO(result, _EXIT);

	*phKey = pstKey;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKEncryptionLEA_EncodeOnCTRMode(HVirtualKey hKey, uem_uint8 *pData, uem_uint32 nDataLen)
{
	uem_result result = ERR_UEM_UNKNOWN;
	uem_uint32 nRoundNum = 0;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(hKey, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif

	if(nDataLen > 0)
	{
		SLEAKey *pstKey = (SLEAKey *)hKey;

		nRoundNum = getRoundNum(pstKey->nKeyLen);

		result = UCEncryptionLEA_EncodeOnCTRMode((uem_uint32 *)pstKey->pszRoundKey, pstKey->pszInitVec, pData, nDataLen, nRoundNum);
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKEncryptionLEA_Finalize(HVirtualKey *phKey)
{
	uem_result result = ERR_UEM_UNKNOWN;
	SLEAKey *pstKey = (SLEAKey *)(*phKey);

	SAFEMEMFREE(pstKey->pszInitVec);
	SAFEMEMFREE(pstKey->pszRoundKey);
	SAFEMEMFREE(pstKey);

	result = ERR_UEM_NOERROR;

	return result;
}
