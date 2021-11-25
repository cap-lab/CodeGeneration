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
#include <UCEncryptionHIGHT.h>

#include <UKVirtualEncryption.h>

#include <UKEncryptionHIGHT.h>

#define BLOCK_SIZE 8
#define KEY_SIZE 16
#define ROUNDKEY_SIZE 136

typedef struct _SHIGHTKey {
	unsigned char pszInitVec[BLOCK_SIZE];
	unsigned char pszRoundKey[ROUNDKEY_SIZE]; // 136
	int nKeyLen;
} SHIGHTKey;

static SHIGHTKey s_stEncKey = {
	{0, },
	{0, },
	KEY_SIZE
};

uem_result UKEncryptionHIGHT_Initialize(HVirtualKey *phKey, void *pstEncKeyInfo)
{
	uem_result result = ERR_UEM_UNKNOWN;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(pstEncKeyInfo, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	SEncryptionKeyInfo *pstKeyInfo = (SEncryptionKeyInfo *)pstEncKeyInfo;

	UC_memcpy(s_stEncKey.pszInitVec, pstKeyInfo->pszInitVec, sizeof(uem_uint8) * BLOCK_SIZE);

	result = UCEncryptionHIGHT_GenerateRoundKey(pstKeyInfo->pszUserKey, s_stEncKey.pszRoundKey);
	ERRIFGOTO(result, _EXIT);

	result = UCEncryptionHIGHT_Encode(s_stEncKey.pszRoundKey, s_stEncKey.pszInitVec);
	ERRIFGOTO(result, _EXIT);

	*phKey = &(s_stEncKey);

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
