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
#include <UCEncryptionLEA.h>

#include <UKVirtualEncryption.h>

#include <UKEncryptionLEA.h>

#define BYTE_TO_BITS 8
#define BLOCK_SIZE 16
#define ROUND_NUM 24
#define ROUNDKEY_SIZE 384 // ROUND_NUM * KEY_LEN

typedef struct _SLEAKey {
	unsigned char pszInitVec[BLOCK_SIZE];
	unsigned char pszRoundKey[ROUNDKEY_SIZE];
	int nKeyLen;
} SLEAKey;

static SLEAKey s_stEncKey = {
	{0, },
	{0, },
	BLOCK_SIZE
};

uem_result UKEncryptionLEA_Initialize(HVirtualKey *phKey, void *pstEncKeyInfo)
{
	uem_result result = ERR_UEM_UNKNOWN;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(pstEncKeyInfo, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	SEncryptionKeyInfo *pstKeyInfo = (SEncryptionKeyInfo *)pstEncKeyInfo;
	uem_uint32 nRoundNum = ROUND_NUM;

	UC_memcpy(s_stEncKey.pszInitVec, pstKeyInfo->pszInitVec, sizeof(uem_uint8) * BLOCK_SIZE);

	result = UCEncryptionLEA_GenerateRoundKey(pstKeyInfo->pszUserKey, s_stEncKey.pszRoundKey, nRoundNum);
	ERRIFGOTO(result, _EXIT);

	result = UCEncryptionLEA_Encode(s_stEncKey.pszRoundKey, s_stEncKey.pszInitVec, nRoundNum);
	ERRIFGOTO(result, _EXIT);

	*phKey = &(s_stEncKey);

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UKEncryptionLEA_EncodeOnCTRMode(HVirtualKey hKey, uem_uint8 *pData, uem_uint32 nDataLen)
{
	uem_result result = ERR_UEM_UNKNOWN;
	uem_uint32 nRoundNum = ROUND_NUM;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(hKey, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif

	if(nDataLen > 0)
	{
		SLEAKey *pstKey = (SLEAKey *)hKey;

		result = UCEncryptionLEA_EncodeOnCTRMode((uem_uint32 *)pstKey->pszRoundKey, pstKey->pszInitVec, pData, nDataLen, nRoundNum);
		ERRIFGOTO(result, _EXIT);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}
