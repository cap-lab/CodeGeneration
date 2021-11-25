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
#include <UCEncryptionSEED.h>

#include <UKVirtualEncryption.h>

#include <UKEncryptionSEED.h>

#define BLOCK_SIZE 16
#define KEY_SIZE 16
#define ROUNDKEY_SIZE 128

typedef struct _SSEEDKey {
	unsigned char pszInitVec[BLOCK_SIZE];
	unsigned char pszRoundKey[ROUNDKEY_SIZE];
	int nKeyLen;
} SSEEDKey;

static SSEEDKey s_stEncKey = {
	{0, },
	{0, },
	KEY_SIZE
};

uem_result UKEncryptionSEED_Initialize(HVirtualKey *phKey, void *pstEncKeyInfo)
{
	uem_result result = ERR_UEM_UNKNOWN;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(pstEncKeyInfo, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif
	SEncryptionKeyInfo *pstKeyInfo = (SEncryptionKeyInfo *)pstEncKeyInfo;

	UC_memcpy(s_stEncKey.pszInitVec, pstKeyInfo->pszInitVec, sizeof(uem_uint8) * BLOCK_SIZE);

	result = UCEncryptionSEED_GenerateRoundKey(pstKeyInfo->pszUserKey, s_stEncKey.pszRoundKey);
	ERRIFGOTO(result, _EXIT);

	result = UCEncryptionSEED_Encode(s_stEncKey.pszRoundKey, s_stEncKey.pszInitVec);
	ERRIFGOTO(result, _EXIT);

	*phKey = &(s_stEncKey);

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
