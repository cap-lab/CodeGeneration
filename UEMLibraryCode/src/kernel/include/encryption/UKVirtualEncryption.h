/*
 * UKVirtualEncryption.h
 *
 *  Created on: 2021. 9. 2.
 *      Author: jrkim
 */

#ifndef SRC_KERNEL_UNCONSTRAINED_INCLUDE_ENCRYPTION_UKVIRTUALENCRYPTION_H_
#define SRC_KERNEL_UNCONSTRAINED_INCLUDE_ENCRYPTION_UKVIRTUALENCRYPTION_H_

#include <uem_common.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef void *HVirtualKey;

typedef uem_result (*FnVirtualEncryptionInitialize)(HVirtualKey *phKey, void *pstKeyInfo);
typedef uem_result (*FnVirtualEncryptionEncode)(HVirtualKey hKey, uem_uint8 *pData, uem_uint32 nDataLen);
typedef uem_result (*FnVirtualEncryptionDecode)(HVirtualKey hKey, uem_uint8 *pData, uem_uint32 nDataLen);
typedef uem_result (*FnVirtualEncryptionFinalize)(HVirtualKey *phKey);

typedef struct _SVirtualEncryptionAPI {
	FnVirtualEncryptionInitialize fnInitialize;
	FnVirtualEncryptionEncode fnEncode;
	FnVirtualEncryptionDecode fnDecode;
	FnVirtualEncryptionFinalize fnFinalize;
} SVirtualEncryptionAPI;

typedef struct _SEncryptionKeyInfo {
	unsigned char *pszUserKey;
	unsigned char *pszInitVec;
	int nKeyLen;
	SVirtualEncryptionAPI *pstEncAPI;
} SEncryptionKeyInfo;

#ifdef __cplusplus
}
#endif


#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_ENCRYPTION_UKVIRTUALENCRYPTION_H_ */
