/*
 * UKEncryptionSEED.h
 *
 *  Created on: 2021. 9. 27.
 *      Author: jrkim
 */

#ifndef SRC_KERNEL_UNCONSTRAINED_INCLUDE_ENCRYPTION_UKENCRYPTIONSEED_H_
#define SRC_KERNEL_UNCONSTRAINED_INCLUDE_ENCRYPTION_UKENCRYPTIONSEED_H_

#include <uem_common.h>

#include <UKVirtualEncryption.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief initialize an encryption.
 *
 * This function initialize roundkey and init vec.
 *
 * @param phKey a key to be used in the encryption.
 * @param pstEncKeyInfo a encryption key info.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKEncryptionSEED_Initialize(HVirtualKey *phKey, void *pstKeyInfo);

/**
 * @brief encode/decode data on CTR mode.
 *
 * This function encode/decode the data.
 *
 * @param hKey a key to be used in the encryption.
 * @param pData a data to be encrypted or decrypted.
 * @param nDataLen a data length.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKEncryptionSEED_EncodeOnCTRMode(HVirtualKey hKey, uem_uint8 *pData, uem_uint32 nDataLen);

/**
 * @brief finalize an encryption.
 *
 * This function deallocate space.
 *
 * @param phKey a key used in the encryption.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * Errors to be returned - .
 */
uem_result UKEncryptionSEED_Finalize(HVirtualKey *phKey);

#ifdef __cplusplus
}
#endif


#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_ENCRYPTION_UKENCRYPTIONSEED_H_ */
