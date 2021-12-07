/*
 * UCEncryptionLEA.h
 *
 *  Created on: 2021. 9. 2.
 *      Author: jrkim
 */

#ifndef SRC_COMMON_INCLUDE_UCENCRYPTIONLEA_H_
#define SRC_COMMON_INCLUDE_UCENCRYPTIONLEA_H_

#include <uem_common.h>

#ifdef __cplusplus
extern "C"
{
#endif

uem_result UCEncryptionLEA_GenerateRoundKey(uem_uint8* userkey, uem_uint8* roundkey, uem_uint8 nRoundNum);
uem_result UCEncryptionLEA_Encode(uem_uint8* roundkey, uem_uint8* data, uem_uint32 nRoundNum);
uem_result UCEncryptionLEA_Decode(uem_uint8* roundkey, uem_uint8* data, uem_uint32 nRoundNum);
uem_result UCEncryptionLEA_EncodeOnCTRMode(uem_uint32 *roundkey, uem_uint8 *iv, uem_uint8 *pData, uem_uint32 nDataLen, uem_uint32 nRoundNum);

#ifdef __cplusplus
}
#endif

#endif /* SRC_COMMON_INCLUDE_UCENCRYPTIONLEA_H_ */
