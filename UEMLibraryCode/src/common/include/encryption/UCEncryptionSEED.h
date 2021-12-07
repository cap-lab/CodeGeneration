/*
 * UCEncryptionSEED.h
 *
 *  Created on: 2021. 9. 27.
 *      Author: jrkim
 */

#ifndef SRC_COMMON_INCLUDE_UCENCRYPTIONSEED_H_
#define SRC_COMMON_INCLUDE_UCENCRYPTIONSEED_H_

#include <uem_common.h>

#ifdef __cplusplus
extern "C"
{
#endif

uem_result UCEncryptionSEED_GenerateRoundKey(uem_uint8* userkey, uem_uint8* roundkey);
uem_result UCEncryptionSEED_Encode(uem_uint8* roundkey, uem_uint8* data);
uem_result UCEncryptionSEED_Decode(uem_uint8* roundkey, uem_uint8* data);
uem_result UCEncryptionSEED_EncodeOnCTRMode(uem_uint32 *roundkey, uem_uint8 *iv, uem_uint8 *pData, uem_uint32 nDataLen);

#ifdef __cplusplus
}
#endif

#endif /* SRC_COMMON_INCLUDE_UCENCRYPTIONSEED_H_ */
