/*
 * UCEndian.h
 *
 *  Created on: 2018. 10. 6.
 *      Author: chjej202
 */

#ifndef SRC_COMMON_UNCONSTRAINED_INCLUDE_COMMUNICATION_UCENDIAN_H_
#define SRC_COMMON_UNCONSTRAINED_INCLUDE_COMMUNICATION_UCENDIAN_H_

#include <uem_common.h>

#ifdef __cplusplus
extern "C"
{
#endif


uem_bool UCEndian_SystemIntToLittleEndianChar(int nValue, char *pBuffer, int nBufferLen);
uem_bool UCEndian_SystemShortToLittleEndianChar(short sValue, char *pBuffer, int nBufferLen);
uem_bool UCEndian_LittleEndianCharToSystemInt(char *pBuffer, int nBufferLen, int *pnValue);
uem_bool UCEndian_LittleEndianCharToSystemShort(char *pBuffer, int nBufferLen, short *psValue);

#ifdef __cplusplus
}
#endif

#endif /* SRC_COMMON_UNCONSTRAINED_INCLUDE_COMMUNICATION_UCENDIAN_H_ */
