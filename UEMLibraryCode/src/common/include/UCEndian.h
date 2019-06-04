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


/**
 * @brief
 *
 * This function
 *
 * @param nValue
 * @param pBuffer
 * @param nBufferLen
 *
 * @return
 */
uem_bool UCEndian_SystemIntToLittleEndianChar(int nValue, IN OUT char *pBuffer, int nBufferLen);

/**
 * @brief
 *
 * This function
 *
 * @param sValue
 * @param pBuffer
 * @param nBufferLen
 *
 * @return
 */
uem_bool UCEndian_SystemShortToLittleEndianChar(short sValue, IN OUT char *pBuffer, int nBufferLen);

/**
 * @brief
 *
 * This function
 *
 * @param pBuffer
 * @param nBufferLen
 * @param[out] pnValue
 *
 * @return
 */
uem_bool UCEndian_LittleEndianCharToSystemInt(char *pBuffer, int nBufferLen, OUT int *pnValue);

/**
 * @brief
 *
 * This function
 *
 * @param pBuffer
 * @param nBufferLen
 * @param[out] psValue
 *
 * @return
 */
uem_bool UCEndian_LittleEndianCharToSystemShort(char *pBuffer, int nBufferLen, OUT short *psValue);

#ifdef __cplusplus
}
#endif

#endif /* SRC_COMMON_UNCONSTRAINED_INCLUDE_COMMUNICATION_UCENDIAN_H_ */
