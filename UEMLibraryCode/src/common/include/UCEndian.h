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
 * @brief Change endianness from system-endian integer to little-endian byte order.
 *
 * This function stores an integer value to buffer with little-endian byte order. \n
 * @a nValue can be both little or big endian depending on what kind of system is used.
 *
 * @param nValue an integer value to be stored on the buffer.
 * @param[in,out] pBuffer a buffer to store an integer value with little-endian order.
 * @param nBufferLen size of the buffer.
 *
 * @return It returns TRUE if the conversion is successfully done. Otherwise, it returns FALSE.
 */
uem_bool UCEndian_SystemIntToLittleEndianChar(int nValue, IN OUT char *pBuffer, int nBufferLen);

/**
 * @brief Change endianness from system-endian short to little-endian byte order.
 *
 * This function stores a short value to buffer with little-endian byte order. \n
 * @a sValue can be both little or big endian depending on what kind of system is used.
 *
 * @param sValue a short value to be stored on the buffer.
 * @param[in,out] pBuffer a buffer to store a short value with little-endian order.
 * @param nBufferLen size of the buffer.
 *
 * @return It returns TRUE if the conversion is successfully done. Otherwise, it returns FALSE.
 */
uem_bool UCEndian_SystemShortToLittleEndianChar(short sValue, IN OUT char *pBuffer, int nBufferLen);

/**
 * @brief Change endianness from little-endian byte order to system-endian integer.
 *
 * This function gets an integer value from the buffer with little-endian order.
 *
 * @param pBuffer a buffer to get an integer value.
 * @param nBufferLen size of the buffer.
 * @param[out] pnValue an integer value retrieved from the buffer.
 *
 * @return It returns TRUE if the conversion is successfully done. Otherwise, it returns FALSE.
 */
uem_bool UCEndian_LittleEndianCharToSystemInt(char *pBuffer, int nBufferLen, OUT int *pnValue);

/**
 * @brief Change endianness from little-endian byte order to system-endian short.
 *
 * This function gets a short value from the buffer with little-endian order.
 *
 * @param pBuffer a buffer to get a short value.
 * @param nBufferLen size of the buffer.
 * @param[out] psValue a short value retrieved from the buffer.
 *
 * @return It returns TRUE if the conversion is successfully done. Otherwise, it returns FALSE.
 */
uem_bool UCEndian_LittleEndianCharToSystemShort(char *pBuffer, int nBufferLen, OUT short *psValue);

#ifdef __cplusplus
}
#endif

#endif /* SRC_COMMON_UNCONSTRAINED_INCLUDE_COMMUNICATION_UCENDIAN_H_ */
