/*
 * UCSerial.h
 *
 *  Created on: 2018. 10. 23.
 *      Author: jej
 */

#ifndef SRC_COMMON_CONSTRAINED_INCLUDE_COMMUNICATION_UCSERIAL_H_
#define SRC_COMMON_CONSTRAINED_INCLUDE_COMMUNICATION_UCSERIAL_H_

#include <uem_common.h>

#include <uem_enum.h>

#ifdef __cplusplus
extern "C"
{
#endif


#ifndef DATA_SERIAL_DEFAULT_BAUD_RATE
	#define DATA_SERIAL_DEFAULT_BAUD_RATE (38400)
#endif


typedef struct _SSerialHandle *HSerial;

/**
 * @brief
 *
 * This function
 *
 * @param hSerial
 *
 * @return
 */
void UCSerial_Initialize(HSerial hSerial);

/**
 * @brief
 *
 * This function
 *
 * @param hSerial
 * @param pData
 * @param nDataLen
 * @param[out] pnSentSize
 *
 * @return
 */
uem_result UCSerial_Send(HSerial hSerial, IN char *pData, IN int nDataLen, OUT int *pnSentSize);

/**
 * @brief
 *
 * This function
 *
 * @param hSerial
 * @param pBuffer
 * @param nBufferLen
 * @param[out] pnReceivedSize
 *
 * @return
 */
uem_result UCSerial_Receive(HSerial hSerial, IN OUT char *pBuffer, IN int nBufferLen, OUT int *pnReceivedSize);

/**
 * @brief
 *
 * This function
 *
 * @param hSerial
 * @param[out] pnAvailableSize
 *
 * @return
 */
uem_result UCSerial_Available(HSerial hSerial, OUT int *pnAvailableSize);

#ifdef __cplusplus
}
#endif

#endif /* SRC_COMMON_CONSTRAINED_INCLUDE_COMMUNICATION_UCSERIAL_H_ */
