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

typedef struct _SSerialHandle *HSerial;

void UCSerial_Initialize(HSerial hSerial);
uem_result UCSerial_Send(HSerial hSerial, IN char *pData, IN int nDataLen, OUT int *pnSentSize);
uem_result UCSerial_Receive(HSerial hSerial, IN OUT char *pBuffer, IN int nBufferLen, OUT int *pnReceivedSize);

#ifdef __cplusplus
}
#endif

#endif /* SRC_COMMON_CONSTRAINED_INCLUDE_COMMUNICATION_UCSERIAL_H_ */
