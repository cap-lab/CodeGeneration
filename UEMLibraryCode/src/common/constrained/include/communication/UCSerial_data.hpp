/*
 * UCSerial_data.h
 *
 *  Created on: 2018. 10. 25.
 *      Author: jej
 */



#ifndef SRC_COMMON_CONSTRAINED_INCLUDE_COMMUNICATION_UCSERIAL_DATA_HPP_
#define SRC_COMMON_CONSTRAINED_INCLUDE_COMMUNICATION_UCSERIAL_DATA_HPP_

#include <Stream.h>

#ifdef ARDUINO_OpenCR
	//#include <HardwareSerial.h>
    #include <USBSerial.h> //USBSerial
	#include <UARTClass.h> //Serial1~Serial4. Serial2 = SerialBT1, Serial4=SerialBT2
#endif

#define DATA_SERIAL Stream

typedef struct _SSerialHandle SSerialHandle;

typedef uem_result (*FnSerialInitialize)(SSerialHandle *pstSerialHandle);

typedef struct _SSerialHandle {
	DATA_SERIAL *pclsHandle;
	FnSerialInitialize fnInitialize;
} SSerialHandle;

#ifdef __cplusplus
extern "C"
{
#endif

void HardwareSerial_Initialize(SSerialHandle *pstSerialHandle);

#ifdef ARDUINO_OpenCR
void USBSerial_Initialize(SSerialHandle *pstSerialHandle);
#else
void SoftwareSerial_Initialize(SSerialHandle *pstSerialHandle);
#endif


#ifdef __cplusplus
}
#endif

#endif /* SRC_COMMON_CONSTRAINED_INCLUDE_COMMUNICATION_UCSERIAL_DATA_HPP_ */
