/*
 * UCSerial_data.h
 *
 *  Created on: 2018. 10. 25.
 *      Author: jej
 */



#ifndef SRC_COMMON_CONSTRAINED_INCLUDE_COMMUNICATION_UCSERIAL_DATA_HPP_
#define SRC_COMMON_CONSTRAINED_INCLUDE_COMMUNICATION_UCSERIAL_DATA_HPP_

#ifndef ARDUINO_OpenCR
	#define DATA_SERIAL SoftwareSerial
#else
	//#include <HardwareSerial.h>
    #include <USBSerial.h> //USBSerial
	#include <UARTClass.h> //Serial1~Serial4. Serial2 = SerialBT1, Serial4=SerialBT2
	#define DATA_SERIAL USBSerial
#endif

typedef struct _SSerialHandle {
	DATA_SERIAL *pclsHandle;
} SSerialHandle;

#endif /* SRC_COMMON_CONSTRAINED_INCLUDE_COMMUNICATION_UCSERIAL_DATA_HPP_ */
