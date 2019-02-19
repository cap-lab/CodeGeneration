/*
 * main.ino.c
 *
 *  Created on: 2018. 8. 25.
 *      Author: jej
 */


#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <Arduino.h>

#include <uem_common.h>

//#include <UCPrint.h>

#include <uem_data.h>
#include <uem_channel_data.h>

#include <UKChannel.h>
#include <UKTaskScheduler.h>
#include <UKAddOnHandler.h>


#ifdef ARDUINO_OpenCR
 	 #define DEBUG_SERIAL                     Serial2
#elif
	#define  DEBUG_SERIAL                     Serial
#endif



void setup() {
	uem_result result;
	DEBUG_SERIAL.begin(9600);
	result = UKAddOnHandler_Init();
	if(result != ERR_UEM_NOERROR)
	{
		while(true) {
			DEBUG_SERIAL.println("ERR1: ");
			DEBUG_SERIAL.println(result);
			delay(2000);
		}
	}
	result = UKChannel_Initialize();
	if(result != ERR_UEM_NOERROR)
	{
		while(true) {
			DEBUG_SERIAL.println("ERR2: ");
			DEBUG_SERIAL.println(result);
			delay(2000);
		}
	}
	result = UKTaskScheduler_Init();
	if(result != ERR_UEM_NOERROR)
	{
		while(true) {
			DEBUG_SERIAL.println("ERR3: ");
			DEBUG_SERIAL.println(result);
			delay(2000);
		}
	}
}

void loop() {
	uem_result result;
	//DEBUG_SERIAL.print("test ");
	//DEBUG_SERIAL.print(a);
	//DEBUG_SERIAL.println();
	result = UKAddOnHandler_Run();
	if(result != ERR_UEM_NOERROR)
	{
		while(true) {
			DEBUG_SERIAL.print("ERR4: ");
			DEBUG_SERIAL.println(result);
			delay(2000);
		}
	}
	result = UKTaskScheduler_Run();
	if(result != ERR_UEM_NOERROR)
	{
		while(true) {
			DEBUG_SERIAL.println("ERR5: ");
			DEBUG_SERIAL.println(result);
			delay(2000);
		}
	}
}



