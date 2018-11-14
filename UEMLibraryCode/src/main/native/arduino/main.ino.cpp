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

void setup() {
	uem_result result;
	Serial.begin(9600);
	result = UKAddOnHandler_Init();
	if(result != ERR_UEM_NOERROR)
	{
		while(true) {
			Serial.println("ERR1: ");
			Serial.println(result);
			delay(2000);
		}
	}
	result = UKChannel_Initialize();
	if(result != ERR_UEM_NOERROR)
	{
		while(true) {
			Serial.println("ERR2: ");
			Serial.println(result);
			delay(2000);
		}
	}
	result = UKTaskScheduler_Init();
	if(result != ERR_UEM_NOERROR)
	{
		while(true) {
			Serial.println("ERR3: ");
			Serial.println(result);
			delay(2000);
		}
	}
}

void loop() {
	uem_result result;
	//Serial.print("test ");
	//Serial.print(a);
	//Serial.println();
	result = UKAddOnHandler_Run();
	if(result != ERR_UEM_NOERROR)
	{
		while(true) {
			Serial.print("ERR4: ");
			Serial.println(result);
			delay(2000);
		}
	}
    result = UKTaskScheduler_Run();
	if(result != ERR_UEM_NOERROR)
	{
		while(true) {
			Serial.println("ERR5: ");
			Serial.println(result);
			delay(2000);
		}
	}
}


