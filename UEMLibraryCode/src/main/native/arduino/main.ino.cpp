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

#include <UCPrint.h>

#include <uem_data.h>
#include <uem_channel_data.h>

static int a;

void setup() {
    a = 0;
	Serial.begin(9600);
	Serial.println("merong");
}

void loop() {
    a++;
    UEM_DEBUG_PRINT("test %d\n", a);
	delay(2000);
}


