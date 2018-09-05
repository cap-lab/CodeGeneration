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

static int a;

void setup() {
	int nLoop = 0;
    a = 0;
	Serial.begin(9600);
	UKChannel_Initialize();

	for(nLoop = 0 ; nLoop < 4;nLoop++)
	{
		g_astTaskIdToTask[nLoop].pstTask->stTaskFunctions.fnInit(g_astTaskIdToTask[nLoop].pstTask->nTaskId);
	}
}

void loop() {
    a++;
    //UEM_DEBUG_PRINT("test %d\n", a);
	//Serial.print("test ");
	//Serial.print(a);
	//Serial.println();
    g_astScheduledTaskList[0].fnCompositeGo(-1);
	delay(2000);
}


