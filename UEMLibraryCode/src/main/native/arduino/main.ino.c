/*
 * main.ino.c
 *
 *  Created on: 2018. 8. 25.
 *      Author: jej
 */


#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

static int a;

#include <uem_data.h>
#include <uem_channel_data.h>

void setup() {
    a = 0;
}

void loop() {
    a++;

}
