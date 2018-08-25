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

void setup() {
    a = 0;
}

void loop() {
    a++;

}
