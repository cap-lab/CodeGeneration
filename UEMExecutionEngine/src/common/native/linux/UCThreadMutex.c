/*
 * UCThreadMutex.c
 *
 *  Created on: 2017. 8. 5.
 *      Author: jej
 */

#include <pthread.h>

#include <UCThreadMutex.h>

typedef struct _SThreadMutex {
	int enId;
	uem_bool bInMutex;
    pthread_mutex_t hMutex;
} SThreadMutex;

