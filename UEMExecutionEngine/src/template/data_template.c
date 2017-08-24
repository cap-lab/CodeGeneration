/*
 * data_template.c
 *
 *  Created on: 2017. 8. 10.
 *      Author: jej
 */

#include <pthread.h>

#include <UCThread.h>

#include <UCThreadEvent.h>

#include <UCThreadMutex.h>


SThreadEvent thread_events_data[] = {
	{ID_UEM_THREAD_EVENT, FALSE, TRUE, PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER},
};

HThreadEvent thread_events[] = {
	&thread_events_data[0],
};

HThreadEvent *g_ahStaticThreadEvents = thread_events;


SThreadMutex thread_mutexes_data[] = {
	{ID_UEM_THREAD_MUTEX, FALSE, TRUE, PTHREAD_MUTEX_INITIALIZER},
};

HThreadMutex thread_mutexes[] = {
	&thread_mutexes_data[0],
};

HThreadMutex *g_ahStaticThreadMutexes = thread_mutexes;

//thread_events[0],
//thread_events[1],
//thread_events[2],


