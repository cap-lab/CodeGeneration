/*
 * esim_lib.h
 *
 *  Created on: 2011. 6. 14.
 *      Author: Dukyoung
 */

#ifndef ESIM_LIB_H_
#define ESIM_LIB_H_


//-------------------------------------------
// LOCK
//-------------------------------------------
typedef int esim_mutex_t;

void esim_mutex_init(esim_mutex_t*, int);
void esim_mutex_unlock(esim_mutex_t*);
void esim_mutex_lock(esim_mutex_t*);


//-------------------------------------------
// BARRIER
//-------------------------------------------
typedef struct esim_bar_t {
	int id;
	esim_mutex_t mutex;
} esim_bar_t;

void esim_barrier_init(esim_bar_t*);
void esim_barrier(esim_bar_t*, int number);


//-------------------------------------------
// MEMORY ALLOC
//-------------------------------------------
void* esim_gmalloc(int size);


unsigned int esim_get_pid(void);
void esim_power_down(void);
void esim_prof(void);
void esim_wakeup(int id);
void esim_sync(void);
void esim_enable_wakeup(void);


//-------------------------------------------
// SEMAPHORE
//-------------------------------------------
typedef struct esim_sem_t {
	int id;
	esim_mutex_t mutex;
} esim_sem_t;

void esim_sem_init(esim_sem_t*);
void esim_sem_clear(esim_sem_t*);
void esim_sem_set(esim_sem_t*);
void esim_sem_wait(esim_sem_t*);


#endif /* ESIM_LIB_H_ */
