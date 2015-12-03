/*
 ============================================================================
 Name        : esim_lib.c
 Author      : Dukyoung
 Version     :
 Copyright   : CAP
 Description : Hello World in C, Ansi-style
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>

#include "esim_lib.h"

#define SYS_SLEEP   0x0
#define SYS_CYCLE   0x1
#define SYS_PID     0x2
#define SYS_SYNC    0x3

#define INTC_MASK   0x0
#define INTC_VALUE  0x4
#define INTC_WAKEUP 0x00000010

static int esim_mutex_id = 0;
static int esim_barrier_id = 0;

static int esim_prof_barrier = 0;
static int esim_prof_lock = 0;

static int esim_sem_id = 0;

static unsigned int esim_global_index = 0x10000000;

static volatile unsigned int* system_base = (volatile unsigned int*)0xE0000010;
static volatile long* LOCK_BASE = (volatile long*)0x10900000;
static volatile long* BARRIER_BASE = (volatile long*)0x10400000;
static volatile long* SEM_BASE = (volatile long*)0x10401000;

//-----------------------------------------------------------------------------

void esim_mutex_init(esim_mutex_t* mutex, int flag)
{
	*mutex = esim_mutex_id++;
}


void esim_mutex_lock(esim_mutex_t* mutex)
{
	int loop_count = 0;
	int esim_pid = esim_get_pid();
	while(1)
	{
		LOCK_BASE[*mutex] = esim_pid + 1;
		if(LOCK_BASE[*mutex]==(esim_pid+1)) break;
		else
		{
			int i=0;
			for(i=0;i<200;i++) i = i;
		}
		loop_count++;
	}
	esim_prof_lock++;
	esim_sync();
	//printf("LOOP.WAIT[%d]\n", loop_count);
}


void esim_mutex_unlock(esim_mutex_t* mutex)
{
	int esim_pid = esim_get_pid();
	LOCK_BASE[*mutex] = (esim_pid + 1);
}


//-----------------------------------------------------------------------------


void esim_barrier_init(esim_bar_t* barrier)
{
	barrier->id = esim_barrier_id;
	if(esim_get_pid()==0)
	{
		BARRIER_BASE[barrier->id] = 0;
		BARRIER_BASE[barrier->id+1] = 0;
	}
	esim_barrier_id = esim_barrier_id + 2;
	esim_mutex_init(&barrier->mutex, NULL);
}


void esim_barrier(esim_bar_t* barrier, int count)
{
	int loop_count = 0;
	int channel = barrier->id;
	printf("--BARRIER.START[%d]\n", channel);
	esim_mutex_lock(&barrier->mutex);
	{
		BARRIER_BASE[channel]++;
		if(BARRIER_BASE[channel]==count) BARRIER_BASE[channel] = 0;
	}
	esim_mutex_unlock(&barrier->mutex);
	while(BARRIER_BASE[channel]!=0)
	{
		int i;
		for(i=0;i<200;i++) i = i;
		loop_count++;
	}

	channel++;

	esim_mutex_lock(&barrier->mutex);
	{
		BARRIER_BASE[channel]++;
		if(BARRIER_BASE[channel]==count) BARRIER_BASE[channel] = 0;
	}
	esim_mutex_unlock(&barrier->mutex);
	while(BARRIER_BASE[channel]!=0)
	{
		int i;
		for(i=0;i<200;i++) i = i;
		loop_count++;
	}
	esim_prof_barrier++;

	// sync
	esim_sync();
	printf("BARRIER.WAIT[%d]\n", loop_count);
}


//-----------------------------------------------------------------------------


void esim_sem_init(esim_sem_t* sem)
{
	sem->id = esim_sem_id;
	if(esim_get_pid()==0) SEM_BASE[sem->id] = 0;

	esim_sem_id++;
	esim_mutex_init(&sem->mutex, NULL);
}

void esim_sem_post(esim_sem_t* sem)
{
	esim_mutex_lock(&sem->mutex);
	
	SEM_BASE[sem->id]++;
	printf("sem post %d %d\n", sem->id, SEM_BASE[sem->id]);
	esim_mutex_unlock(&sem->mutex);
}

void esim_sem_clear(esim_sem_t* sem)
{
	esim_mutex_lock(&sem->mutex);
	SEM_BASE[sem->id] = 0;
	printf("sem clear %d %d\n", sem->id, SEM_BASE[sem->id]);
	
	esim_mutex_unlock(&sem->mutex);
	
}

void esim_sem_wait(esim_sem_t* sem)
{
	int done = 0;
	printf("sem wait %d %d\n", sem->id, SEM_BASE[sem->id]);
	while(!done)
	{
		if(SEM_BASE[sem->id]<=0)
		{
			int i;
			for(i=0;i<200;i++) i = i;
		}
		else
		{
			esim_mutex_lock(&sem->mutex);
			if(SEM_BASE[sem->id]>0)
			{
				done = 1;
				SEM_BASE[sem->id]--;
			}
			esim_mutex_unlock(&sem->mutex);
		}
	}
	printf("sem wait out %d %d\n", sem->id, SEM_BASE[sem->id]);
	
}



//-----------------------------------------------------------------------------


void* esim_gmalloc(int size)
{
	void* buffer = (void*)esim_global_index;
	esim_global_index += size;
	printf("%x assigned\n", (unsigned)buffer);
	return buffer;
}


unsigned int esim_get_pid()
{
	return system_base[SYS_PID];
}


void esim_power_down()
{
	esim_enable_wakeup();
    system_base[SYS_SLEEP] = 0;
}


void esim_sync()
{
	system_base[SYS_SYNC] = 0;
}


void esim_prof()
{
	printf("esim_lock_count    : %d\n", esim_prof_lock);
	printf("esim_barrier_count : %d\n", esim_prof_barrier);
}

void esim_wakeup(int pid)
{
	volatile unsigned int* intc = (volatile unsigned int*)(0x30A00000 + 0x1000 * pid);
	intc[INTC_VALUE] = INTC_WAKEUP;
}

void esim_enable_wakeup()
{
	int esim_pid = esim_get_pid();
	volatile unsigned int* base_intc = (volatile unsigned int*)(0x30A00000 + esim_pid * 0x1000);
	base_intc[INTC_MASK] = base_intc[INTC_MASK] | INTC_WAKEUP;
}

void esim_clear_wakeup()
{
	int esim_pid = esim_get_pid();
	volatile unsigned int* base_intc = (volatile unsigned int*)(0x30A00000 + esim_pid * 0x1000);
	base_intc[INTC_MASK] = base_intc[INTC_MASK] & (~INTC_WAKEUP);
	base_intc[INTC_VALUE] = base_intc[INTC_VALUE] & (~INTC_WAKEUP);
}

