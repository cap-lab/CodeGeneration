#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#include <stdarg.h>

#include <pthread.h>

//#define NDEBUG
#include <assert.h>

#include "CIC_SPEs.h"


// each bit represent locking/unlocking status of each SPE
static CIC_SPEs_status spe_status = 0x0;

static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t cond = PTHREAD_COND_INITIALIZER;

CIC_SPEs_status CIC_SPEs_make_request(int count, ...)
{
    int i;
    unsigned int shift;
    CIC_SPEs_status ret = 0x0;

    va_list ap;
    va_start(ap, count);

    for(i=0;i<count;i++)
    {
        shift = va_arg(ap, unsigned int);
        if(shift >= NUM_SPE)
        {
            printf("CIC_SPEs_make_request(): too high processor number(%d)!\n", shift);
        }
        else
        {
            ret = ret | (0x1 << shift);
        }
    }

    va_end(ap);

    return ret;
}

bool CIC_SPEs_lock(CIC_SPEs_status request)
{
restart_lock:
    pthread_mutex_lock(&mutex);

    if((spe_status & request) == 0x0)
    {
        spe_status = spe_status | request;
    }
    else
    {
        pthread_cond_wait(&cond, &mutex);
        pthread_mutex_unlock(&mutex);
        goto restart_lock;
    }

    pthread_mutex_unlock(&mutex);

    return true;
}

bool CIC_SPEs_unlock(CIC_SPEs_status request)
{
    pthread_mutex_lock(&mutex);

    if((spe_status & request) != request)
    {
        printf("CIC_SPEs_unlock(): argument error(0x%x)!\n", request);
    }

    spe_status = spe_status & (~request);

    pthread_cond_broadcast(&cond);

    pthread_mutex_unlock(&mutex);

    return true;
}

