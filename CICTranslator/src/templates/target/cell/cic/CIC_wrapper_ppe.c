#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#include <pthread.h>

//#define NDEBUG
#include <assert.h>

#include <libspe2.h>
#include <mars/task.h>

#include "CIC_CMD.h"
#include "CIC_wrapper.h"

#include "CIC_port.h"

#define PROC_DEBUG (1)

#define ARRAYLEN(ARR) (sizeof(ARR)/sizeof((ARR)[0]))
#define ROUNDUP16(arg) ((arg+15)&~0xf)


extern struct spe_program_handle SPE_PROG_NAME;

static struct mars_task_id task_id[NUM_SPE_TO_USE];
static struct mars_task_args task_args;

static uint64_t cic_channel_info_queue[NUM_SPE_TO_USE];

static uint64_t cic_channel_index_queue[NUM_SPE_TO_USE];















void FUNC_INIT(int t_id)
{
    int ret;
    int i, j;

    //printf("size: %d, %d\n", sizeof(cic_channel_info_ppe), sizeof(cic_channel_info_ppe[0]));

    for(i=0;i<NUM_SPE_TO_USE;i++)
    {
        ret = mars_task_create_with_affinity(mars_ctx, &task_id[i], 0, TASKNAME, SPE_PROG_NAME.elf_image, MARS_TASK_CONTEXT_SAVE_SIZE_MAX);
#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
        if(ret)
        {
            printf("MARS task create failed! (%d)\n", ret);
            exit(EXIT_FAILURE);
        }
#endif
    }

    for(j=0;j<NUM_SPE_TO_USE;j++)
    {
        ret = mars_task_queue_create(mars_ctx, &cic_channel_info_queue[j], sizeof(uint64_t)+sizeof(int32_t)+sizeof(uint32_t), ARRAYLEN(cic_channel_info_ppe), MARS_TASK_QUEUE_HOST_TO_MPU);
#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
        if(ret)
        {
            printf("MARS task queue create failed! (%d) at line %d\n", ret, __LINE__);
            exit(EXIT_FAILURE);
        }
#endif
        for(i=0;i<ARRAYLEN(cic_channel_info_ppe);i++)
        {
            ret = mars_task_queue_create(mars_ctx, &(cic_channel_info_ppe[i][j].queue_ea), ROUNDUP16(cic_channel_info_ppe[i][j].size), cic_channel_info_ppe[i][j].depth, cic_channel_info_ppe[i][j].direction);
#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
            if(ret)
            {
                printf("MARS task queue create failed! (%d) at line %d\n", ret, __LINE__);
                exit(EXIT_FAILURE);
            }
#endif
            ret = mars_task_queue_push(cic_channel_info_queue[j], &cic_channel_info_ppe[i][j]);
#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
            if(ret)
            {
                printf("MARS task queue push failed! (%d) at line %d\n", ret, __LINE__);
                exit(EXIT_FAILURE);
            }
#endif
        }
    }

    for(i=0;i<NUM_SPE_TO_USE;i++)
    {
        ret = mars_task_queue_create(mars_ctx, &cic_channel_index_queue[i], 16, 8, MARS_TASK_QUEUE_HOST_TO_MPU);
#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
        if(ret)
        {
            printf("MARS task queue create failed! (%d) at line %d\n", ret, __LINE__);
            exit(EXIT_FAILURE);
        }
#endif
    }

    // execute spe tasks
    for(i=0;i<NUM_SPE_TO_USE;i++)
    {
        task_args.type.u32[0] = i; // task id
        task_args.type.u32[1] = CIC_RUNCOUNT; // repetition count -  0 means task use task queue for repetition count // TODO
        task_args.type.u64[1] = cic_channel_info_queue[i]; // channel info queue
        task_args.type.u64[2] = cic_channel_index_queue[i]; // channel index queue

        ret = mars_task_schedule(&task_id[i], &task_args, 255);
#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
        if(ret)
        {
            printf("MARS task schedule failed! (%d) at line %d\n", ret, __LINE__);
            exit(EXIT_FAILURE);
        }
#endif
    }
}














static uint8_t data_buf[NUM_SPE_TO_USE][MARS_TASK_QUEUE_ENTRY_SIZE_MAX];

static pthread_mutex_t runcount_mutex = PTHREAD_MUTEX_INITIALIZER;
static int flag = CIC_RUNCOUNT;
static int runcount_value = CIC_RUNCOUNT;

static void *wrapper_thread_routine(void* pdata)
{
    int ret;
    int j;
    int data_index = -1;
    int result;

    int thread_index = (int)pdata;

#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
    if(thread_index < 0 || thread_index >= NUM_SPE_TO_USE)
    {
        printf("wrapper_thread_routine(): wrong thread index\n");
        exit(EXIT_FAILURE);
    }
#endif

    while(true)
    {
        if(flag != 0){
            result = pthread_mutex_lock(&runcount_mutex);
            if(runcount_value > 0)
            {
                runcount_value--;
            }
            else
            {
                result = pthread_mutex_unlock(&runcount_mutex);
                return NULL;
            }
            result = pthread_mutex_unlock(&runcount_mutex);
        }

        // check the index of the available data
        for(j=0;j<ARRAYLEN(cic_channel_info_ppe);j++)
        {
            if(cic_channel_info_ppe[j][thread_index].direction == MARS_TASK_QUEUE_HOST_TO_MPU)
            {
                int index_buf[4];
                data_index = AC_CHECK(cic_channel_info_ppe[j][thread_index].channel_id);
                index_buf[3] = data_index;
                ret = mars_task_queue_push(cic_channel_index_queue[thread_index], index_buf);
#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
                if(ret)
                {
                    printf("MARS task queue push failed! (%d) at line %d\n", ret, __LINE__);
                    exit(EXIT_FAILURE);
                }
#endif
                break;
            }
        }
#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
        if(data_index == -1)
        {
            printf(TASKNAME ":wrapper_ppe:AC_CHECK() error!\n");
            assert(false);
        }
#endif

        // send the data to spe task
        for(j=0;j<ARRAYLEN(cic_channel_info_ppe);j++)
        {
            cic_channel_info_entry *entry;
            entry = &cic_channel_info_ppe[j][thread_index];
            if(entry->direction == MARS_TASK_QUEUE_HOST_TO_MPU)
            {
                AC_RECEIVE(entry->channel_id, data_buf[thread_index], entry->size, data_index);
                ret = mars_task_queue_push(entry->queue_ea, data_buf[thread_index]);
#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
                if(ret)
                {
                    printf("MARS task queue push failed! (%d) at line %d\n", ret, __LINE__);
                    exit(EXIT_FAILURE);
                }
#endif
            }
        }

        // wait for spe task's completion
        // just do nothing....


        // receive data from spe tasks
        for(j=0;j<ARRAYLEN(cic_channel_info_ppe);j++)
        {
            cic_channel_info_entry *entry;
            entry = &cic_channel_info_ppe[j][thread_index];
            if(entry->direction == MARS_TASK_QUEUE_MPU_TO_HOST)
            {
                ret = mars_task_queue_pop(entry->queue_ea, data_buf[thread_index]);
#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
                if(ret)
                {
                    printf("MARS task queue push failed! (%d) at line %d\n", ret, __LINE__);
                    exit(EXIT_FAILURE);
                }
#endif
                AC_SEND(entry->channel_id, data_buf[thread_index], entry->size, data_index);
            }
        }
    } // while(true)

    return 0;
}

int FUNC_GO(void)
{
    // TODO
    int i;
    pthread_t th[NUM_SPE_TO_USE];

    for(i=0;i<NUM_SPE_TO_USE;i++)
    {
        pthread_create(&th[i], NULL, wrapper_thread_routine, (void *)i);
    }
    for(i=0;i<NUM_SPE_TO_USE;i++)
    {
        pthread_join(th[i], NULL);
    }

    return 0;
}












void FUNC_WRAPUP(void)
{
    int ret;
    int i, j;
    uint32_t count;

    for(i=0;i<NUM_SPE_TO_USE;i++)
    {
#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
        ret = mars_task_queue_count(cic_channel_info_queue[i], &count);
        if(ret || count != 0)
        {
            printf("MARS task queue count failed! (%d)\n", ret);
            exit(EXIT_FAILURE);
        }
#endif
        ret = mars_task_queue_destroy(cic_channel_info_queue[i]);
#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
        if(ret)
        {
            printf("MARS task queue destroy failed! (%d)\n", ret);
            exit(EXIT_FAILURE);
        }
#endif
#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
        ret = mars_task_queue_count(cic_channel_index_queue[i], &count);
        if(ret || count != 0)
        {
            printf("MARS task queue count failed! (%d)\n", ret);
            exit(EXIT_FAILURE);
        }
#endif
        ret = mars_task_queue_destroy(cic_channel_index_queue[i]);
#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
        if(ret)
        {
            printf("MARS task queue destroy failed! (%d)\n", ret);
            exit(EXIT_FAILURE);
        }
#endif

        for(j=0;j<ARRAYLEN(cic_channel_info_ppe);j++)
        {
#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
            ret = mars_task_queue_count(cic_channel_info_ppe[j][i].queue_ea, &count);
            if(ret || count != 0)
            {
                printf("MARS task queue count failed! (%d)\n", ret);
                exit(EXIT_FAILURE);
            }
#endif
            ret = mars_task_queue_destroy(cic_channel_info_ppe[j][i].queue_ea);
#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
            if(ret)
            {
                printf("MARS task queue destroy failed! (%d)\n", ret);
                exit(EXIT_FAILURE);
            }
#endif
        }
    }

    for(i=0;i<NUM_SPE_TO_USE;i++)
    {
        ret = mars_task_destroy(&task_id[i]);
#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
        if(ret)
        {
            // this is not the bug....
            //printf("MARS task destroy failed! (%d)\n", ret);
            //exit(EXIT_FAILURE);
        }
#endif
    }
}

#undef PROC_DEBUG
#undef ARRAYLEN
#undef ROUND16

