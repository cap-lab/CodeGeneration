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

//Added by jhw at 10.01.11 for library
#include "LIB_wrapper.h"
#include "LIB_port.h"

//Added by jhw at 10.03.11 for library
#define FUNC 0
#define RET  1
// To here

#define PROC_DEBUG (1)

#define ARRAYLEN(ARR) (sizeof(ARR)/sizeof((ARR)[0]))
#define ROUNDUP16(arg) ((arg+15)&~0xf)


extern struct spe_program_handle L_SPE_PROG_NAME;

//Added by jhw at 10.01.11 for library
static struct mars_task_id stub_id[NUM_SPE_TO_USE];

//Added by jhw at 10.01.11 for library
static struct mars_task_args stub_args;

static uint64_t __attribute__((aligned(32))) lib_channel_info_queue[NUM_SPE_TO_USE];

static uint64_t __attribute__((aligned(32))) lib_channel_index_queue[NUM_SPE_TO_USE];

//static uint64_t lib_channel_event_flag[NUM_SPE_TO_USE];



void LIB_INIT(void)
{
    int ret;
    int i, j;


    for(i=0;i<NUM_SPE_TO_USE;i++)
    {
        {
            ret = mars_task_create_with_affinity(mars_ctx, &stub_id[i], 0, LIBNAME, L_SPE_PROG_NAME.elf_image, MARS_TASK_CONTEXT_SAVE_SIZE_MAX);
#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
            if(ret)
            {
                printf("MARS task create failed! (%d)\n", ret);
                exit(EXIT_FAILURE);
            }
#endif
        }
    }

    for(j=0;j<NUM_SPE_TO_USE;j++)
    {
        // Added by jhw at 10.01.06 for library
        ret = mars_task_queue_create(mars_ctx, &lib_channel_info_queue[j], sizeof(uint64_t)+sizeof(int32_t)+sizeof(uint32_t), ARRAYLEN(lib_channel_info_ppe), MARS_TASK_QUEUE_HOST_TO_MPU);
#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
        if(ret)
        {
            printf("MARS task queue create failed! (%d) at line %d\n", ret, __LINE__);
            exit(EXIT_FAILURE);
        }
#endif
        //to here

        // Added by jhw at 10.01.06 for library
        for(i=0;i<ARRAYLEN(lib_channel_info_ppe);i++)
        {
            ret = mars_task_queue_create(mars_ctx, &(lib_channel_info_ppe[i][j].queue_ea), ROUNDUP16(lib_channel_info_ppe[i][j].size), lib_channel_info_ppe[i][j].depth, lib_channel_info_ppe[i][j].direction);
#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
            if(ret)
            {
                printf("MARS task queue create failed! (%d) at line %d\n", ret, __LINE__);
                exit(EXIT_FAILURE);
            }
#endif
            ret = mars_task_queue_push(lib_channel_info_queue[j], &lib_channel_info_ppe[i][j]);
#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
            if(ret)
            {
                printf("MARS task queue push failed! (%d) at line %d\n", ret, __LINE__);
                exit(EXIT_FAILURE);
            }
#endif
        }
    }
    // to here


    for(i=0;i<NUM_SPE_TO_USE;i++)
    {
        {
            ret = mars_task_queue_create(mars_ctx, &lib_channel_index_queue[i], 16, 8, MARS_TASK_QUEUE_HOST_TO_MPU);
#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
            if(ret)
            {
                printf("MARS task queue create failed! (%d) at line %d\n", ret, __LINE__);
                exit(EXIT_FAILURE);
            }
#endif
        }
    }

    // Added by jhw for library at 10.03.11
    for(i=0;i<NUM_SPE_TO_USE;i++)
    {
        ret = mars_task_event_flag_create(mars_ctx, &lib_channel_event_flag_ppe[i], MARS_TASK_EVENT_FLAG_MPU_TO_HOST, MARS_TASK_EVENT_FLAG_CLEAR_AUTO);
#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
        if(ret)
        {
            printf("MARS task event flag create failed! (%d) at line %d\n", ret, __LINE__);
            exit(EXIT_FAILURE);
        }
#endif
    }
    //To here

    // execute spe tasks
    for(i=0;i<NUM_SPE_TO_USE;i++)
    {
        {
            stub_args.type.u32[0] = i;
            //stub_args.type.u32[1] = 0;
            stub_args.type.u64[1] = lib_channel_info_queue[i];
            stub_args.type.u64[2] = lib_channel_index_queue[i];
            stub_args.type.u64[3] = lib_channel_event_flag_ppe[i];
            ret = mars_task_schedule(&stub_id[i], &stub_args, 255);
#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
            if(ret)
            {
                printf("MARS task schedule failed! (%d) at line %d\n", ret, __LINE__);
                exit(EXIT_FAILURE);
            }
#endif
        }
    }
}



static uint8_t __attribute__((aligned(32)))  lib_data_buf[NUM_SPE_TO_USE][MARS_TASK_QUEUE_ENTRY_SIZE_MAX];

//static pthread_mutex_t lib_runcount_mutex = PTHREAD_MUTEX_INITIALIZER;

static void *lib_wrapper_thread_routine(void* pdata)
{
    int ret;
    int j;
    int data_index=-1;

    int wait_channel = 0; 	// To set next channel that wrapper will wait on...
    int send_channel = 0; 	// To set next channel that wrapper will send to...
    int func_ret = 0;          	
    uint32_t ret_mask=0;

    int thread_index = (int)pdata;

#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
    if(thread_index < 0 || thread_index >= NUM_SPE_TO_USE)
    {
        printf("wrapper_thread_routine(): wrong thread index\n");
        exit(EXIT_FAILURE);
    }
#endif

    // initialize wait_channel variable
    func_ret = FUNC;
    wait_channel = lib_channel_info_ppe[0][thread_index].channel_id;

    while(true)
    {
        // check the index of the available data
        int index_buf[4];
        data_index = LIB_AC_CHECK(wait_channel, func_ret);
        //printf("wrapper_ppe : LIB_CHECK %d %d\n", func_ret, wait_channel);

        index_buf[3] = data_index;
        ret = mars_task_queue_push(lib_channel_index_queue[thread_index], index_buf);
#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
        if(ret)
        {
            printf("MARS task queue push failed! (%d) at line %d\n", ret, __LINE__);
            exit(EXIT_FAILURE);
        }
#endif


#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
        if(data_index == -1)
        {
            printf(LIBNAME ":wrapper_ppe:AC_CHECK() error!\n");
            assert(false);
        }
#endif

        // Receive the data from channel and Send the data to spe task
        for(j=0;j<ARRAYLEN(lib_channel_info_ppe);j++)
        {
            lib_channel_info_entry *entry;
            entry = &lib_channel_info_ppe[j][thread_index];
            if((entry->channel_id == wait_channel) && (entry->port_id == func_ret))
            {
                LIB_RECEIVE(wait_channel, func_ret, lib_data_buf[thread_index], entry->size, data_index);
                if(func_ret == RET)     unlock_lib_channel(wait_channel);
                //printf("wrapper_ppe : LIB_RECEIVE %d %d\n", func_ret, wait_channel);
                ret = mars_task_queue_push(entry->queue_ea, lib_data_buf[thread_index]);
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

        // wait for spe task's completion use mars event flag (include setting wait_channel & func_ret)
        ret = mars_task_event_flag_wait(lib_channel_event_flag_ppe[thread_index], 0xFFFFFFFF, MARS_TASK_EVENT_FLAG_MASK_OR, &ret_mask);

#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
        if(ret)
        {
            printf("MARS task event flag wait failed! (%d) at line %d\n", ret, __LINE__);
            exit(EXIT_FAILURE);
        }
#endif

        // Calculate channel & func_ret
        uint32_t temp=0;
        temp = ret_mask;
        send_channel = (int)temp / 100;
        temp = temp % 100;
        func_ret = (int)temp / 10;
        ret_mask = 0x0;
        //printf("ret_mask : %u, send_channel : %d, func_ret : %d\n", ret_mask, send_channel, func_ret);

        // Receive data from spe tasks and Send the data to channel in PPE
        for(j=0;j<ARRAYLEN(lib_channel_info_ppe);j++)
        {
            lib_channel_info_entry *entry;
            entry = &lib_channel_info_ppe[j][thread_index];
            if((entry->channel_id == send_channel) && (entry->port_id == func_ret))
            {
                ret = mars_task_queue_pop(entry->queue_ea, lib_data_buf[thread_index]);
#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
                if(ret)
                {
                    printf("MARS task queue push failed! (%d) at line %d\n", ret, __LINE__);
                    exit(EXIT_FAILURE);
                }
#endif
                if(func_ret == FUNC)     lock_lib_channel(send_channel);
                LIB_SEND(send_channel, func_ret, lib_data_buf[thread_index], entry->size, data_index);
                //printf("wrapper_ppe : LIB_SEND %d %d\n", func_ret, send_channel);
                break;
            }
        }


        // Set wait_channel variable
        wait_channel = send_channel;
        if(func_ret == FUNC)		func_ret = RET;
        else if(func_ret == RET)	func_ret = FUNC;

    } 
    return 0;
}


int LIB_GO(void)
{
    // TODO
    int i;
    // Added by jhw at 10.01.11 for library
    pthread_t l_th[NUM_SPE_TO_USE];

    for(i=0;i<NUM_SPE_TO_USE;i++)
    {
        pthread_create(&l_th[i], NULL, lib_wrapper_thread_routine, (void *)i);
    }

    for(i=0;i<NUM_SPE_TO_USE;i++)
    {
        pthread_join(l_th[i], NULL);
    }

    return 0;
}




void LIB_WRAPUP(void)
{
    int ret;
    int i, j;
    uint32_t count;

    for(i=0;i<NUM_SPE_TO_USE;i++)
    {
        // Added by jhw at 10.01.06 for library
        {
#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
            ret = mars_task_queue_count(lib_channel_info_queue[i], &count);
            if(ret || count != 0)
            {
                printf("MARS task queue count failed! (%d)\n", ret);
                exit(EXIT_FAILURE);
            }
#endif
            ret = mars_task_queue_destroy(lib_channel_info_queue[i]);
#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
            if(ret)
            {
                printf("MARS task queue destroy failed! (%d)\n", ret);
                exit(EXIT_FAILURE);
            }
#endif
        }

        // Added by jhw at 10.01.06 for library
        {
            ret = mars_task_queue_count(lib_channel_index_queue[i], &count);
#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
            if(ret || count != 0)
            {
                printf("MARS task queue count failed! (%d)\n", ret);
                exit(EXIT_FAILURE);
            }
#endif
            ret = mars_task_queue_destroy(lib_channel_index_queue[i]);
#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
            if(ret)
            {
                printf("MARS task queue destroy failed! (%d)\n", ret);
                exit(EXIT_FAILURE);
            }
#endif
        }

        // Added by jhw at 10.01.06 for library
        for(j=0;j<ARRAYLEN(lib_channel_info_ppe);j++)
        {
            ret = mars_task_queue_count(lib_channel_info_ppe[j][i].queue_ea, &count);
#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
            if(ret || count != 0)
            {
                printf("MARS task queue count failed! (%d)\n", ret);
                exit(EXIT_FAILURE);
            }
#endif
            ret = mars_task_queue_destroy(lib_channel_info_ppe[j][i].queue_ea);
#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
            if(ret)
            {
                printf("MARS task queue destroy failed! (%d)\n", ret);
                exit(EXIT_FAILURE);
            }
#endif
        }  
    }
    //Added by jhw for library wrapper at 10.03.11
    for(i=0;i<NUM_SPE_TO_USE;i++)
    {
        {
            ret = mars_task_event_flag_destroy(lib_channel_event_flag_ppe[i]);
        }
    }

    for(i=0;i<NUM_SPE_TO_USE;i++)
    {
        {
            ret = mars_task_destroy(&stub_id[i]);
        }
    }	
}

#undef PROC_DEBUG
#undef ARRAYLEN
#undef ROUND16

