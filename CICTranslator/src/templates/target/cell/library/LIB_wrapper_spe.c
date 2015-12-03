#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

//#define NDEBUG
#include <assert.h>

#include <mars/task.h>

#include <spu_mfcio.h>
#include "CIC_CMD.h"

// Addded by jhw at 10.01.14 for library
#include "LIB_wrapper.h"

#define PROC_DEBUG (1)

#define ARRAYLEN(ARR) (sizeof(ARR)/sizeof((ARR)[0]))
#define ROUNDUP16(arg) ((arg+15)&~0xf)



static uint64_t lib_channel_info_queue;
static uint64_t lib_channel_index_queue;

static uint32_t lib_stub_id;




/*
static void lib_task_go_end(int return_code)
{
    //spu_write_out_intr_mbox(CIC_CMD_NOTIFY_GO_END);
    //spu_read_in_mbox();

    //mars_task_queue_push(CIC_CMD_NOTIFY_GO_END);
    //mars_task_queue_pop()
}

static void lib_task_wrapup_end(void)
{
    //spu_write_out_intr_mbox(CIC_CMD_NOTIFY_WRAPUP_END);
    //spu_read_in_mbox();

    //mars_task_queue_push(CIC_CMD_NOTIFY_WRAPUP_END);
    //mars_task_queue_pop()
}
*/




static bool inited = false;

int mars_task_main(const struct mars_task_args *task_args)
{
    int i;
    int ret;

    if(inited == false)
    {
        lib_stub_id = task_args->type.u32[0];
        lib_channel_info_queue = task_args->type.u64[1];
        lib_channel_index_queue = task_args->type.u64[2];
        lib_channel_event_flag_spe = task_args->type.u64[3];
#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
//    printf("cic_task_id: %d\n", cic_task_id);
//    printf("cic_repeat: %d\n", cic_repeat);
//    printf("cic_channel_info_queue: 0x%llx\n", cic_channel_info_queue);
#endif

	for(i=0;i<ARRAYLEN(lib_channel_info_spe);i++)
        {
            typedef struct
            {
                uint64_t queue_ea;
                int32_t channel_id;
                uint32_t direction;
            } queue_entry;
            queue_entry data;
            ret = mars_task_queue_pop(lib_channel_info_queue, &data);
#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
            if(ret)
            {
                printf("MARS task queue pop failed! (%d) at line %d\n", ret, __LINE__);
                exit(EXIT_FAILURE);
            }
            if(data.channel_id != lib_channel_info_spe[i].channel_id || data.direction != lib_channel_info_spe[i].direction)
            {
                printf("MARS task queue pop failed! (%d) at line %d\n", ret, __LINE__);
                exit(EXIT_FAILURE);
            }
#endif
            lib_channel_info_spe[i].queue_ea = data.queue_ea;
        }

        inited = true;
    }

#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
    //printf("SPE(%d): \"%s\" (%d, %d) - init\n", mars_task_get_kernel_id(), mars_task_get_name(), cic_task_id, cic_repeat);
#endif

    (lib_stub_function.task_init)();

#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
    //printf("SPE(%d): \"%s\" (%d, %d) - go \n", mars_task_get_kernel_id(), mars_task_get_name(), cic_task_id, cic_repeat);
#endif

// Addded by jhw at 10.01.14 for library
    (lib_stub_function.task_go)();
#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
    //printf("SPE(%d): \"%s\" (%d, %d) - wrapup\n", mars_task_get_kernel_id(), mars_task_get_name(), cic_task_id, cic_repeat);
#endif

    (lib_stub_function.task_wrapup)();
#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
    //printf("SPE(%d): \"%s\" (%d, %d) - end\n", mars_task_get_kernel_id(), mars_task_get_name(), cic_task_id, cic_repeat);
#endif

    return EXIT_SUCCESS;
}

#undef ARRAYLEN
#undef ROUNDUP16
#undef PROC_DEBUG

