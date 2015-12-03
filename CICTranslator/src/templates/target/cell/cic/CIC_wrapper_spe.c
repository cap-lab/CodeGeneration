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
#include "CIC_wrapper.h"

#include "time_stamp.h"

#define PROC_DEBUG (1)

#define ARRAYLEN(ARR) (sizeof(ARR)/sizeof((ARR)[0]))
#define ROUNDUP16(arg) ((arg+15)&~0xf)



static uint64_t cic_channel_info_queue;
static uint64_t cic_channel_index_queue;

static uint32_t cic_task_id;
static uint32_t cic_repeat;










static void cic_task_go_end(int return_code)
{
    //spu_write_out_intr_mbox(CIC_CMD_NOTIFY_GO_END);
    //spu_read_in_mbox();

    //mars_task_queue_push(CIC_CMD_NOTIFY_GO_END);
    //mars_task_queue_pop()
}

static void cic_task_wrapup_end(void)
{
    //spu_write_out_intr_mbox(CIC_CMD_NOTIFY_WRAPUP_END);
    //spu_read_in_mbox();

    //mars_task_queue_push(CIC_CMD_NOTIFY_WRAPUP_END);
    //mars_task_queue_pop()
}













static bool inited = false;

int mars_task_main(const struct mars_task_args *task_args)
{
    uint32_t loop_count;
    int i;
    int ret;
    /*
       FILE* fp;
       char* task_name;
       char temp[255];
     */
    if(inited == false)
    {
        cic_task_id = task_args->type.u32[0];
        cic_repeat = task_args->type.u32[1];
        cic_channel_info_queue = task_args->type.u64[1];
        cic_channel_index_queue = task_args->type.u64[2];

#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
        //    printf("cic_task_id: %d\n", cic_task_id);
        //    printf("cic_repeat: %d\n", cic_repeat);
        //    printf("cic_channel_info_queue: 0x%llx\n", cic_channel_info_queue);
#endif

        for(i=0;i<ARRAYLEN(cic_channel_info_spe);i++)
        {
            typedef struct
            {
                uint64_t queue_ea;
                int32_t channel_id;
                uint32_t direction;
            } queue_entry;
            queue_entry data;
            ret = mars_task_queue_pop(cic_channel_info_queue, &data);
#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
            if(ret)
            {
                printf("MARS task queue pop failed! (%d) at line %d\n", ret, __LINE__);
                exit(EXIT_FAILURE);
            }
            if(data.channel_id != cic_channel_info_spe[i].channel_id || data.direction != cic_channel_info_spe[i].direction)
            {
                printf("MARS task queue pop failed! (%d) at line %d\n", ret, __LINE__);
                exit(EXIT_FAILURE);
            }
#endif
            cic_channel_info_spe[i].queue_ea = data.queue_ea;
        }

        inited = true;
    }
    /*
       task_name = temp;
       strcpy(task_name, "trace_");
       strcat(task_name, TASKNAME);
       strcat(task_name, ".txt");

       fp = fopen(task_name, "w");
     */
#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
    //printf("SPE(%d): \"%s\" (%d, %d) - init\n", mars_task_get_kernel_id(), mars_task_get_name(), cic_task_id, cic_repeat);
#endif

    (cic_task_function.task_init)((int)cic_task_id);

#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
    //printf("SPE(%d): \"%s\" (%d, %d) - go \n", mars_task_get_kernel_id(), mars_task_get_name(), cic_task_id, cic_repeat);
#endif

    if(cic_repeat == 0){
        while(1)
        {
            int return_code;
            return_code = (cic_task_function.task_go)();
            cic_task_go_end(return_code);
        }
    }
    else{
        for(loop_count=0;loop_count<cic_repeat;loop_count++)
        {
            int return_code;
            /*
               TIMESTAMP(fp, 0, 0, TASKID, -1, -1);
               fflush(fp);
             */
            return_code = (cic_task_function.task_go)();
            /*
               TIMESTAMP(fp, 1, 0, TASKID, -1, -1);
               fflush(fp);
             */
            cic_task_go_end(return_code);
        }
    }

#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
    //printf("SPE(%d): \"%s\" (%d, %d) - wrapup\n", mars_task_get_kernel_id(), mars_task_get_name(), cic_task_id, cic_repeat);
#endif
    (cic_task_function.task_wrapup)();
    cic_task_wrapup_end();

    //    fclose(fp);
#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
    //printf("SPE(%d): \"%s\" (%d, %d) - end\n", mars_task_get_kernel_id(), mars_task_get_name(), cic_task_id, cic_repeat);
#endif

    return EXIT_SUCCESS;
}

#undef ARRAYLEN
#undef ROUNDUP16
#undef PROC_DEBUG

