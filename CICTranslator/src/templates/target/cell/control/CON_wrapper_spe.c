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

#include "CON_wrapper.h"
#include "time_stamp.h"

#define PROC_DEBUG (1)

#define ARRAYLEN(ARR) (sizeof(ARR)/sizeof((ARR)[0]))
#define ROUNDUP16(arg) ((arg+15)&~0xf)



static uint64_t con_channel_info_queue;
static uint64_t con_channel_index_queue;
static uint64_t cic_channel_info_queue;

static uint32_t con_task_id;




/*
static void con_task_go_end(int return_code)
{
    //spu_write_out_intr_mbox(CIC_CMD_NOTIFY_GO_END);
    //spu_read_in_mbox();

    //mars_task_queue_push(CIC_CMD_NOTIFY_GO_END);
    //mars_task_queue_pop()
}

static void con_task_wrapup_end(void)
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

    FILE* fp;
    char* task_name;
    char temp[255];

    if(inited == false)
    {
        con_task_id = task_args->type.u32[0];
        con_channel_info_queue = task_args->type.u64[0];
        con_channel_index_queue = task_args->type.u64[1];
        cic_channel_info_queue = task_args->type.u64[2];
        con_channel_event_flag_spe = task_args->type.u64[3];

		typedef struct
		{
			uint64_t queue_ea;
			int32_t channel_id;
			uint32_t direction;
		} queue_entry;
		queue_entry data;


		for(i=0;i<ARRAYLEN(con_channel_info_spe);i++)
        {
            ret = mars_task_queue_pop(con_channel_info_queue, &data);
#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
            if(ret)
            {
                printf("MARS task queue pop failed! (%d) at line %d\n", ret, __LINE__);
                exit(EXIT_FAILURE);
            }
            if(data.channel_id != con_channel_info_spe[i].channel_id || data.direction != con_channel_info_spe[i].direction)
            {
				printf("*. %s\n", CONTASKNAME);
				printf("data.chennel_id : %d      con_channel_info_spe[i].channel_id : %d   \n", data.channel_id, con_channel_info_spe[i].channel_id);
				printf("data.direction  : %d      con_channel_info_spe[i].direction  : %d   \n", data.direction, con_channel_info_spe[i].direction);
                printf("MARS task queue pop failed! (%d) at line %d\n", ret, __LINE__);
                exit(EXIT_FAILURE);
            }
#endif
            con_channel_info_spe[i].queue_ea = data.queue_ea;
		}


		for(i=0;i<ARRAYLEN(cic_channel_info_spe);i++)
		{
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

    task_name = temp;
    strcpy(task_name, "trace_");
    strcat(task_name, CONTASKNAME);
    strcat(task_name, ".txt");

    fp = fopen(task_name, "w");

#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
	//printf("SPE(%d): \"%s\" (%d, %d) - init\n", mars_task_get_kernel_id(), mars_task_get_name(), cic_task_id, cic_repeat);
#endif

	(con_task_function.task_init)();


// Addded by jhw at 10.01.14 for library

	uint32_t ret_mask=0;
	while(true)
	{
		mars_task_signal_wait();
		//if(ret_mask == 1)
		{
#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
	//printf("SPE(%d): \"%s\" (%d, %d) - go \n", mars_task_get_kernel_id(), mars_task_get_name(), cic_task_id, cic_repeat);
#endif
            TIMESTAMP(fp, 0, 0, TASKID, -1, -1);
            fflush(fp);
			(con_task_function.task_go)();
        	mars_task_event_flag_set(con_channel_event_flag_spe, (uint32_t)1);
            TIMESTAMP(fp, 1, 0, TASKID, -1, -1);
            fflush(fp);
		}
		//else if(ret_mask == 2)
		{
#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
	//printf("SPE(%d): \"%s\" (%d, %d) - wrapup\n", mars_task_get_kernel_id(), mars_task_get_name(), cic_task_id, cic_repeat);
#endif
		//	(con_task_function.task_wrapup)();
		//	break;
		}
		ret_mask = 0;
	}

    fclose(fp);

#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
	//printf("SPE(%d): \"%s\" (%d, %d) - end\n", mars_task_get_kernel_id(), mars_task_get_name(), cic_task_id, cic_repeat);
#endif

    return EXIT_SUCCESS;
}

#undef ARRAYLEN
#undef ROUNDUP16
#undef PROC_DEBUG

