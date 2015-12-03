

#ifndef __SPU__

#error

#endif

#include <stdbool.h>

#include <assert.h>

#include "LIB_port.h"
#include "LIB_wrapper.h"

#define PROC_DEBUG (1)

#define ARRAYLEN(ARR) (sizeof(ARR)/sizeof((ARR)[0]))
#define ROUNDUP16(arg) ((arg+15)&~0xf)

/*
   static int lock_lib_channel(int channel_id)
   {
   return 0;
   }
   static int unlock_lib_channel(int channel_id)
   {
   return 0;
   }
*/

static int init_lib_port(int channel_id)
{
#if defined(PROC_DEBUG) && (PROC_DEBUG==1)

    //int i=0;
    //channel_id = lib_channel_info_spe[i].channel_id;

    return channel_id;

    assert(false);
#else
    return channel_id;
#endif
}

static int read_libport(int channel_id, int func_ret, unsigned char *buf, int len, int index)
{
    int i;
    uint32_t cond=0;

    if(func_ret == 0)		cond = MARS_TASK_QUEUE_HOST_TO_MPU;
    else if(func_ret == 1)	cond = MARS_TASK_QUEUE_MPU_TO_HOST;

    for(i=0;i<ARRAYLEN(lib_channel_info_spe);i++)
    {
        if(lib_channel_info_spe[i].channel_id == channel_id && lib_channel_info_spe[i].port_id == func_ret)
        {
            lib_channel_info_entry *entry;
            entry = &lib_channel_info_spe[i];
            /*
#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
            if(entry->queue_ea == 0x0)
            {
                printf("SPE:read_libport(): not initialized!\n");
                assert(false);
            }
            if(entry->direction != MARS_TASK_QUEUE_HOST_TO_MPU)
            {
                printf("SPE:read_libport(): wrong operation!\n");
                assert(false);
            }
            if(entry->size < len)
            {
                printf("SPE:read_libport(): len should be %d!\n", entry->size);
                assert(false);
            }
#endif
             */
            {
                if(((unsigned int)buf&0xf) != 0)
                {
                    uint8_t data_buf[ROUNDUP16(entry->size)];
                    mars_task_queue_pop(entry->queue_ea, data_buf);
                    //printf("port_spe : read_libport %d %d\n", func_ret, channel_id);
                    memcpy(buf, data_buf, len);
                }
                else
                {
                    mars_task_queue_pop(entry->queue_ea, buf);
                    //printf("port_spe : read_libport %d %d\n", func_ret, channel_id);
                }
            }
            break;
        }
    }

    return len;
}

static int write_libport(int channel_id, int func_ret, unsigned char *buf, int len, int index)
{
    int i;
    uint32_t cond=0;
    uint32_t temp=0;

    if(func_ret == 0)		cond = MARS_TASK_QUEUE_HOST_TO_MPU;
    else if(func_ret == 1)	cond = MARS_TASK_QUEUE_MPU_TO_HOST;

    for(i=0;i<ARRAYLEN(lib_channel_info_spe);i++)
    {
        if(lib_channel_info_spe[i].channel_id == channel_id && lib_channel_info_spe[i].port_id == func_ret)
        {
            lib_channel_info_entry *entry;
            entry = &lib_channel_info_spe[i];
            /*
#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
            if(entry->queue_ea == 0x0)
            {
                printf("SPE:write_libport(): not initialized!\n");
                assert(false);
            }
            if(entry->direction != MARS_TASK_QUEUE_MPU_TO_HOST)
            {
                printf("SPE:write_libport(): wrong operation!\n");
                assert(false);
            }
            if(entry->size < len)
            {
                printf("SPE:write_libport(): len should be %d!\n", entry->size);
                assert(false);
            }
#endif
             */
            {
                if(((unsigned int)buf&0xf) != 0)
                {
                    uint8_t data_buf[ROUNDUP16(entry->size)];
                    memcpy(data_buf, buf, len);
                    mars_task_queue_push(entry->queue_ea, data_buf);
                    temp = (channel_id * 100) + (func_ret * 10) + 1;
                    //printf("port_spe : write_libport %d %d     (temp : %d)\n", func_ret, channel_id, temp);
                    mars_task_event_flag_set(lib_channel_event_flag_spe, temp);
                }
                else
                {
                    mars_task_queue_push(entry->queue_ea, buf);
                    temp = (channel_id * 100) + (func_ret * 10) + 1;
                    //printf("port_spe : write_libport %d %d     (temp : %d)\n", func_ret, channel_id, temp);
                    mars_task_event_flag_set(lib_channel_event_flag_spe, temp);
                }
            }
            break;
        }
    }
    return len;
}

static int check_libport(int channel_id, int func_ret)
{
    int index[4];
    mars_task_queue_pop(lib_channel_index_queue, &index);

#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
    //printf("check_libport(%d) returns %d\n", channel_id, index[3]);
#endif

    return index[3];
}

#undef ROUNDUP
#undef ARRAYLEN
#undef PROC_DEBUG


