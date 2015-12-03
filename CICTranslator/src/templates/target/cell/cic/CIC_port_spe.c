
#ifndef __SPU__

    #error

#endif

#include <stdbool.h>
//#define NDEBUG
#include <assert.h>

#include "CIC_port.h"
#include "CIC_wrapper.h"

#define PROC_DEBUG (1)

#define ARRAYLEN(ARR) (sizeof(ARR)/sizeof((ARR)[0]))
#define ROUNDUP16(arg) ((arg+15)&~0xf)

static int init_port(int task_id, int port_id)
{
#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
    int i;

    for(i=0; i<ARRAYLEN(cic_channel_info_spe); i++)
    {
        if(cic_channel_info_spe[i].port_id == port_id)
        {
            return port_id;
        }
    }
    assert(false);
#else
    return port_id;
#endif
}

// Added by jhw at 09.12.22

static int init_task_port(int task_id, const char* port_name)
{
#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
    unsigned int i;

    for(i=0; i<ARRAYLEN(cic_channel_info_spe); i++)
    {
        if(strcmp(cic_channel_info_spe[i].port_name, port_name) == 0)
        {
            return cic_channel_info_spe[i].port_id;
        }
    }
#else
    return cic_channel_info_spe[i].port_id;
#endif
    /*
       printf("init_task_port(): Can't find channel(task id: %d, port name: %s)\n", task_id, port_name);
       exit(EXIT_FAILURE);
     */
}


static int read_acport(int channel_id, unsigned char *buf, int len, int index)
{
    int i;

    for(i=0;i<ARRAYLEN(cic_channel_info_spe);i++)
    {
        if(cic_channel_info_spe[i].port_id == channel_id)
        {
            cic_channel_info_entry *entry;
            entry = &cic_channel_info_spe[i];
#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
            if(entry->queue_ea == 0x0)
            {
                printf("SPE:read_acport(): not initialized!\n");
                assert(false);
            }
            if(entry->direction != MARS_TASK_QUEUE_HOST_TO_MPU)
            {
                printf("SPE:read_acport(): wrong operation!\n");
                assert(false);
            }
            if(entry->size < len)
            {
                printf("SPE:read_acport(): len should be %d!\n", entry->size);
                assert(false);
            }
#endif
            {
                if(((unsigned int)buf&0xf) != 0)
                {
                    uint8_t data_buf[ROUNDUP16(entry->size)];
                    mars_task_queue_pop(entry->queue_ea, data_buf);
                    memcpy(buf, data_buf, len);
                }
                else
                {
                    mars_task_queue_pop(entry->queue_ea, buf);
                }
            }
            break;
        }
    }

    return len;
}

static int write_acport(int channel_id, unsigned char *buf, int len, int index)
{
    int i;

    for(i=0;i<ARRAYLEN(cic_channel_info_spe);i++)
    {
        if(cic_channel_info_spe[i].port_id == channel_id)
        {
            cic_channel_info_entry *entry;
            entry = &cic_channel_info_spe[i];
#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
            if(entry->queue_ea == 0x0)
            {
                printf("SPE:write_acport(): not initialized!\n");
                assert(false);
            }
            if(entry->direction != MARS_TASK_QUEUE_MPU_TO_HOST)
            {
                printf("SPE:write_acport(): wrong operation!\n");
                assert(false);
            }
            if(entry->size < len)
            {
                printf("SPE:write_acport(): len should be %d!\n", entry->size);
                assert(false);
            }
#endif
            {
                if(((unsigned int)buf&0xf) != 0)
                {
                    uint8_t data_buf[ROUNDUP16(entry->size)];
                    memcpy(data_buf, buf, len);
                    mars_task_queue_pop(entry->queue_ea, data_buf);
                }
                else
                {
                    mars_task_queue_push(entry->queue_ea, buf);
                }
            }
            break;
        }
    }

    return len;
}

static int check_acport(int channel_id)
{
    int index[4];
    mars_task_queue_pop(cic_channel_index_queue, &index);

#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
    //printf("check_acport(%d) returns %d\n", channel_id, index[3]);
#endif

    return index[3];
}

#undef ROUNDUP
#undef ARRAYLEN
#undef PROC_DEBUG

