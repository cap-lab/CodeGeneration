
#ifndef __CON_WRAPPER_H__
#define __CON_WRAPPER_H__

#include <stdint.h>
#include "CIC_CMD.h"

#define STR(a) #a

extern struct mars_context *mars_ctx;

typedef struct
{
    void    (*task_init)(void);
    void    (*task_go)(void);
    void    (*task_wrapup)(void);
} con_task_function_t;


typedef struct
{
    uint64_t queue_ea;
    int32_t channel_id;
    uint32_t direction;
    int32_t port_id;
    uint32_t size;
    uint32_t depth;
} con_channel_info_entry;

typedef struct
{
    uint64_t queue_ea;
    int32_t channel_id;
    uint32_t direction;
	char* port_name;
    int32_t port_id;
    uint32_t size;
    uint32_t depth;
} cic_channel_info_entry;

#endif /* __CON_WRAPPER_H__ */

