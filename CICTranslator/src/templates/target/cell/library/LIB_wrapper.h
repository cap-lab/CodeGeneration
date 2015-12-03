
#ifndef __LIB_WRAPPER_H__
#define __LIB_WRAPPER_H__

#include <stdint.h>
#include "CIC_CMD.h"

#define STR(a) #a

extern struct mars_context *mars_ctx;

typedef struct
{
    void    (*task_init)(void);
    void     (*task_go)(void);
    void    (*task_wrapup)(void);
} lib_stub_function_t;

// Fixed by jhw at 09.12.29 for add port_name

typedef struct
{
    uint64_t queue_ea;
    int32_t channel_id;
    uint32_t direction;
    int32_t port_id;
    uint32_t size;
    uint32_t depth;
} lib_channel_info_entry;

/*
typedef struct
{
    CIC_CMD cmd;
    uint32_t channel_id;
    uint32_t len;
    uint32_t index;
} cic_spe_task_request_queue_entry;
*/

#endif /* __LIB_WRAPPER_H__ */

