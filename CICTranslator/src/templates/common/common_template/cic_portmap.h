#ifndef __CIC_PORTMAPS_H__
#define __CIC_PORTMAPS_H__

CIC_TYPEDEF CIC_T_STRUCT {
    CIC_T_INT task_id;
    CIC_T_INT port_id;
    CIC_T_CHAR *port_name;
    CIC_T_INT channel_id;
    CIC_T_CHAR op;
    CIC_T_INT port_rate;
} CIC_UT_PORTMAP;

#endif /* __CIC_PORTMAPS_H__ */