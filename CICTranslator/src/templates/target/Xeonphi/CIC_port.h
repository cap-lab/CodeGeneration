#ifndef __CIC_PORT_H__
#define __CIC_PORT_H__

#include <stdio.h>
#include <string.h>
#include <stdlib.h>


#define SET_ARGUMENT(a, b) set_argument(a, b, argument)  
#define FREE_ARGUMENT(a, b) set_argument(a, b)
#define GET_PARALLEL_INFO(a, b) get_parallel_info(a, b, parallel_id,  max_parallel)

#define AC_RECEIVE(a, b, c, d) read_acport( task_data_addr[a], (unsigned char *)b, c, d)
#define AC_SEND(a, b, c, d) write_acport( task_check_addr[a], (unsigned char *)b, c, d)
#define MQ_RECEIVE(a, b, c) read_port( task_data_addr[a], (unsigned char *)b, c)
#define MQ_SEND(a, b, c) write_port( task_check_addr[a], (unsigned char *)b, c)

extern int init_task_port(int id, char* name);
extern void read_acport( volatile long* task_data_addr, unsigned char * user_data, int size, int index);
extern void write_acport(volatile  long* task_data_addr, unsigned char * user_data, int size, int index);
extern void read_port( volatile long* task_data_addr, unsigned char * user_data, int size);
extern void write_port(volatile  long* task_data_addr, unsigned char * user_data, int size);

#endif /* __CIC_PORT_H__ */

