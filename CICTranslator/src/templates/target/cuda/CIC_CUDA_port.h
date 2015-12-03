#ifndef __CIC_CUDA_PORT_H__
#define __CIC_CUDA_PORT_H__

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static int init_task_port(int task_id, const char* port_name);
static int read_cuda_port(int channel_id, unsigned char *buf, int len);
static int write_cuda_port(int channel_id, unsigned char *buf, int len);
static int cuda_available(int channel_id);

#define MQ_RECEIVE(a, b, c) read_cuda_port(a, (unsigned char *)b, c)
#define MQ_SEND(a, b, c) write_cuda_port(a, (unsigned char *)b, c)
#define MQ_AVAILABLE(a) cuda_available(a)

#endif /* __CIC_CUDA_PORT_H__ */

