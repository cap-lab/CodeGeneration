#include "cuda_runtime.h"

#include "target_system_model.h"
#include "target_task_model.h"
#include "cic_channels.h"
#include "cic_portmap.h"
#include "cic_tasks.h"
#include "cic_gpuinfo.h"

#define ARRAYLEN(ARR) (sizeof(ARR)/sizeof(ARR[0]))

extern portmap addressmap[];
extern CHANNEL channels[];
extern TASK tasks[];
extern GPUTASKINFO gpuTaskInfo[];
extern GPUCHANNELINFO gpuChannelInfo[];
extern CLUSTER_BUFFER cluster_buffers[];

extern int _num_tasks;
extern int _num_channels;
extern int _num_portmaps;
extern int _num_cluster_buffers;

static int init_task_port(int task_id, const char* port_name)
{
    int i;
    int channel_id;

    for(i=0; i<_num_portmaps; i++)
    {
        if(addressmap[i].task_id == task_id && strcmp(addressmap[i].port_name,port_name)==0)
        {
            unsigned int j;
            channel_id = addressmap[i].channel_id;
            for(j=0; j<_num_channels; j++)
            {
                if(channels[j].channel_id == channel_id)
                {   
                    return j;
                }
            }  
            printf("Warning : Task[%s] Port[%s] is not connected.\n",tasks[task_id].name, port_name);
            return -1;
        }
    }

    printf("Warning : init_task_port(): No port %s in task %s is available or connected\n", port_name, tasks[task_id].name);
    return -1;
}

static CLUSTER_BUFFER *get_cluster(char direction)
{
    int i=0;    
    for(i=0; i<_num_cluster_buffers; i++)
    {
        if(cluster_buffers[i].task_id == TASK_INDEX && cluster_buffers[i].direction == direction)
            return &(cluster_buffers[i]);
    }
    printf("get_cluster(): Can't find cluster(cluster_index: %d)\n", TASK_INDEX);
    return NULL;
}

static int read_cuda_port(int channel_index, unsigned char *device_buf, int len) // blocking
{
    CLUSTER_BUFFER *cluster;
    unsigned char *buf;
    int cur_size=0;

    cluster = get_cluster('r');
    CHANNEL *channel;
    channel = &channels[channel_index];

    if(__run_count >= __runs)    return -1;


    if(len < 0)
    {
        printf("%d %d\n", channel_index, len);
        printf("read_port(): len should be larger than 0!\n");
        exit(EXIT_FAILURE);
    }

    if(len > channel->max_size)
    {
        printf("read_port(): max_size is too small!\n");
        exit(EXIT_FAILURE);
    }

read_start:

    if(channel->start <= channel->end)  cur_size = (unsigned int)(channel->end - channel->start);
    else                                cur_size = (unsigned int)(channel->end + channel->max_size + sizeof(int) - channel->start);

    if(len > cur_size) // blocking case
    {
        MUTEX_LOCK(&(channel->mutex));
        channel->request_read = true;
        COND_BROADCAST(&(channel->cond));
        COND_WAIT(&(channel->cond), &(channel->mutex));
        MUTEX_UNLOCK(&(channel->mutex));
        goto read_start; // try reading again
    }
    
    MUTEX_LOCK(&cluster->p_mutex);
    buf = cluster->start;

    if(channel->start + len <= channel->buf + channel->max_size + sizeof(int))
    {
        memcpy(buf, channel->start, len);
        channel->start += len;
    }
    else
    {
        int part = channel->max_size + sizeof(int) - (channel->start - channel->buf);
        if(part != 0)
        {
            memcpy(buf, channel->start, part);
        }
        memcpy(buf + part, channel->buf, len - part);
        channel->start = channel->buf + (len - part);
    }

    cluster->start = cluster->start + len;
    cluster->cur_size = cluster->cur_size + len;

    if(cluster->cur_size == cluster->sample_size)
    {
        CUDA_ERROR_CHECK(cudaMemcpyAsync(__in_dev_buf[__current_stream], cluster->buf, cluster->sample_size, cudaMemcpyHostToDevice, __stream[__current_stream]));
        cluster->cur_size = 0;
        cluster->start = cluster->buf;
    }
    MUTEX_UNLOCK(&cluster->p_mutex);

    channel->request_read = false;
    COND_BROADCAST(&(channel->cond));

    return len;
}

static int write_cuda_port(int channel_index, unsigned char *device_buf, int len) // blocking
{
    CHANNEL *channel;
    CLUSTER_BUFFER *cluster = get_cluster('w');
    int cur_size=0;
    unsigned char *buf;

    if(__run_count <= 0)    return -1;

    if(len < 0)
    {
        printf("write_port(): len should be larger than 0!\n");
        exit(EXIT_FAILURE);
    }

    if(cluster->cur_size == 0)
    {
        CUDA_ERROR_CHECK(cudaMemcpyAsync(cluster->buf, __out_dev_buf[__previous_1_stream], cluster->sample_size, cudaMemcpyDeviceToHost, __stream[__previous_1_stream]));
        MUTEX_LOCK(&cluster->p_mutex);
        cluster->cur_size = cluster->sample_size;
        cluster->start = cluster->buf;
        MUTEX_UNLOCK(&cluster->p_mutex);
    }

    CUDA_ERROR_CHECK(cudaStreamSynchronize(__stream[__previous_1_stream]));
    
write_start:
    buf = cluster->start;

    while(channel_index >= 0) {
        channel = &channels[channel_index];

        if(channel->start <= channel->end)  cur_size = (unsigned int)(channel->end - channel->start);
        else                                cur_size = (unsigned int)(channel->end + channel->max_size + sizeof(int) - channel->start);

        if(len + cur_size > channel->max_size+sizeof(int)) // blocking case
        {
            MUTEX_LOCK(&(channel->mutex));
            channel->request_write = true;
            COND_BROADCAST(&(channel->cond));
            COND_WAIT(&(channel->cond), &(channel->mutex));
            MUTEX_UNLOCK(&(channel->mutex));
            goto write_start; // try reading again
        }

        if(channel->buf + channel->max_size+sizeof(int) >= channel->end + len)
        {
            memcpy(channel->end, buf, len);
            channel->end += len;
        }
        else
        {
            int part = channel->max_size+sizeof(int) - (channel->end - channel->buf);
            if(part != 0)
            {
                memcpy(channel->end, buf, part);
            }
            memcpy(channel->buf, buf + part, len - part);
            channel->end = channel->buf + (len - part);
        }

        channel->request_write = false;
        COND_BROADCAST(&(channel->cond));

        channel_index = channel->next_channel_index;
    }
    MUTEX_LOCK(&cluster->p_mutex);
    cluster->start = cluster->start + len;
    cluster->cur_size = cluster->cur_size - len;
    MUTEX_UNLOCK(&cluster->p_mutex);

    return len;
}

static int cuda_available(int channel_index) // non-blocking
{
    int ret;

    CHANNEL *channel;
    channel = &channels[channel_index];

    MUTEX_LOCK(&(channel->mutex));

    ret = channel->cur_size;

    MUTEX_UNLOCK(&(channel->mutex));

    return ret;
}

TASK_PREINIT
{
    int i=0, j=0;

    for(i=0; i<__num_of_streams; i++)
        (cudaStreamCreate(&__stream[i]));

    for(i=0; i<_num_cluster_buffers; i++)
    {
        if(cluster_buffers[i].task_id == TASK_INDEX)
        {
            for(j=0; j<__num_of_streams; j++)
            {
                if(cluster_buffers[i].direction == 'r'){
                    CUDA_ERROR_CHECK(cudaMalloc((void**) &__in_dev_buf[j], cluster_buffers[i].sample_size));
                    CUDA_ERROR_CHECK(cudaMemset((void*) __in_dev_buf[j], 0x0, cluster_buffers[i].sample_size)); 
                } 
                else if(cluster_buffers[i].direction == 'w'){
                    CUDA_ERROR_CHECK(cudaMalloc((void**) &__out_dev_buf[j], cluster_buffers[i].sample_size));
                    CUDA_ERROR_CHECK(cudaMemset((void*) __out_dev_buf[j], 0x0, cluster_buffers[i].sample_size)); 
                }
            }
        }
    }
}

TASK_PREGO
{
    __current_stream = __run_count % __num_of_streams;
    __previous_1_stream = (__run_count + __num_of_streams - 1) % __num_of_streams;
}

TASK_POSTGO
{
    int i=0;

    for(i=0; i<_num_cluster_buffers; i++)
    {
        if(cluster_buffers[i].task_id == TASK_INDEX && cluster_buffers[i].direction == 'w'){
            CUDA_ERROR_CHECK(cudaMemset((void*) __out_dev_buf[__previous_1_stream], 0x0, cluster_buffers[i].sample_size));
        }
    }

    __run_count++;
}

TASK_POSTWRAPUP
{
    int i=0;

    for(i=0; i<__num_of_streams; i++)
    {
        CUDA_ERROR_CHECK(cudaStreamDestroy(__stream[i]));
        CUDA_ERROR_CHECK(cudaFree(__in_dev_buf[i]));
        CUDA_ERROR_CHECK(cudaFree(__out_dev_buf[i]));
    }
}
