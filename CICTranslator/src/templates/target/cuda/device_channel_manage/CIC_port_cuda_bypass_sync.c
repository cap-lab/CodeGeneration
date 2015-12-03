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

extern int _num_tasks;
extern int _num_channels;
extern int _num_portmaps;

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

static int read_cuda_port(int channel_index, unsigned char *buf, int len) // blocking
{
    CHANNEL *channel;
    channel = &channels[channel_index];
    int cur_size=0;

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

    if(channel->start + len <= channel->buf + channel->max_size+sizeof(int) )
    {
        if(gpuChannelInfo[channel_index].cpu_gpu != 2)        CUDA_ERROR_CHECK(cudaMemcpy(buf, channel->start, len, cudaMemcpyHostToDevice));
        else if(gpuChannelInfo[channel_index].cpu_gpu == 2)   CUDA_ERROR_CHECK(cudaMemcpy(buf, channel->start, len, cudaMemcpyDeviceToDevice));
        channel->start += len;
    }
    else
    {
        int part = channel->max_size+sizeof(int) - (channel->start - channel->buf);
        if(part != 0)
        {
            if(gpuChannelInfo[channel_index].cpu_gpu != 2)        CUDA_ERROR_CHECK(cudaMemcpy(buf, channel->start, part, cudaMemcpyHostToDevice));
            else if(gpuChannelInfo[channel_index].cpu_gpu == 2)   CUDA_ERROR_CHECK(cudaMemcpy(buf, channel->start, part, cudaMemcpyDeviceToDevice));
        }
        if(gpuChannelInfo[channel_index].cpu_gpu != 2)        CUDA_ERROR_CHECK(cudaMemcpy(buf+part, channel->buf, len-part, cudaMemcpyHostToDevice));
        else if(gpuChannelInfo[channel_index].cpu_gpu == 2)   CUDA_ERROR_CHECK(cudaMemcpy(buf+part, channel->buf, len-part, cudaMemcpyDeviceToDevice));
        channel->start = channel->buf + (len - part);
    }

    channel->request_read = false;
    COND_BROADCAST(&(channel->cond));

    return len;
}

static int write_one_cuda_port(CHANNEL* channel, unsigned char *buf, int len) // blocking
{
    int cur_size=0;
    int channel_index = channel->channel_id;

    if(len < 0)
    {
        printf("write_port(): len should be larger than 0!\n");
        exit(EXIT_FAILURE);
    }

    if(len > channel->max_size)
    {
        printf("write_port(): max_size is too small!\n");
        exit(EXIT_FAILURE);
    }

write_start:

    if(channel->start <= channel->end)  cur_size = (unsigned int)(channel->end - channel->start);
    else                                cur_size = (unsigned int)(channel->end + channel->max_size + sizeof(int) - channel->start);

    if(len + cur_size > channel->max_size + sizeof(int)) // blocking case
    {
        MUTEX_LOCK(&(channel->mutex));
        channel->request_write = true;
        COND_BROADCAST(&(channel->cond));
        COND_WAIT(&(channel->cond), &(channel->mutex));
        MUTEX_UNLOCK(&(channel->mutex));
        goto write_start; // try writing again
    }

    if(channel->buf + channel->max_size+sizeof(int) >= channel->end + len)
    {
        if(gpuChannelInfo[channel_index].cpu_gpu != 2)        CUDA_ERROR_CHECK(cudaMemcpy(channel->end, buf, len, cudaMemcpyDeviceToHost));
        else if(gpuChannelInfo[channel_index].cpu_gpu == 2)   CUDA_ERROR_CHECK(cudaMemcpy(channel->end, buf, len, cudaMemcpyDeviceToDevice));
        channel->end += len;
    }
    else
    {
        int part = channel->max_size+sizeof(int) - (channel->end - channel->buf);
        if(part != 0)
        {
            if(gpuChannelInfo[channel_index].cpu_gpu != 2)        CUDA_ERROR_CHECK(cudaMemcpy(channel->end, buf, part, cudaMemcpyDeviceToHost));
            else if(gpuChannelInfo[channel_index].cpu_gpu == 2)   CUDA_ERROR_CHECK(cudaMemcpy(channel->end, buf, part, cudaMemcpyDeviceToDevice));
        }
        if(gpuChannelInfo[channel_index].cpu_gpu != 2)        CUDA_ERROR_CHECK(cudaMemcpy(channel->buf, buf+part, len-part, cudaMemcpyDeviceToHost));
        else if(gpuChannelInfo[channel_index].cpu_gpu == 2)   CUDA_ERROR_CHECK(cudaMemcpy(channel->buf, buf+part, len-part, cudaMemcpyDeviceToDevice));
        channel->end = channel->buf + (len - part);
    }

    channel->request_write = false;
    COND_BROADCAST(&(channel->cond));

    return len;
}

static int write_cuda_port(int channel_index, unsigned char *buf, int len) // blocking
{
    int result = 0;

    while(channel_index >= 0) {
        result = write_one_cuda_port(&channels[channel_index],buf,len);
        channel_index = channels[channel_index].next_channel_index;
    }

    return result;
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
    int channel_index=0;

    for(i=0; i<ARRAYLEN(__dev_buf); i++)
    {
        for(j=0; j<_num_portmaps; j++)
        {
            if(addressmap[j].task_id == TASK_INDEX && addressmap[j].port_id == i)
            {  
                channel_index = addressmap[j].channel_id;
                break;
            }
        }
        CUDA_ERROR_CHECK(cudaMalloc((void**) &__dev_buf[i], channels[channel_index].sampleSize)); 
        CUDA_ERROR_CHECK(cudaMemset((void*)__dev_buf[i], 0x0, channels[channel_index].sampleSize));
    }

}

TASK_PREGO
{
    int i, j=0;

    for(i=0; i<__num_of_channels; i++)
    {
        for(j=0; j<_num_portmaps; j++)
        {
            if(addressmap[j].task_id == TASK_INDEX && addressmap[j].port_id == i && addressmap[j].op == 'w')
            {
                CUDA_ERROR_CHECK(cudaMemset((void*)__dev_buf[i], 0x0, channels[addressmap[j].channel_id].sampleSize));
                break;
            }
        }
    }
}

TASK_POSTGO
{
}

TASK_POSTWRAPUP
{
    int i=0;
    for(i=0; i<ARRAYLEN(__dev_buf); i++)
    {
        CUDA_ERROR_CHECK(cudaFree(__dev_buf[i]));
    }
}
