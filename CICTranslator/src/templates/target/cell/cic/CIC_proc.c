
// produced by outputPTHREAD() of genPTHREAD.py

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include <pthread.h>

#include <unistd.h>
#include <sys/time.h>
#include <math.h>

//#define NDEBUG
#include <assert.h>

#ifdef THREAD_STYLE
#define THREAD_TYPE pthread_t
#define MUTEX_TYPE pthread_mutex_t
#define COND_TYPE pthread_cond_t
#endif

#include "cic_channels.h"
#include "cic_portmap.h"

// Added for jhw at 10.01.11 for library
#include "lib_portmap.h"

// For control feature
#include "control_info.h"
#include "param_list.h"
#include "con_channels.h"

//#include "time_stamp.h"

#define ARRAYLEN(ARR) (sizeof(ARR)/sizeof(ARR[0]))

#include "cic_tasks.h"

static void init_task(void)
{
    int i;
    for(i=0; i<ARRAYLEN(tasks); i++)
    {
        (*tasks[i].init)(i);
    }
}

static void wrapup_task(void)
{
    int i;
    for(i=0; i<ARRAYLEN(tasks); i++)
    {
        (*tasks[i].wrapup)();
    }
}


///////////////////////////////////////////// Previous task_routine /////////////////////////////////////////////////
/*
static void *task_routine(void *pdata)
{
    int i;
    int task_index = (int)pdata;

#if defined(PARAM_DEBUG) && (PARAM_DEBUG==1)
    if(task_index < 0 || task_index >= ARRAYLEN(tasks))
    {
        printf("task_routine(): Can't find task's index\n");
    }
#endif

    for(i=0; i<tasks[task_index].run_count; ++i)
    {
        (*tasks[task_index].go)();
    }

    return NULL;
}

static void go_task(void)
{
    int i;
    {
        for(i=0; i<ARRAYLEN(tasks); i++)
        {
            pthread_create(&(tasks[i].th), NULL, task_routine, (void *)i);
        }
        for(i=0; i<ARRAYLEN(tasks); i++)
        {
            pthread_join(tasks[i].th, NULL);
        }
    }
}
*/
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



//////////////////////////////////////////////////// task routine //////////////////////////////////////////////////////

volatile static bool all_threads_created = false;

// Added by jhw at 10.05.21 for data-driven tasks 
static void *data_task_routine(void *pdata) 
{ 
	int task_index = (int)pdata; 
	int old_state;
/*
    FILE* fp = NULL;
    char temp[255];
    char* task_name;
    
    if(tasks[task_index].is_Wrapper == 0)
    {
        task_name = temp;
        strcpy(task_name, "trace_");
        strcat(task_name, tasks[task_index].name);
        strcat(task_name, ".txt");

        fp = fopen(task_name, "w");
    }
*/
	pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, &old_state); 

	while(!all_threads_created); 

// run infinitely before pthread_cancel
	while(true) { 
/*
        if(tasks[task_index].is_Wrapper == 0)
        {
            TIMESTAMP(fp, 0, 0, task_index, -1, -1);
            fflush(fp);
            (*tasks[task_index].go)();
            TIMESTAMP(fp, 1, 0, task_index, -1, -1);
            fflush(fp);
        }
        else
            (*tasks[task_index].go)();
*/
		(*tasks[task_index].go)();
    }
/*
	if(tasks[task_index].is_Wrapper == 0)
		fclose(fp);
*/
	return NULL; 
} 

// Added by jhw for time-driven tasks
static void *time_task_routine(void *pdata) 
{ 
	int task_index = (int)pdata; 
	int iteration_count, count; 
/*
    FILE* fp = NULL;
    char temp[255];
    char* task_name;

    if(tasks[task_index].is_Wrapper == 0)
    {
        task_name = temp;
        strcpy(task_name, "trace_");
        strcat(task_name, tasks[task_index].name);
        strcat(task_name, ".txt");

        fp = fopen(task_name, "w");
    }
*/
	iteration_count = tasks[task_index].run_count; 

	while(!all_threads_created); 

// run number of iteration_count
    for(count = 0; count < iteration_count; ++count){ 
/*
        if(tasks[task_index].is_Wrapper == 0)
        {
            TIMESTAMP(fp, 0, 0, task_index, -1, -1);
            fflush(fp);
            (*tasks[task_index].go)();
            TIMESTAMP(fp, 1, 0, task_index, -1, -1);
            fflush(fp);
        }
        else
            (*tasks[task_index].go)();
*/
    	(*tasks[task_index].go)();
    }
/*
    if(tasks[task_index].is_Wrapper == 0)
    	fclose(fp);
*/
	return NULL; 
}

// For control scheduler
typedef struct{
	pthread_t cs_thread;
	pthread_mutex_t cs_mutex;
	pthread_cond_t cs_cond;
}Control_Scheduler;

// One is push scheduler, the other is pop scheduler
static Control_Scheduler control_scheduler[2];

// Added by jhw for runonce tasks
static void *runonce_task_routine(void *pdata) 
{ 
	int task_index = (int)pdata; 
	int result = 0; 
	int old_state;
/*
	FILE* fp = NULL;
	char temp[255];
	char* task_name;
*/
	pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, &old_state);

	result = pthread_mutex_lock(&(tasks[task_index].p_mutex));
	if(result != 0)
	{
		printf("runonce_task_routine() : pthread_mutex_lock() failed!\n");
		exit(-1);
	}
/*
	if(tasks[task_index].is_Wrapper == 0)
	{
		task_name = temp;
		strcpy(task_name, "trace_");
		strcat(task_name, tasks[task_index].name);
		strcat(task_name, ".txt");

		fp = fopen(task_name, "w");
	}
*/
	while(!all_threads_created);

// run if state of task is Run (if not, task_routine thread will sleep)
	while(true)
	{
		if(tasks[task_index].state != Run)
		{
			result = pthread_cond_wait(&(tasks[task_index].p_cond), &(tasks[task_index].p_mutex));
			if(result != 0)
			{
				printf("runonce_task_routine() : pthread_cond_wait() failed!\n");
				exit(-1);
			}
		}
		else
		{
/*
			if(tasks[task_index].is_Wrapper == 0)
			{
				TIMESTAMP(fp, 0, 0, task_index, -1, -1);
				fflush(fp);
				(*tasks[task_index].go)();
				TIMESTAMP(fp, 1, 0, task_index, -1, -1);
				fflush(fp);
			}
			else
				(*tasks[task_index].go)();
*/
			(*tasks[task_index].go)();
			tasks[task_index].state = Wait;
			pthread_cond_broadcast(&(control_scheduler[1].cs_cond));
		}
	}
/*
	if(tasks[task_index].is_Wrapper == 0)
		fclose(fp);
*/
	result = pthread_mutex_unlock(&(tasks[task_index].p_mutex));
	if(result != 0)
	{
		printf("runonce_task_routine() : pthread_mutex_unlock() failed!\n");
		exit(-1);
	}

	return NULL;
}

// Declaration of control schedulers (implementations :  from 2932 line~)
static void *control_scheduler_push_routine(void*);
static void *control_scheduler_pop_routine(void*);

// execute all go() functions of tasks
static void go_task(void)
{
	int i=0;

// run all types of task.    
	for(i=0; i<ARRAYLEN(tasks); i++)
	{
#ifdef WIN32
		tasks[i].thread = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE) task_routine, (void*)&i, NULL,NULL);
#else
		if(tasks[i].driven_type == TimeDriven)
			pthread_create(&(tasks[i].thread), NULL, time_task_routine, (void *)i);
		else if(tasks[i].driven_type == DataDriven)
			pthread_create(&(tasks[i].thread), NULL, data_task_routine, (void *)i);
		else if(tasks[i].driven_type == RunOnce)
			pthread_create(&(tasks[i].thread), NULL, runonce_task_routine, (void *)i);
		else
			continue;
#endif
	}

// run control schedulers
    if(ARRAYLEN(control_channel) != 0)
    {
        pthread_create(&(control_scheduler[0].cs_thread), NULL, control_scheduler_push_routine, NULL);
	    pthread_create(&(control_scheduler[1].cs_thread), NULL, control_scheduler_pop_routine, NULL);
    }

// to run all tasks & schedulers at the same time
	all_threads_created = true;

// to join time-driven tasks
	for(i=0; i<ARRAYLEN(tasks); i++)
	{
		if(tasks[i].driven_type == TimeDriven){
			void *result;
			pthread_join(tasks[i].thread, &result);
		}
	}

// wait 3 seconds for data-driven tasks
	sleep(3);

// to cancel all tasks except time-driven tasks
	for(i=0; i<ARRAYLEN(tasks); i++)
	{
		if(tasks[i].driven_type != TimeDriven){
			pthread_cancel(tasks[i].thread);
		}

	}
// to cancel control schedulers 
    if(ARRAYLEN(control_channel) != 0)
    {
	    for(i=0; i<2; i++)
	    {
		    pthread_cancel(control_scheduler[i].cs_thread);
	    }
    }

	return;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////





/////////////////////////////////////////////////// Library ////////////////////////////////////////////////////////////////////
#include "lib_stubs.h"
#include "lib_refs.h"

static void init_lib(void)
{
	int i;
	for(i=0; i<ARRAYLEN(stubs); i++)
	{
		(*stubs[i].init)();
	}
	for(i=0; i<ARRAYLEN(lib_refs); i++)
	{
		(*lib_refs[i].init)();
	}
}

static void wrapup_lib(void)
{
	int i;

	for(i=0; i<ARRAYLEN(stubs); i++)
	{
		(*stubs[i].wrapup)();
	}
	for(i=0; i<ARRAYLEN(lib_refs); i++)
	{
		(*lib_refs[i].wrapup)();
	}
}

static void *stub_routine(void *pdata)
{
	int stub_index = (int)pdata;

#if defined(PARAM_DEBUG) && (PARAM_DEBUG==1)
	if(stub_index < 0 || stub_index >= ARRAYLEN(stubs))
	{
		printf("task_routine(): Can't find task's index\n");
    }
#endif

    (*stubs[stub_index].go)();

    return NULL;
}

static void go_stub(void)
{
    int i;
    {
        for(i=0; i<ARRAYLEN(stubs); i++)
        {
            pthread_create(&(stubs[i].th), NULL, stub_routine, (void *)i);
        }
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////




static void init_channel(void)
{
    int i;
    for(i=0; i<ARRAYLEN(channels); i++)
    {
        if( 0 != pthread_mutex_init(&(channels[i].mutex), NULL))
        {
            printf("init_channel(): pthread_mutex_init() failed!\n");
            exit(EXIT_FAILURE);
        }
        if( 0 != pthread_cond_init(&(channels[i].cond), NULL))
        {
            printf("init_channel(): pthread_cond_init() failed!\n");
            exit(EXIT_FAILURE);
        }
        if(channels[i].type == CHANNEL_TYPE_NORMAL)
        {
            unsigned char *ptr;
            ptr = (unsigned char *)malloc(channels[i].max_size);
            if(ptr == NULL)
            {
                printf("initialize_port(): malloc failed!\n");
                exit(EXIT_FAILURE);
            }
            channels[i].buf = ptr;
            channels[i].start = ptr;
            memset(ptr, 0x0, channels[i].initData);
            channels[i].end = ptr + channels[i].initData;
            channels[i].cur_size = channels[i].initData;
        }
        else if(channels[i].type == CHANNEL_TYPE_ARRAY_CHANNEL)
        {
            if(channels[i].max_size <= 0)
            {
                printf("init_channel(): size should be positive!\n");
                exit(EXIT_FAILURE);
            }

            if(channels[i].sampleSize <= 0)
            {
                printf("init_channel(): sampleSize should be positive!\n");
                exit(EXIT_FAILURE);
            }

            if(channels[i].max_size % channels[i].sampleSize != 0)
            {
                printf("init_channel(): size should be divided by sampleSize!\n");
                exit(EXIT_FAILURE);
            }

            if(channels[i].max_size / channels[i].sampleSize < channels[i].initData)
            {
                printf("init_channel(): initData is too large!\n");
                exit(EXIT_FAILURE);
            }

            {
                int j;
                const int count = channels[i].max_size / channels[i].sampleSize;
                channels[i].head = malloc(count * sizeof(AC_DATA));
                if(channels[i].head == NULL)
                {
                    printf("init_channel(): malloc() failed!\n");
                    exit(EXIT_FAILURE);
                }

                channels[i].avail_index_start = channels[i].avail_index_end = NULL;

                for(j=0;j<count;j++)
                {
                    channels[i].head[j].avail_node = malloc(sizeof(AC_AVAIL_LIST));
                    if(channels[i].head[j].avail_node == NULL)
                    {
                        printf("init_channel(): malloc() failed!\n");
                        exit(EXIT_FAILURE);
                    }
                    channels[i].head[j].avail_node->avail_index = j;
                    channels[i].head[j].avail_node->prev = channels[i].head[j].avail_node->next = NULL;

                    if(j < channels[i].initData)
                    {
                        channels[i].head[j].used = 1;
                        {
                            AC_DATA *ptr;
                            CHANNEL *channel;
                            ptr = &(channels[i].head[j]);
                            channel = &(channels[i]);
                            if(channel->avail_index_start == NULL)
                            {
                                assert(channel->avail_index_end == NULL);
                                channel->avail_index_start = channel->avail_index_end = ptr->avail_node;
                                assert(channel->avail_index_start->prev == NULL);
                                assert(channel->avail_index_start->next == NULL);
                            }
                            else
                            {
                                AC_AVAIL_LIST *node;
                                assert(channel->avail_index_end != NULL);
                                node = channel->avail_index_end;
                                channel->avail_index_end = node->next = ptr->avail_node;
                                ptr->avail_node->prev = node;
                                ptr->avail_node->next = NULL;
                            }
                        }
                    }
                    else
                    {
                        channels[i].head[j].used = 0;
                    }
                    channels[i].head[j].buf = malloc(channels[i].sampleSize);
                    if(channels[i].head[j].buf == NULL)
                    {
                        printf("init_channel(): malloc() failed!\n");
                        exit(EXIT_FAILURE);
                    }
                }
            }
        }
    }
}

static void wrapup_channel(void)
{
    int i;
    for(i=0; i<ARRAYLEN(channels); i++)
    {
        if(channels[i].type == CHANNEL_TYPE_NORMAL)
        {
            free(channels[i].buf);
        }
        else if(channels[i].type == CHANNEL_TYPE_ARRAY_CHANNEL)
        {
            int j;
            const int count = channels[i].max_size / channels[i].sampleSize;
            for(j=0;j<count;j++)
            {
                free(channels[i].head[j].avail_node);
                free(channels[i].head[j].buf);
            }
            free(channels[i].head);
        }
    }
}

// Added by jhw at 10.01.07 for library
#include "lib_channels.h"

static void init_lib_channel(void)
{
    int i,k;
    LIB_CHANNEL_UNIT *lib_channel_temp = NULL;

    for(i=0; i<ARRAYLEN(lib_channels); i++)
    {
        if( 0 != pthread_mutex_init(&(lib_channels[i].l_mutex), NULL))
        {
            printf("init_lib_channel(): pthread_mutex_init() failed!\n");
            exit(EXIT_FAILURE);
        }

        if( 0 != pthread_cond_init(&(lib_channels[i].l_cond), NULL))
        {
            printf("init_lib_channel(): pthread_cond_init() failed!\n");
            exit(EXIT_FAILURE);
        }

	for(k=0; k<2; k++)
	{
		if(k == 0)	lib_channel_temp = &lib_channels[i].func_channel; 
		else if(k == 1)	lib_channel_temp = &lib_channels[i].ret_channel; 

		if( 0 != pthread_mutex_init(&(lib_channel_temp->mutex), NULL))
		{
			printf("init_lib_channel(): pthread_mutex_init() failed!\n");
			exit(EXIT_FAILURE);
		}

		if( 0 != pthread_cond_init(&(lib_channel_temp->cond), NULL))
		{
			printf("init_lib_channel(): pthread_cond_init() failed!\n");
			exit(EXIT_FAILURE);
		}

	
		if(lib_channel_temp->type == CHANNEL_TYPE_ARRAY_CHANNEL)
		{
			if(lib_channel_temp->max_size <= 0)
			{
				printf("init_lib_channel(): size should be positive!\n");
				exit(EXIT_FAILURE);
			}

			if(lib_channel_temp->sampleSize <= 0)
			{
				printf("init_lib_channel(): sampleSize should be positive!\n");
				exit(EXIT_FAILURE);
			}

			if(lib_channel_temp->max_size % lib_channel_temp->sampleSize != 0)
			{
				printf("init_lib_channel(): size should be divided by sampleSize!\n");
				exit(EXIT_FAILURE);
			}

			if(lib_channel_temp->max_size / lib_channel_temp->sampleSize < lib_channel_temp->initData)
			{
				printf("init_lib_channel(): initData is too large!\n");
				exit(EXIT_FAILURE);
			}

			{
				int j;
				const int count = lib_channel_temp->max_size / lib_channel_temp->sampleSize;
				lib_channel_temp->head = malloc(count * sizeof(AC_DATA));

				if(lib_channel_temp->head == NULL)
				{
					printf("init_lib_channel(): malloc() failed!\n");
					exit(EXIT_FAILURE);
				}

				lib_channel_temp->avail_index_start = lib_channel_temp->avail_index_end = NULL;

				for(j=0;j<count;j++)
				{
					lib_channel_temp->head[j].avail_node = malloc(sizeof(AC_AVAIL_LIST));
					if(lib_channel_temp->head[j].avail_node == NULL)
					{
						printf("init_lib_channel(): malloc() failed!\n");
						exit(EXIT_FAILURE);
					}
					lib_channel_temp->head[j].avail_node->avail_index = j;
					lib_channel_temp->head[j].avail_node->prev = lib_channel_temp->head[j].avail_node->next = NULL;

					if(j < lib_channel_temp->initData)
					{
						lib_channel_temp->head[j].used = 1;
						{
							AC_DATA *ptr;
							LIB_CHANNEL_UNIT *channel;
							ptr = &(lib_channel_temp->head[j]);
							channel = (lib_channel_temp);
							if(channel->avail_index_start == NULL)
							{
								assert(channel->avail_index_end == NULL);
								channel->avail_index_start = channel->avail_index_end = ptr->avail_node;
								assert(channel->avail_index_start->prev == NULL);
								assert(channel->avail_index_start->next == NULL);
							}
							else
							{
								AC_AVAIL_LIST *node;
								assert(channel->avail_index_end != NULL);
								node = channel->avail_index_end;
								channel->avail_index_end = node->next = ptr->avail_node;
								ptr->avail_node->prev = node;
								ptr->avail_node->next = NULL;
							}
						}
					}
					else
					{
						lib_channel_temp->head[j].used = 0;
					}
					lib_channel_temp->head[j].buf = malloc(lib_channel_temp->sampleSize);
					if(lib_channel_temp->head[j].buf == NULL)
					{
						printf("init_lib_channel(): malloc() failed!\n");
						exit(EXIT_FAILURE);
					}
				}
			}
		}
	}
    }

}

// Added by jhw at 10.01.07 for library
static void wrapup_lib_channel(void)
{
    int i, k;
    LIB_CHANNEL_UNIT *lib_channel_temp = NULL;

    for(i=0; i<ARRAYLEN(lib_channels); i++)
    {
	for(k=0; k<2; k++)
	{
		if(k==0)	lib_channel_temp = &lib_channels[i].func_channel;
		else if(k==1)	lib_channel_temp = &lib_channels[i].ret_channel;

        	if(lib_channel_temp->type == CHANNEL_TYPE_ARRAY_CHANNEL)
        	{	
            		int j;
            		const int count = (lib_channel_temp->max_size) / (lib_channel_temp->sampleSize);
            		for(j=0;j<count;j++)
            		{
                		free(lib_channel_temp->head[j].avail_node);
                		free(lib_channel_temp->head[j].buf);
            		}
            		free(lib_channel_temp->head);
        	}
    	}
    }
}

// Added by jhw at 10.01.11 for library
int init_lib_port(int stub_id)
{
    int i, channel_id;
    for(i=0; i<ARRAYLEN(lib_addressmap); i++)
    {
       if(lib_addressmap[i].stub_id == stub_id)
       {
	   channel_id = lib_addressmap[i].channel_id;
	   return channel_id;
       }
    }

    printf("init_lib_port(): Can't find channel(task id: %d)\n", stub_id);
    exit(EXIT_FAILURE);
}

int init_port(int task_id, int port_id)
{
    int i, channel_id;
    for(i=0; i<ARRAYLEN(addressmap); i++)
    {
        if(addressmap[i].task_id == task_id && addressmap[i].port_id == port_id)
        {
            channel_id = addressmap[i].channel_id;
            return channel_id;
        }
    }

    printf("init_port(): Can't find channel(task id: %d, port id: %d)\n", task_id, port_id);
    exit(EXIT_FAILURE);
}

// Add by jhw at 09.12.22
int init_task_port(int task_id, const char* port_name)
{
    unsigned int i;
    int channel_id;

    for(i=0; i<ARRAYLEN(addressmap); i++)
    {
        if(addressmap[i].task_id == task_id && (strcmp(addressmap[i].port_name,port_name)==0))
        {
            channel_id = addressmap[i].channel_id;
            unsigned int j;
            for(j=0; j<ARRAYLEN(channels); j++)
                if(channels[j].channel_id == channel_id)
                    return j;
        }
    }

    printf("init_port(): Can't find channel(task id: %d, port name: %s)\n", task_id, port_name);
    exit(EXIT_FAILURE);
}

static CHANNEL *get_channel(int channel_id)
{
    int i;
    for(i=0;i<ARRAYLEN(channels);i++)
    {
        if(channels[i].channel_id == channel_id)
        {
            return &(channels[i]);
        }
    }
    printf("get_channel(): Can't find channel(channel_id: %d)\n", channel_id);
    exit(EXIT_FAILURE);
}

// Added by jhw at 10.01.11 for library
static LIB_CHANNEL *get_lib_channel(int channel_id)
{
    int i;
    for(i=0;i<ARRAYLEN(lib_channels);i++)
    {
        if(lib_channels[i].channel_id == channel_id)
        {
            return &(lib_channels[i]);
        }
    }
    printf("get_channel(): Can't find channel(channel_id: %d)\n", channel_id);
    exit(EXIT_FAILURE);
}

int get_mytask_id(void)
{
    int i;
    for(i=0; i<ARRAYLEN(tasks); i++)
    {
        if(tasks[i].thread == pthread_self())
        {
            return tasks[i].task_id;
        }
    }
    printf("PROC_DEBUG(ERR) get_mytask_id(): Can't find task's id\n");
    exit(EXIT_FAILURE);
}

#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
static char *get_task_name(int task_id)
{
    int i;
    for(i=0; i<ARRAYLEN(tasks); i++)
    {
        if(tasks[i].task_id == task_id)
        {
            return tasks[i].name;
        }
    }
    printf("PROC_DEBUG(ERR) get_task_name(): Can't find task's name\n");
    exit(EXIT_FAILURE);
}

static void check_valid_operation(int channel_id, CHANNEL_TYPE type, char op)
{
    int task_id = get_mytask_id();
    int i;

    CHANNEL *channel;
    channel = get_channel(channel_id);
    if(channel->type != type)
    {
        printf("PROC_DEBUG(ERR) check_valid_operation(): channel %d type mismatch!\n", channel_id);
        exit(EXIT_FAILURE);
    }

    for(i=0; i<ARRAYLEN(addressmap); i++)
    {
        if(addressmap[i].task_id == task_id &&
                addressmap[i].channel_id == channel_id &&
                addressmap[i].op == op)
        {
            return;
        }
    }
    if(op == 'r')
        printf("PROC_DEBUG(ERR) read_port(): channel %d is not for %s\n", channel_id, get_task_name(get_mytask_id()));
    else if(op == 'w')
        printf("PROC_DEBUG(ERR) write_port(): channel %d is not for %s\n", channel_id, get_task_name(get_mytask_id()));
    else
        printf("PROC_DEBUG(ERR) check_valid_operation(): fatal error!\n");
    exit(EXIT_FAILURE);
}
#endif



int read_port(int channel_id, unsigned char *buf, int len) // blocking
{
    int result;

    CHANNEL *channel;
    channel = get_channel(channel_id);

#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
    {
        int i;
        check_valid_operation(channel_id, CHANNEL_TYPE_NORMAL, 'r');
        for(i=0; i<ARRAYLEN(addressmap); i++)
        {
            if(addressmap[i].task_id != get_mytask_id() &&
                    addressmap[i].channel_id == channel_id &&
                    addressmap[i].op == 'w')
            {
                i = addressmap[i].task_id;
                break;
            }
        }
        printf("PROC_DEBUG(CH) %s read %d(from'%s') %d %d/%d start\n", get_task_name(get_mytask_id()), channel_id, get_task_name(i), len, channel->cur_size, channel->max_size);
    }
#endif

#if defined(PARAM_DEBUG) && (PARAM_DEBUG==1)
    if(len < 0)
    {
        printf("%d %d\n", channel_id, len);
        printf("read_port(): len should be larger than 0!\n");
        exit(EXIT_FAILURE);
    }

    if(len > channel->max_size)
    {
        printf("read_port(): max_size is too small!\n");
        exit(EXIT_FAILURE);
    }
#endif

read_start:

    result = pthread_mutex_lock(&(channel->mutex));
    if(result != 0)
    {
        printf("read_port(): pthread_mutex_lock() failed!\n");
        exit(EXIT_FAILURE);
    }

    if(len > channel->cur_size) // blocking case
    {
        channel->request_read = true;
        if(channel->request_write == true)
        {
            printf("read_port(): deadlock!\n");
            exit(EXIT_FAILURE);
        }
        result = pthread_cond_wait(&(channel->cond), &(channel->mutex));
        if(result != 0)
        {
            printf("read_port(): pthread_cond_wait() failed!\n");
            exit(EXIT_FAILURE);
        }
        result = pthread_mutex_unlock(&(channel->mutex));
        if(result != 0)
        {
            printf("read_port(): pthread_mutex_unlock() failed!\n");
            exit(EXIT_FAILURE);
        }
        goto read_start; // try reading again
    }

    if(channel->start + len <= channel->buf + channel->max_size )
    {
        memcpy(buf, channel->start, len);
        channel->start += len;
    }
    else
    {
        int part = channel->max_size - (channel->start - channel->buf);
        if(part != 0)
        {
            memcpy(buf, channel->start, part);
        }
        memcpy(buf + part, channel->buf, len - part);
        channel->start = channel->buf + (len - part);
    }

    channel->request_read = false;
    channel->request_write = false;
    channel->cur_size -= len;

    result = pthread_cond_broadcast(&(channel->cond));
    if(result != 0)
    {
        printf("read_port(): pthread_cond_broadcast() failed!\n");
        exit(EXIT_FAILURE);
    }
    result = pthread_mutex_unlock(&(channel->mutex));
    if(result != 0)
    {
        printf("read_port(): pthread_cond_unlock() failed!\n");
        exit(EXIT_FAILURE);
    }

#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
    {
        int i;
        for(i=0; i<ARRAYLEN(addressmap); i++)
        {
            if(addressmap[i].task_id != get_mytask_id() &&
                    addressmap[i].channel_id == channel_id &&
                    addressmap[i].op == 'w')
            {
                i = addressmap[i].task_id;
                break;
            }
        }
        printf("PROC_DEBUG(CH) %s read %d(from'%s') %d %d/%d end\n", get_task_name(get_mytask_id()), channel_id, get_task_name(i), len, channel->cur_size, channel->max_size);
    }
#endif

    return len;
}

int write_one_port(int channel_id, unsigned char *buf, int len) // blocking
{
    int result;

    CHANNEL *channel;
    channel = get_channel(channel_id);

#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
    {
        int i;
        check_valid_operation(channel_id, CHANNEL_TYPE_NORMAL, 'w');
        for(i=0; i<ARRAYLEN(addressmap); i++)
        {
            if(addressmap[i].task_id != get_mytask_id() &&
                    addressmap[i].channel_id == channel_id &&
                    addressmap[i].op == 'r')
            {
                i = addressmap[i].task_id;
                break;
            }
        }
        printf("PROC_DEBUG(CH) %s write %d(to'%s') %d %d/%d start\n", get_task_name(get_mytask_id()), channel_id, get_task_name(i), len, channel->cur_size, channel->max_size);
    }
#endif

#if defined(PARAM_DEBUG) && (PARAM_DEBUG==1)
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
#endif

write_start:

    result = pthread_mutex_lock(&(channel->mutex));
    if(result != 0)
    {
        printf("write_port(): pthread_mutex_lock() failed!\n");
        exit(EXIT_FAILURE);
    }

    if(len + channel->cur_size > channel->max_size) // blocking case
    {
        channel->request_write = true;
        if(channel->request_read == true)
        {
            printf("write_port(): deadlock!\n");
            exit(EXIT_FAILURE);
        }
        result = pthread_cond_wait(&(channel->cond), &(channel->mutex));
        if(result != 0)
        {
            printf("write_port(): pthread_cond_wait() failed!\n");
            exit(EXIT_FAILURE);
        }
        result = pthread_mutex_unlock(&(channel->mutex));
        if(result != 0)
        {
            printf("write_port(): pthread_mutex_unlock() failed!\n");
            exit(EXIT_FAILURE);
        }
        goto write_start; // try writing again
    }

    if(channel->buf + channel->max_size >= channel->end + len)
    {
        memcpy(channel->end, buf, len);
        channel->end += len;
    }
    else
    {
        int part = channel->max_size - (channel->end - channel->buf);
        if(part != 0)
        {
            memcpy(channel->end, buf, part);
        }
        memcpy(channel->buf, buf + part, len - part);
        channel->end = channel->buf + (len - part);
    }

    channel->request_write = false;
    channel->request_read = false;
    channel->cur_size += len;

    result = pthread_cond_broadcast(&(channel->cond));
    if(result != 0)
    {
        printf("write_port(): pthread_cond_broadcast() failed!\n");
        exit(EXIT_FAILURE);
    }
    result = pthread_mutex_unlock(&(channel->mutex));
    if(result != 0)
    {
        printf("write_port(): pthread_mutex_unlock() failed!\n");
        exit(EXIT_FAILURE);
    }

#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
    {
        int i;
        for(i=0; i<ARRAYLEN(addressmap); i++)
        {
            if(addressmap[i].task_id != get_mytask_id() &&
                    addressmap[i].channel_id == channel_id &&
                    addressmap[i].op == 'r')
            {
                i = addressmap[i].task_id;
                break;
            }
        }
        printf("PROC_DEBUG(CH) %s write %d(to'%s') %d %d/%d end\n", get_task_name(get_mytask_id()), channel_id, get_task_name(i), len, channel->cur_size, channel->max_size);
    }
#endif

    return len;
}

int write_port(int channel_index, unsigned char *buf, int len) // blocking
{
    int result = 0;

    while(channel_index >= 0) {
        result = write_one_port(channel_index, buf, len);
        channel_index = channels[channel_index].next_channel_index;
    }

    return result;
}

int available(int channel_id) // non-blocking
{
    int ret;
    int result;

    CHANNEL *channel;
    channel = get_channel(channel_id);

    result = pthread_mutex_lock(&(channel->mutex));
    if(result != 0)
    {
        printf("available(): pthread_mutex_lock() failed!\n");
        exit(EXIT_FAILURE);
    }

    ret = channel->cur_size;

    result = pthread_mutex_unlock(&(channel->mutex));
    if(result != 0)
    {
        printf("available(): pthread_mutex_unlock() failed!\n");
        exit(EXIT_FAILURE);
    }

    return ret;
}

int read_acport(int channel_id, unsigned char *buf, int len, int index)
{
    int result;
    AC_DATA *ptr;
    CHANNEL *channel;

    channel = get_channel(channel_id);
    ptr = &channel->head[index];

#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
    {
        int i;
        check_valid_operation(channel_id, CHANNEL_TYPE_ARRAY_CHANNEL, 'r');
        for(i=0; i<ARRAYLEN(addressmap); i++)
        {
            if(addressmap[i].task_id != get_mytask_id() &&
                    addressmap[i].channel_id == channel_id &&
                    addressmap[i].op == 'w')
            {
                i = addressmap[i].task_id;
                break;
            }
        }
        printf("PROC_DEBUG(CH) %s acread %d(from'%s') %d %d start\n", get_task_name(get_mytask_id()), channel_id, get_task_name(i), len, index);
    }
#endif

#if defined(PARAM_DEBUG) && (PARAM_DEBUG==1)
    if(len != channel->sampleSize)
    {
        printf("read_acport(): len should be %d!\n", channel->sampleSize);
        exit(EXIT_FAILURE);
    }

    if(index < 0)
    {
        printf("read_acport(): index should be non-negative!\n");
        exit(EXIT_FAILURE);
    }

    if(index >= channel->max_size/channel->sampleSize)
    {
        printf("read_acport(): index is too large!\n");
        exit(EXIT_FAILURE);
    }
#endif

read_ac_start:

    result = pthread_mutex_lock(&(channel->mutex));
    if(result != 0)
    {
        printf("read_acport(): pthread_mutex_lock() failed!\n");
        exit(EXIT_FAILURE);
    }

    if(ptr->used == 0) // blocking case
    {
#if 0
        channel->request_read = true;
        if(channel->request_write == true)
        {
            printf("read_acport(): deadlock!\n");
            exit(EXIT_FAILURE);
        }
#endif
        result = pthread_cond_wait(&(channel->cond), &(channel->mutex));
        if(result != 0)
        {
            printf("read_acport(): pthread_cond_wait() failed!\n");
            exit(EXIT_FAILURE);
        }
        result = pthread_mutex_unlock(&(channel->mutex));
        if(result != 0)
        {
            printf("read_acport(): pthread_mutex_unlock() failed!\n");
            exit(EXIT_FAILURE);
        }
        goto read_ac_start;
    }

#if 0
    channel->request_read = false;
#endif
    memcpy(buf, ptr->buf, len);

    // for check_acport()
    {
        AC_AVAIL_LIST *node;
        node = ptr->avail_node;

        if(ptr->used == 2) // already checked case
        {
            ; // do nothing
            assert(node->prev == NULL);
            assert(node->next == NULL);
        }
        else
        {
            if(node->prev != NULL)
            {
                assert(node != channel->avail_index_start);
                node->prev->next = node->next;
            }
            else
            {
                assert(node == channel->avail_index_start);
                channel->avail_index_start = node->next;
            }
            if(node->next != NULL)
            {
                assert(node != channel->avail_index_end);
                node->next->prev = node->prev;
            }
            else
            {
                assert(node == channel->avail_index_end);
                channel->avail_index_end = node->prev;
            }
            node->prev = node->next = NULL;
        }
    }
    ptr->used = 0;

    result = pthread_mutex_unlock(&(channel->mutex));
    if(result != 0)
    {
        printf("read_acport(): pthread_mutex_unlock() failed!\n");
        exit(EXIT_FAILURE);
    }
    result = pthread_cond_broadcast(&(channel->cond));
    if(result != 0)
    {
        printf("read_acport(): pthread_cond_broadcast() failed!\n");
        exit(EXIT_FAILURE);
    }

#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
    {
        int i;
        for(i=0; i<ARRAYLEN(addressmap); i++)
        {
            if(addressmap[i].task_id != get_mytask_id() &&
                    addressmap[i].channel_id == channel_id &&
                    addressmap[i].op == 'w')
            {
                i = addressmap[i].task_id;
                break;
            }
        }
        printf("PROC_DEBUG(CH) %s acread %d(from'%s') %d %d end\n", get_task_name(get_mytask_id()), channel_id, get_task_name(i), len, index);
    }
#endif

    return len;
}

int write_one_acport(int channel_id, unsigned char *buf, int len, int index)
{
    int result;
    AC_DATA *ptr;
    CHANNEL *channel;

    channel = get_channel(channel_id);
    ptr = &channel->head[index];

#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
    {
        int i;
        check_valid_operation(channel_id, CHANNEL_TYPE_ARRAY_CHANNEL, 'w');
        for(i=0; i<ARRAYLEN(addressmap); i++)
        {
            if(addressmap[i].task_id != get_mytask_id() &&
                    addressmap[i].channel_id == channel_id &&
                    addressmap[i].op == 'r')
            {
                i = addressmap[i].task_id;
                break;
            }
        }
        printf("PROC_DEBUG(CH) %s acwrite %d(to'%s') %d %d start\n", get_task_name(get_mytask_id()), channel_id, get_task_name(i), len, index);
    }
#endif

#if defined(PARAM_DEBUG) && (PARAM_DEBUG==1)
    if(len != channel->sampleSize)
    {
        printf("write_acport(): len should be %d!\n", channel->sampleSize);
        exit(EXIT_FAILURE);
    }

    if(index < 0)
    {
        printf("write_acport(): index should be non-negative!\n");
        exit(EXIT_FAILURE);
    }

    if(index >= channel->max_size/channel->sampleSize)
    {
        printf("write_acport(): index is too large!\n");
        exit(EXIT_FAILURE);
    }
#endif

write_ac_start:

    result = pthread_mutex_lock(&(channel->mutex));
    if(result != 0)
    {
        printf("write_acport(): pthread_mutex_lock() failed!\n");
        exit(EXIT_FAILURE);
    }

    if(ptr->used != 0) // blocking case
    {
#if 0
        channel->request_write = true;
        if(channel->request_read == true)
        {
            printf("write_acport(): deadlock!\n");
            exit(EXIT_FAILURE);
        }
#endif
        result = pthread_cond_wait(&(channel->cond), &(channel->mutex));
        if(result != 0)
        {
            printf("read_acport(): pthread_cond_wait() failed!\n");
            exit(EXIT_FAILURE);
        }
        result = pthread_mutex_unlock(&(channel->mutex));
        if(result != 0)
        {
            printf("read_acport(): pthread_mutex_unlock() failed!\n");
            exit(EXIT_FAILURE);
        }
        goto write_ac_start;
    }

#if 0
    channel->request_write = false;
#endif
    memcpy(ptr->buf, buf, len);
    ptr->used = 1;

    // for check_acport()
    {
        if(channel->avail_index_start == NULL)
        {
            assert(channel->avail_index_end == NULL);
            channel->avail_index_start = channel->avail_index_end = ptr->avail_node;
            assert(channel->avail_index_start->prev == NULL);
            assert(channel->avail_index_start->next == NULL);
        }
        else
        {
            AC_AVAIL_LIST *node;
            assert(channel->avail_index_end != NULL);
            node = channel->avail_index_end;
            channel->avail_index_end = node->next = ptr->avail_node;
            ptr->avail_node->prev = node;
            ptr->avail_node->next = NULL;
        }
    }

    result = pthread_mutex_unlock(&(channel->mutex));
    if(result != 0)
    {
        printf("write_acport(): pthread_mutex_unlock() failed!\n");
        exit(EXIT_FAILURE);
    }
    result = pthread_cond_broadcast(&(channel->cond));
    if(result != 0)
    {
        printf("write_acport(): pthread_cond_broadcast() failed!\n");
        exit(EXIT_FAILURE);
    }

#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
    {
        int i;
        for(i=0; i<ARRAYLEN(addressmap); i++)
        {
            if(addressmap[i].task_id != get_mytask_id() &&
                    addressmap[i].channel_id == channel_id &&
                    addressmap[i].op == 'r')
            {
                i = addressmap[i].task_id;
                break;
            }
        }
        printf("PROC_DEBUG(CH) %s acwrite %d(to'%s') %d %d end\n", get_task_name(get_mytask_id()), channel_id, get_task_name(i), len, index);
    }
#endif

    return len;
}

int write_acport(int channel_index, unsigned char *buf, int len, int index) // blocking
{
    int result = 0;

    while(channel_index >= 0) {
        result = write_one_acport(channel_index, buf, len, index);
        channel_index = channels[channel_index].next_channel_index;
    }

    return result;
}

int ac_available (int channel_id, int index)
{
    CHANNEL *channel;
    channel = get_channel(channel_id);

    if(channel->head[index].used != 0)
    {
        return 1;
    }
    else if(channel->head[index].used == 0)
    {
        return 0;
    }
    else
    {
        printf("ac_available(): unknown status!\n");
        exit(EXIT_FAILURE);
    }
}

int check_acport(int channel_id)
{
    int ret = -1;
    int result;

    CHANNEL *channel;
    channel = get_channel(channel_id);

check_ac_start:

    result = pthread_mutex_lock(&(channel->mutex));
    if(result != 0)
    {
        printf("check_acport(): pthread_mutex_lock() failed!\n");
        exit(EXIT_FAILURE);
    }

    {
        AC_AVAIL_LIST *node;
        node = channel->avail_index_start;

        if(node == NULL) // blocking case
        {
            int result;
            result = pthread_cond_wait(&(channel->cond), &(channel->mutex));
            if(result != 0)
            {
                printf("check_acport(): pthread_cond_wait() failed!\n");
                exit(EXIT_FAILURE);
            }
            result = pthread_mutex_unlock(&(channel->mutex));
            if(result != 0)
            {
                printf("check_acport(): pthread_mutex_unlock() failed!\n");
                exit(EXIT_FAILURE);
            }
            goto check_ac_start;
        }
        else
        {
            ret = node->avail_index;
            channel->avail_index_start = channel->avail_index_start->next;

            if(channel->avail_index_start != NULL)
            {
                assert(channel->avail_index_end != NULL);
                channel->avail_index_start->prev = NULL;
            }
            else
            {
                channel->avail_index_end = NULL;
            }
            assert(node->prev == NULL);
            node->next = NULL;
            channel->head[ret].used = 2;
        }
    }

    result = pthread_mutex_unlock(&(channel->mutex));
    if(result != 0)
    {
        printf("check_acport(): pthread_mutex_unlock() failed!\n");
        exit(EXIT_FAILURE);
    }

    return ret;
}

// Added by jhw at 10.01.07 for library
int read_libport(int channel_id, int func_ret, unsigned char *buf, int len, int index)
{
    int result;
    AC_DATA *ptr;
    LIB_CHANNEL *lib_channel = NULL;
    LIB_CHANNEL_UNIT *channel = NULL;

    lib_channel = get_lib_channel(channel_id);

    if(func_ret == 0)		channel = &(lib_channel->func_channel);
    else if(func_ret == 1)	channel = &(lib_channel->ret_channel);

    ptr = &channel->head[index];


#if defined(PARAM_DEBUG) && (PARAM_DEBUG==1)
    if(len != channel->sampleSize)
    {
        printf("read_libport(): len should be %d!\n", channel->sampleSize);
        exit(EXIT_FAILURE);
    }

    if(index < 0)
    {
        printf("read_libport(): index should be non-negative!\n");
        exit(EXIT_FAILURE);
    }

    if(index >= channel->max_size/channel->sampleSize)
    {
        printf("read_libport(): index is too large!\n");
        exit(EXIT_FAILURE);
    }
#endif

read_lib_start:

    result = pthread_mutex_lock(&(channel->mutex));
    if(result != 0)
    {
        printf("read_libport(): pthread_mutex_lock() failed!\n");
        exit(EXIT_FAILURE);
    }

    if(ptr->used == 0) // blocking case
    {
#if 0
        channel->request_read = true;
        if(channel->request_write == true)
        {
            printf("read_libport(): deadlock!\n");
            exit(EXIT_FAILURE);
        }
#endif
        result = pthread_cond_wait(&(channel->cond), &(channel->mutex));
        if(result != 0)
        {
            printf("read_libport(): pthread_cond_wait() failed!\n");
            exit(EXIT_FAILURE);
        }
        result = pthread_mutex_unlock(&(channel->mutex));
        if(result != 0)
        {
            printf("read_libport(): pthread_mutex_unlock() failed!\n");
            exit(EXIT_FAILURE);
        }
        goto read_lib_start;
    }

#if 0
    channel->request_read = false;
#endif
    memcpy(buf, ptr->buf, len);

    // for check_acport()
    {
        AC_AVAIL_LIST *node;
        node = ptr->avail_node;

        if(ptr->used == 2) // already checked case
        {
            ; // do nothing
            assert(node->prev == NULL);
            assert(node->next == NULL);
        }
        else
        {
            if(node->prev != NULL)
            {
                assert(node != channel->avail_index_start);
                node->prev->next = node->next;
            }
            else
            {
                assert(node == channel->avail_index_start);
                channel->avail_index_start = node->next;
            }
            if(node->next != NULL)
            {
                assert(node != channel->avail_index_end);
                node->next->prev = node->prev;
            }
            else
            {
                assert(node == channel->avail_index_end);
                channel->avail_index_end = node->prev;
            }
            node->prev = node->next = NULL;
        }
    }
    ptr->used = 0;

    result = pthread_mutex_unlock(&(channel->mutex));
    if(result != 0)
    {
        printf("read_libport(): pthread_mutex_unlock() failed!\n");
        exit(EXIT_FAILURE);
    }
    result = pthread_cond_broadcast(&(channel->cond));
    if(result != 0)
    {
        printf("read_libport(): pthread_cond_broadcast() failed!\n");
        exit(EXIT_FAILURE);
    }

/*
    lib_channel->used = 0;
    result = pthread_mutex_unlock(&(lib_channel->l_mutex));
    result = pthread_cond_broadcast(&(lib_channel->l_cond));
*/

#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
    {
        int i;
        for(i=0; i<ARRAYLEN(addressmap); i++)
        {
            if(addressmap[i].task_id != get_mytask_id() &&
                    addressmap[i].channel_id == channel_id &&
                    addressmap[i].op == 'w')
            {
                i = addressmap[i].task_id;
                break;
            }
        }
        printf("PROC_DEBUG(CH) %s acread %d(from'%s') %d %d end\n", get_task_name(get_mytask_id()), channel_id, get_task_name(i), len, index);
    }
#endif

    return len;
}

int write_libport(int channel_id, int func_ret, unsigned char *buf, int len, int index)
{
    int result;
    AC_DATA *ptr;

    LIB_CHANNEL *lib_channel = NULL;
    LIB_CHANNEL_UNIT *channel = NULL;

    lib_channel = get_lib_channel(channel_id);

    if(func_ret == 0)		channel = &(lib_channel->func_channel);
    else if(func_ret == 1)	channel = &(lib_channel->ret_channel);

    ptr = &channel->head[index];

#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
    {
        int i;
        check_valid_operation(channel_id, CHANNEL_TYPE_ARRAY_CHANNEL, 'w');
        for(i=0; i<ARRAYLEN(addressmap); i++)
        {
            if(addressmap[i].task_id != get_mytask_id() &&
                    addressmap[i].channel_id == channel_id &&
                    addressmap[i].op == 'r')
            {
                i = addressmap[i].task_id;
                break;
            }
        }
        printf("PROC_DEBUG(CH) %s acwrite %d(to'%s') %d %d start\n", get_task_name(get_mytask_id()), channel_id, get_task_name(i), len, index);
    }
#endif

#if defined(PARAM_DEBUG) && (PARAM_DEBUG==1)
    if(len != channel->sampleSize)
    {
        printf("write_libport(): len should be %d!\n", channel->sampleSize);
        exit(EXIT_FAILURE);
    }

    if(index < 0)
    {
        printf("write_libport(): index should be non-negative!\n");
        exit(EXIT_FAILURE);
    }

    if(index >= channel->max_size/channel->sampleSize)
    {
        printf("write_libport(): index is too large!\n");
        exit(EXIT_FAILURE);
    }
#endif
/*
lib_wait:
	if(lib_channel->used == 1)
	{
		result = pthread_cond_wait(&(lib_channel->l_cond), &(lib_channel->l_mutex));
		goto lib_wait;
	}
	result = pthread_mutex_lock(&(lib_channel->l_mutex));
	lib_channel->used == 1;
*/

write_lib_start:
    result = pthread_mutex_lock(&(channel->mutex));
    if(result != 0)
    {
        printf("write_libport(): pthread_mutex_lock() failed!\n");
        exit(EXIT_FAILURE);
    }

    if(ptr->used != 0) // blocking case
    {
#if 0
        channel->request_write = true;
        if(channel->request_read == true)
        {
            printf("write_acport(): deadlock!\n");
            exit(EXIT_FAILURE);
        }
#endif
        result = pthread_cond_wait(&(channel->cond), &(channel->mutex));
        if(result != 0)
        {
            printf("read_libport(): pthread_cond_wait() failed!\n");
            exit(EXIT_FAILURE);
        }
        result = pthread_mutex_unlock(&(channel->mutex));
        if(result != 0)
        {
            printf("read_libport(): pthread_mutex_unlock() failed!\n");
            exit(EXIT_FAILURE);
        }
        goto write_lib_start;
    }

#if 0
    channel->request_write = false;
#endif
    memcpy(ptr->buf, buf, len);
    ptr->used = 1;
    // for check_acport()
    {
        if(channel->avail_index_start == NULL)
        {
            assert(channel->avail_index_end == NULL);
            channel->avail_index_start = channel->avail_index_end = ptr->avail_node;
            assert(channel->avail_index_start->prev == NULL);
            assert(channel->avail_index_start->next == NULL);
        }
        else
        {
            AC_AVAIL_LIST *node;
            assert(channel->avail_index_end != NULL);
            node = channel->avail_index_end;
            channel->avail_index_end = node->next = ptr->avail_node;
            ptr->avail_node->prev = node;
            ptr->avail_node->next = NULL;
        }
    }

    result = pthread_mutex_unlock(&(channel->mutex));
    if(result != 0)
    {
        printf("write_libport(): pthread_mutex_unlock() failed!\n");
        exit(EXIT_FAILURE);
    }
    result = pthread_cond_broadcast(&(channel->cond));
    if(result != 0)
    {
        printf("write_libport(): pthread_cond_broadcast() failed!\n");
        exit(EXIT_FAILURE);
    }

#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
    {
        int i;
        for(i=0; i<ARRAYLEN(addressmap); i++)
        {
            if(addressmap[i].task_id != get_mytask_id() &&
                    addressmap[i].channel_id == channel_id &&
                    addressmap[i].op == 'r')
            {
                i = addressmap[i].task_id;
                break;
            }
        }
        printf("PROC_DEBUG(CH) %s libwrite %d(to'%s') %d %d end\n", get_task_name(get_mytask_id()), channel_id, get_task_name(i), len, index);
    }
#endif

    return len;
}

int lib_available (int channel_id, int func_ret, int index)
{
    LIB_CHANNEL *lib_channel;
    LIB_CHANNEL_UNIT *channel;

    lib_channel = get_lib_channel(channel_id);

    if(func_ret == 0)		channel = &lib_channel->func_channel;
    else if(func_ret == 1)	channel = &lib_channel->ret_channel;
    else
    {
        printf("ac_available(): unknown func_ret status!\n");
        exit(EXIT_FAILURE);
    }

    if(channel->head[index].used != 0)
    {
        return 1;
    }
    else if(channel->head[index].used == 0)
    {
        return 0;
    }
    else
    {
        printf("ac_available(): unknown status!\n");
        exit(EXIT_FAILURE);
    }
}

int check_libport(int channel_id, int func_ret)
{
    int ret = -1;
    int result;

    LIB_CHANNEL *lib_channel;
    LIB_CHANNEL_UNIT *channel;

    lib_channel = get_lib_channel(channel_id);

    if(func_ret == 0)		channel = &lib_channel->func_channel;
    else if(func_ret == 1)	channel = &lib_channel->ret_channel;
    else
    {
        printf("ac_available(): unknown func_ret status!\n");
        exit(EXIT_FAILURE);
    }

check_lib_start:

    result = pthread_mutex_lock(&(channel->mutex));
    if(result != 0)
    {
        printf("check_libport(): pthread_mutex_lock() failed!\n");
        exit(EXIT_FAILURE);
    }

    {
        AC_AVAIL_LIST *node;
        node = channel->avail_index_start;

        if(node == NULL) // blocking case
        {
            int result;
            result = pthread_cond_wait(&(channel->cond), &(channel->mutex));
            if(result != 0)
            {
                printf("check_libport(): pthread_cond_wait() failed!\n");
                exit(EXIT_FAILURE);
            }
            result = pthread_mutex_unlock(&(channel->mutex));
            if(result != 0)
            {
                printf("check_libport(): pthread_mutex_unlock() failed!\n");
                exit(EXIT_FAILURE);
            }
            goto check_lib_start;
        }
        else
        {
            ret = node->avail_index;
            channel->avail_index_start = channel->avail_index_start->next;

            if(channel->avail_index_start != NULL)
            {
                assert(channel->avail_index_end != NULL);
                channel->avail_index_start->prev = NULL;
            }
            else
            {
                channel->avail_index_end = NULL;
            }
            assert(node->prev == NULL);
            node->next = NULL;
            channel->head[ret].used = 2;
        }
    }

    result = pthread_mutex_unlock(&(channel->mutex));
    if(result != 0)
    {
        printf("check_libport(): pthread_mutex_unlock() failed!\n");
        exit(EXIT_FAILURE);
    }

    return ret;
}

int lock_lib_channel(int channel_id)
{
	int result;
	LIB_CHANNEL *lib_channel = get_lib_channel(channel_id);

	lib_wait:
	result = pthread_mutex_lock(&(lib_channel->l_mutex));
	if(lib_channel->used == 1)
	{
		result = pthread_cond_wait(&(lib_channel->l_cond), &(lib_channel->l_mutex));
	    result = pthread_mutex_unlock(&(lib_channel->l_mutex));
		goto lib_wait;
	}
	lib_channel->used = 1;

	return 0;
}

int unlock_lib_channel(int channel_id)
{
	int result;

	LIB_CHANNEL *lib_channel = get_lib_channel(channel_id);

    	lib_channel->used = 0;
    	result = pthread_mutex_unlock(&(lib_channel->l_mutex));
    	result = pthread_cond_broadcast(&(lib_channel->l_cond));

 	return 0;
}

// Added by jhw at 10.06.04 for control

static CON_CHANNEL *get_con_channel(int channel_id)
{
    int i;
    for(i=0;i<ARRAYLEN(con_channels);i++)
    {
        if(con_channels[i].channel_id == channel_id)
        {
            return &(con_channels[i]);
        }
    }
    printf("get_channel(): Can't find channel(channel_id: %d)\n", channel_id);
    exit(EXIT_FAILURE);
}


static void init_con_channel(void)
{
    int i,k;
    CON_CHANNEL_UNIT *con_channel_temp = NULL;

    for(i=0; i<ARRAYLEN(con_channels); i++)
    {
		if( 0 != pthread_mutex_init(&(con_channels[i].c_mutex), NULL))
		{
			printf("init_con_channel(): pthread_mutex_init() failed!\n");
			exit(EXIT_FAILURE);
		}

		if( 0 != pthread_cond_init(&(con_channels[i].c_cond), NULL))
		{
			printf("init_con_channel(): pthread_cond_init() failed!\n");
			exit(EXIT_FAILURE);
		}

		for(k=0; k<2; k++)
		{
			if(k == 0)	con_channel_temp = &con_channels[i].send_channel; 
			else if(k == 1)	con_channel_temp = &con_channels[i].recv_channel; 

			if( 0 != pthread_mutex_init(&(con_channel_temp->mutex), NULL))
			{
				printf("init_con_channel(): pthread_mutex_init() failed!\n");
				exit(EXIT_FAILURE);
			}

			if( 0 != pthread_cond_init(&(con_channel_temp->cond), NULL))
			{
				printf("init_con_channel(): pthread_cond_init() failed!\n");
				exit(EXIT_FAILURE);
			}


			if(con_channel_temp->type == CHANNEL_TYPE_ARRAY_CHANNEL)
			{
				if(con_channel_temp->max_size <= 0)
				{
					printf("init_con_channel(): size should be positive!\n");
					exit(EXIT_FAILURE);
				}

				if(con_channel_temp->sampleSize <= 0)
				{
					printf("init_con_channel(): sampleSize should be positive!\n");
					exit(EXIT_FAILURE);
				}

				if(con_channel_temp->max_size % con_channel_temp->sampleSize != 0)
				{
					printf("init_con_channel(): size should be divided by sampleSize!\n");
					exit(EXIT_FAILURE);
				}

				if(con_channel_temp->max_size / con_channel_temp->sampleSize < con_channel_temp->initData)
				{
					printf("init_con_channel(): initData is too large!\n");
					exit(EXIT_FAILURE);
				}

				{
					int j;
					const int count = con_channel_temp->max_size / con_channel_temp->sampleSize;
					con_channel_temp->head = malloc(count * sizeof(AC_DATA));

					if(con_channel_temp->head == NULL)
					{
						printf("init_con_channel(): malloc() failed!\n");
						exit(EXIT_FAILURE);
					}

					con_channel_temp->avail_index_start = con_channel_temp->avail_index_end = NULL;

					for(j=0;j<count;j++)
					{
						con_channel_temp->head[j].avail_node = malloc(sizeof(AC_AVAIL_LIST));
						if(con_channel_temp->head[j].avail_node == NULL)
						{
							printf("init_con_channel(): malloc() failed!\n");
							exit(EXIT_FAILURE);
						}
						con_channel_temp->head[j].avail_node->avail_index = j;
						con_channel_temp->head[j].avail_node->prev = con_channel_temp->head[j].avail_node->next = NULL;

						if(j < con_channel_temp->initData)
						{
							con_channel_temp->head[j].used = 1;
							{
								AC_DATA *ptr;
								CON_CHANNEL_UNIT *channel;
								ptr = &(con_channel_temp->head[j]);
								channel = (con_channel_temp);
								if(channel->avail_index_start == NULL)
								{
									assert(channel->avail_index_end == NULL);
									channel->avail_index_start = channel->avail_index_end = ptr->avail_node;
									assert(channel->avail_index_start->prev == NULL);
									assert(channel->avail_index_start->next == NULL);
								}
								else
								{
									AC_AVAIL_LIST *node;
									assert(channel->avail_index_end != NULL);
									node = channel->avail_index_end;
									channel->avail_index_end = node->next = ptr->avail_node;
									ptr->avail_node->prev = node;
									ptr->avail_node->next = NULL;
								}
							}
						}
						else
						{
							con_channel_temp->head[j].used = 0;
						}
						con_channel_temp->head[j].buf = malloc(con_channel_temp->sampleSize);
						if(con_channel_temp->head[j].buf == NULL)
						{
							printf("init_con_channel(): malloc() failed!\n");
							exit(EXIT_FAILURE);
						}
					}
				}
			}
		}
	}
}

// Added by jhw at 10.01.07 for library
static void wrapup_con_channel(void)
{
	int i, k;
	CON_CHANNEL_UNIT *con_channel_temp = NULL;

	for(i=0; i<ARRAYLEN(con_channels); i++)
	{
		for(k=0; k<2; k++)
		{
			if(k==0)		con_channel_temp = &con_channels[i].send_channel;
			else if(k==1)	con_channel_temp = &con_channels[i].recv_channel;

			if(con_channel_temp->type == CHANNEL_TYPE_ARRAY_CHANNEL)
			{	
				int j;
				const int count = (con_channel_temp->max_size) / (con_channel_temp->sampleSize);
				for(j=0;j<count;j++)
				{
					free(con_channel_temp->head[j].avail_node);
					free(con_channel_temp->head[j].buf);
				}
				free(con_channel_temp->head);
			}
		}
	}
}


int read_con_acport(int channel_id, int send_recv, unsigned char *buf, int len, int index)
{
	int result;
    AC_DATA *ptr;
    CON_CHANNEL *con_channel = NULL;
    CON_CHANNEL_UNIT *channel = NULL;

    con_channel = get_con_channel(channel_id);
    if(send_recv == 1)		channel = &(con_channel->send_channel);
    else if(send_recv == 0)	channel = &(con_channel->recv_channel);

    ptr = &channel->head[index];

#if defined(PARAM_DEBUG) && (PARAM_DEBUG==1)
    if(len != channel->sampleSize)
    {
        printf("read_libport(): len should be %d!\n", channel->sampleSize);
        exit(EXIT_FAILURE);
    }

    if(index < 0)
    {
        printf("read_libport(): index should be non-negative!\n");
        exit(EXIT_FAILURE);
    }

    if(index >= channel->max_size/channel->sampleSize)
    {
        printf("read_libport(): index is too large!\n");
        exit(EXIT_FAILURE);
    }
#endif

read_con_start:

    result = pthread_mutex_lock(&(channel->mutex));
    if(result != 0)
    {
        printf("read_libport(): pthread_mutex_lock() failed!\n");
        exit(EXIT_FAILURE);
    }

    if(ptr->used == 0) // blocking case
    {
#if 0
        channel->request_read = true;
        if(channel->request_write == true)
        {
            printf("read_libport(): deadlock!\n");
            exit(EXIT_FAILURE);
        }
#endif
        result = pthread_cond_wait(&(channel->cond), &(channel->mutex));
        if(result != 0)
        {
            printf("read_libport(): pthread_cond_wait() failed!\n");
            exit(EXIT_FAILURE);
        }
        result = pthread_mutex_unlock(&(channel->mutex));
        if(result != 0)
        {
            printf("read_libport(): pthread_mutex_unlock() failed!\n");
            exit(EXIT_FAILURE);
        }
        goto read_con_start;
    }

#if 0
    channel->request_read = false;
#endif
    memcpy(buf, ptr->buf, len);

    // for check_acport()
    {
        AC_AVAIL_LIST *node;
        node = ptr->avail_node;

        if(ptr->used == 2) // already checked case
        {
            ; // do nothing
            assert(node->prev == NULL);
            assert(node->next == NULL);
        }
        else
        {
            if(node->prev != NULL)
            {
                assert(node != channel->avail_index_start);
                node->prev->next = node->next;
            }
            else
            {
                assert(node == channel->avail_index_start);
                channel->avail_index_start = node->next;
            }
            if(node->next != NULL)
            {
                assert(node != channel->avail_index_end);
                node->next->prev = node->prev;
            }
            else
            {
                assert(node == channel->avail_index_end);
                channel->avail_index_end = node->prev;
            }
            node->prev = node->next = NULL;
        }
    }
    ptr->used = 0;
	channel->used = 0;

    result = pthread_mutex_unlock(&(channel->mutex));
    if(result != 0)
    {
        printf("read_libport(): pthread_mutex_unlock() failed!\n");
        exit(EXIT_FAILURE);
    }
    result = pthread_cond_broadcast(&(channel->cond));
    if(result != 0)
    {
        printf("read_libport(): pthread_cond_broadcast() failed!\n");
        exit(EXIT_FAILURE);
    }

/*
    lib_channel->used = 0;
    result = pthread_mutex_unlock(&(lib_channel->l_mutex));
    result = pthread_cond_broadcast(&(lib_channel->l_cond));
*/

#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
    {
        int i;
        for(i=0; i<ARRAYLEN(addressmap); i++)
        {
            if(addressmap[i].task_id != get_mytask_id() &&
                    addressmap[i].channel_id == channel_id &&
                    addressmap[i].op == 'w')
            {
                i = addressmap[i].task_id;
                break;
            }
        }
        printf("PROC_DEBUG(CH) %s acread %d(from'%s') %d %d end\n", get_task_name(get_mytask_id()), channel_id, get_task_name(i), len, index);
    }
#endif

    return len;
}

int write_con_acport(int channel_id, int send_recv, unsigned char *buf, int len, int index)
{
    int result;
    AC_DATA *ptr;

    CON_CHANNEL *con_channel = NULL;
    CON_CHANNEL_UNIT *channel = NULL;

    con_channel = get_con_channel(channel_id);

    if(send_recv == 1)			channel = &(con_channel->send_channel);
    else if(send_recv == 0)		channel = &(con_channel->recv_channel);

    ptr = &channel->head[index];

write_con_start:
    result = pthread_mutex_lock(&(channel->mutex));
    if(result != 0)
    {
        printf("write_con_acport(): pthread_mutex_lock() failed!\n");
        exit(EXIT_FAILURE);
    }

    if(ptr->used != 0) // blocking case
    {
        result = pthread_cond_wait(&(channel->cond), &(channel->mutex));
        if(result != 0)
        {
            printf("write_con_acport(): pthread_cond_wait() failed!\n");
            exit(EXIT_FAILURE);
        }
        result = pthread_mutex_unlock(&(channel->mutex));
        if(result != 0)
        {
            printf("write_con_acport(): pthread_mutex_unlock() failed!\n");
            exit(EXIT_FAILURE);
        }
        goto write_con_start;
    }
    memcpy(ptr->buf, buf, len);
    ptr->used = 1;
	channel->used = 1;
    {
        if(channel->avail_index_start == NULL)
        {
            assert(channel->avail_index_end == NULL);
            channel->avail_index_start = channel->avail_index_end = ptr->avail_node;
            assert(channel->avail_index_start->prev == NULL);
            assert(channel->avail_index_start->next == NULL);
        }
        else
        {
            AC_AVAIL_LIST *node;
            assert(channel->avail_index_end != NULL);
            node = channel->avail_index_end;
            channel->avail_index_end = node->next = ptr->avail_node;
            ptr->avail_node->prev = node;
            ptr->avail_node->next = NULL;
        }
    }

    result = pthread_mutex_unlock(&(channel->mutex));
    if(result != 0)
    {
        printf("write_con_acport(): pthread_mutex_unlock() failed!\n");
        exit(EXIT_FAILURE);
    }
    result = pthread_cond_broadcast(&(channel->cond));
    if(result != 0)
    {
        printf("write_con_acport(): pthread_cond_broadcast() failed!\n");
        exit(EXIT_FAILURE);
    }
#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
    {
        int i;
        for(i=0; i<ARRAYLEN(addressmap); i++)
        {
            if(addressmap[i].task_id != get_mytask_id() &&
                    addressmap[i].channel_id == channel_id &&
                    addressmap[i].op == 'r')
            {
                i = addressmap[i].task_id;
                break;
            }
        }
        printf("PROC_DEBUG(CH) %s conwrite %d(to'%s') %d %d end\n", get_task_name(get_mytask_id()), channel_id, get_task_name(i), len, index);
    }
#endif

    return len;
}

int check_con_acport(int channel_id, int send_recv)
{
    int ret = -1;
    int result;

    CON_CHANNEL *con_channel;
    CON_CHANNEL_UNIT *channel;
    con_channel = get_con_channel(channel_id);

    if(send_recv == 1)		channel = &con_channel->send_channel;
    else if(send_recv == 0)	channel = &con_channel->recv_channel;
    else
    {
        printf("ac_available(): unknown func_ret status!\n");
        exit(EXIT_FAILURE);
    }

check_con_start:

    result = pthread_mutex_lock(&(channel->mutex));
    if(result != 0)
    {
        printf("check_con_acport(): pthread_mutex_lock() failed!\n");
        exit(EXIT_FAILURE);
    }

    {
        AC_AVAIL_LIST *node;
        node = channel->avail_index_start;

        if(node == NULL) // blocking case
        {
            int result;
            result = pthread_cond_wait(&(channel->cond), &(channel->mutex));
            if(result != 0)
            {
                printf("check_con_acport(): pthread_cond_wait() failed!\n");
                exit(EXIT_FAILURE);
            }
            result = pthread_mutex_unlock(&(channel->mutex));
            if(result != 0)
            {
                printf("check_con_acport(): pthread_mutex_unlock() failed!\n");
                exit(EXIT_FAILURE);
            }
            goto check_con_start;
        }
        else
        {
            ret = node->avail_index;
            channel->avail_index_start = channel->avail_index_start->next;

            if(channel->avail_index_start != NULL)
            {
                assert(channel->avail_index_end != NULL);
                channel->avail_index_start->prev = NULL;
            }
            else
            {
                channel->avail_index_end = NULL;
            }
            assert(node->prev == NULL);
            node->next = NULL;
            channel->head[ret].used = 2;
        }
    }

    result = pthread_mutex_unlock(&(channel->mutex));
    if(result != 0)
    {
        printf("check_con_acport(): pthread_mutex_unlock() failed!\n");
        exit(EXIT_FAILURE);
    }

    return ret;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

typedef struct {
	CONTROL_SEND_PACKET packet;                                                                                 
	int input_order;
} CONTROL_JOB_Q_ITEM;                                                                                           

typedef struct {
	CONTROL_JOB_Q_ITEM item[CONTROL_JOB_Q_SIZE];                                                                
	int cur_size;                                                                                               
	int max_size;
	pthread_mutex_t q_mutex;
	pthread_cond_t q_cond;
} CONTROL_JOB_Q_INFO;                                                                                           

static CONTROL_JOB_Q_INFO control_job_queue;                                                                    

typedef struct {
	int priority;                                                                                               
	int age;
} CURENT_CONTROL_PRIORITY_INFO;                                                                                 

static CURENT_CONTROL_PRIORITY_INFO ccp[CONTROL_GROUP_COUNT];                                                   


static void control_job_queue_init()                                                                            
{   
	int i;
	control_job_queue.cur_size = 0;
	control_job_queue.max_size = CONTROL_JOB_Q_SIZE-1;                                                          
	for (i=0;i<CONTROL_JOB_Q_SIZE;i++) {
		(control_job_queue.item[i]).packet.valid = 0;                                                           
		(control_job_queue.item[i]).input_order = -1;                                                           
	}                   
    pthread_mutex_init(&(control_job_queue.q_mutex), NULL);   
    pthread_cond_init(&(control_job_queue.q_cond), NULL);   
}                      

static int get_my_task_id(void)
{
	int i;
	for(i=0; i<ARRAYLEN(tasks); i++)
	{
		if(tasks[i].thread == pthread_self())
		{
			return tasks[i].task_id;
		}
	}
	printf("PROC_DEBUG(ERR) get_my_task_id(): Can't find task's id\n");
	exit(EXIT_FAILURE);
}

unsigned int get_current_time_base()
{
	unsigned int time;
	struct timeval now_time;

	/* set time base */
	gettimeofday(&now_time, 0);
	time = (unsigned int)(((now_time.tv_sec % 100)*pow(10, 6)) + now_time.tv_usec);
	return time;
}

static unsigned int get_control_channel_index(int control_task_id)
{
	int i;
	/* get control channel array index */
	for(i=0;i<CONTROL_CHANNEL_COUNT;i++) {
		if(control_task_id == control_channel[i].control_task_id) {
			return i;
		}
	}
	printf("NOT CALLED CONTROL CHANNEL\n");
	return -1;
}

static int get_valid_timer_slot_id(unsigned int cont_ch_index)
{
	int i, temp_slot_index;

	i = control_channel[cont_ch_index].empty_slot_index;
	temp_slot_index = 0;

	if(control_channel[cont_ch_index].timer_slot[temp_slot_index] != 0) {  // check the next point is available
		int cur_time_val = get_current_time_base();
		if(control_channel[cont_ch_index].timer_slot[temp_slot_index]  <= cur_time_val) {
			control_channel[cont_ch_index].timer_slot[temp_slot_index] = 0;
		}
		else { // traverse until it find an available time slot
			for(; temp_slot_index != i; ++temp_slot_index) {
				if(temp_slot_index>=MAX_TIMER_SLOT_SIZE)
					temp_slot_index = 0;
				if(control_channel[cont_ch_index].timer_slot[temp_slot_index] == 0)
					break;
				else if(control_channel[cont_ch_index].timer_slot[temp_slot_index] <= cur_time_val) {
					control_channel[cont_ch_index].timer_slot[temp_slot_index] = 0;
					break;
				}
			}

			if(temp_slot_index == i)
			{
				return -1;
			}
		}
	}
	return temp_slot_index;
}

static int get_valid_time_base_id(unsigned int cont_ch_index)
{
	int i, temp_base_index;

	i = control_channel[cont_ch_index].empty_base_index;
	temp_base_index = control_channel[cont_ch_index].empty_base_index + 1;

	if(temp_base_index>MAX_TIME_BASE_COUNT) // return back to the first time_slot
		temp_base_index = 0;

	control_channel[cont_ch_index].empty_base_index = temp_base_index;

	if(control_channel[cont_ch_index].time_base[temp_base_index]!=0)
		printf( "TIME BASE SLOT FULL, replace the old one. \\n");
	return i;

}


/* heap for dynamic priority queue */
static void up_heap()
{
	int i = control_job_queue.cur_size;
	CONTROL_JOB_Q_ITEM cur_node = control_job_queue.item[i];

	int j = i>>1; // parent
	j = j & ~(1 << 31);

	while(( cur_node.packet.my_go_time < (control_job_queue.item[j]).packet.my_go_time ) ||
			( (cur_node.packet.my_go_time == (control_job_queue.item[j]).packet.my_go_time ) && (cur_node.input_order < control_job_queue.item[j].input_order) )) { // && (cur_no
		if(j <= 0) break;

		control_job_queue.item[i] = control_job_queue.item[j]; // change child <-> parent
		i = j;
		j = j >> 1;
		j = j & ~(1 << 31);
	}
	control_job_queue.item[i] = cur_node;
}

static void down_heap() {
	int i = 1;
	CONTROL_JOB_Q_ITEM cur_node = control_job_queue.item[i];
	//  CONTROL_SEND_PACKET cur_node = (control_job_queue.item[i]);
	int j = i<< 1;// first child
	int k = j+1;// second child
	if( k <= control_job_queue.cur_size) { //&& ( (control_job_queue.item[k]).packet.my_go_time <= (control_job_queue.item[j]).packet.my_go_time ) ) 
		if( control_job_queue.item[k].packet.my_go_time < control_job_queue.item[j].packet.my_go_time ) {
			j = k;
		}
		else if( control_job_queue.item[k].packet.my_go_time == control_job_queue.item[j].packet.my_go_time ) {
			if( control_job_queue.item[k].input_order < control_job_queue.item[j].input_order) {
				j = k;
			}
		}
	}

	while( ( (j<=control_job_queue.cur_size) && ( (control_job_queue.item[j]).packet.my_go_time < cur_node.packet.my_go_time)  )
			|| ( (j<=control_job_queue.cur_size) && ( (control_job_queue.item[j]).packet.my_go_time == cur_node.packet.my_go_time) && (control_job_queue.item[j].input_order < cur_node.input_order ) ) ) {
		control_job_queue.item[i] = control_job_queue.item[j];
		i = j;
		j = j<<1;
		k = j+1;
		//if( k <= control_job_queue.cur_size && ( (control_job_queue.item[k]).packet.my_go_time <= (control_job_queue.item[j]).packet.my_go_time ) ) 
		//  j = k;
		//
		if( k <= control_job_queue.cur_size) { //&& ( (control_job_queue.item[k]).packet.my_go_time <= (control_job_queue.item[j]).packet.my_go_time ) ) 
			if( control_job_queue.item[k].packet.my_go_time < control_job_queue.item[j].packet.my_go_time ) {
				j = k;
			}
			else if( control_job_queue.item[k].packet.my_go_time == control_job_queue.item[j].packet.my_go_time ) {
				if( control_job_queue.item[k].input_order < control_job_queue.item[j].input_order) {
					j = k;
				}
			}
		}
	}
	control_job_queue.item[i] = cur_node;
}

static long queue_increase;
static int push_pqueue(CONTROL_SEND_PACKET item)
{
	if(control_job_queue.cur_size < control_job_queue.max_size)
	{
		queue_increase++;
		control_job_queue.cur_size++;
		control_job_queue.item[control_job_queue.cur_size].packet = item;
		control_job_queue.item[control_job_queue.cur_size].input_order = queue_increase;
		up_heap();
		return 1;
	}
	else
		return -1;
}

static CONTROL_SEND_PACKET* top_pqueue(void)
{
	if(control_job_queue.cur_size > 0) {
		return &(control_job_queue.item[1].packet);
	}
	else {
		return NULL;
	}
}

static CONTROL_SEND_PACKET pop_pqueue(void)
{
	CONTROL_SEND_PACKET result;

	if(control_job_queue.cur_size > 0)
	{
		result = control_job_queue.item[1].packet;
		control_job_queue.item[1] = control_job_queue.item[control_job_queue.cur_size];
		(control_job_queue.item[control_job_queue.cur_size]).packet.valid = 0;
		(control_job_queue.item[control_job_queue.cur_size]).input_order = -1;
		control_job_queue.cur_size--;
		down_heap();
		return result;
	}
	else
	{
		result.valid = 0;
		return result;
	}
}

static void print_packet(CONTROL_SEND_PACKET s)
{
	printf("v : %d, tt_id: %d, c_id: %d, c_p: %d, c_g: %d, cmd_t: %d, p_id: %d, go_ti: %d, p_val: %d \n",
			s.valid,
			s.target_task_id,
			s.my_control_task_id,
			s.my_control_priority,
			s.my_control_group_id,
			s.command_type,
			s.param_id,
			s.my_go_time,
			s.param_value);
}

long get_param_int(char* t_name, char* p_name)
{
	long ret=0;
	int i=0;
	char *task_name;
	char *param_name;

	if(t_name == NULL)
	{
		int task_id = get_my_task_id();
		task_name = tasks[task_id].name;
		param_name = p_name;
	}
	else
	{
		task_name = t_name;
		param_name = p_name;
	}

	for(i=0; i<ARRAYLEN(param_list); i++)
	{
		if(strcmp(task_name, param_list[i].task_name) == 0 && strcmp(param_name, param_list[i].param_name) == 0)
		{
			ret = (long)(param_list[i].param_value);
			break;
		}
	}

	return ret;
}

void set_param_int(char* t_name, char* p_name, long p_value, int time_base_id, unsigned int time_offset)
{
	CONTROL_SEND_PACKET send_packet;

	int index=-1;
	unsigned int target_task_id=0;
	int param_id=-1;
	int my_control_task_id = -1;

	unsigned int i;
	my_control_task_id = get_my_task_id();

	//printf("set param start\n");
	index = get_control_channel_index(my_control_task_id);
	
	/* find target task id */
	for(i=0;i<ARRAYLEN(tasks);i++){
		if(strcmp(t_name, tasks[i].name)==0) {
			target_task_id = i;
			break;
		}
	}
	/* find param id */
	for(i=0;i<ARRAYLEN(param_list);i++) {
		if(strcmp(p_name, param_list[i].param_name)==0 &&
				target_task_id == param_list[i].task_id) {
			param_id = param_list[i].param_id;
			break;
		}
	}
	/* make set_param packet */
	send_packet.valid = 1;
	send_packet.target_task_id = (unsigned char)target_task_id;
	send_packet.my_control_task_id = (unsigned char)my_control_task_id;
	send_packet.my_control_priority = (unsigned char)control_channel[index].control_priority;
	send_packet.my_control_group_id = (unsigned char)control_channel[index].control_group_id;
	send_packet.command_type = (unsigned char)SET_PARAM_INT;
	send_packet.param_id = (unsigned char)param_id;
	send_packet.param_value = (void*)p_value;
	send_packet.my_go_time = time_base_id + time_offset;

	pthread_mutex_lock(&(control_job_queue.q_mutex));
	push_pqueue(send_packet);
	pthread_mutex_unlock(&(control_job_queue.q_mutex));

	return;
}

double get_param_float(char* t_name, char* p_name)
{
	double ret=0;
	int i=0;
	char *task_name;
	char *param_name;

	if(t_name == NULL)
	{
		int task_id = get_my_task_id();
		task_name = tasks[task_id].name;
		param_name = p_name;
	}
	else
	{
		task_name = t_name;
		param_name = p_name;
	}

	for(i=0; i<ARRAYLEN(param_list); i++)
	{
		if(strcmp(task_name, param_list[i].task_name) == 0 && strcmp(param_name, param_list[i].param_name) == 0)
		{
			ret = *(double*)(param_list[i].param_value);
			break;
		}
	}

	return ret;
}

void set_param_float(char* t_name, char* p_name, double p_value, int time_base_id, unsigned int time_offset)
{
	CONTROL_SEND_PACKET send_packet;

	int index=-1;
	unsigned int target_task_id=0;
	int param_id=-1;
	int my_control_task_id = -1;

	unsigned int i;
	my_control_task_id = get_my_task_id();

	//printf("set param start\n");
	index = get_control_channel_index(my_control_task_id);

	/* find target task id */
	for(i=0;i<ARRAYLEN(tasks);i++){
		if(strcmp(t_name, tasks[i].name)==0) {
			target_task_id = i;
			break;
		}
	}
	/* find param id */
	for(i=0;i<ARRAYLEN(param_list);i++) {
		if(strcmp(p_name, param_list[i].param_name)==0 &&
				target_task_id == param_list[i].task_id) {
			param_id = param_list[i].param_id;
			break;
		}
	}
	/* make set_param packet */
	send_packet.valid = 1;
	send_packet.target_task_id = (unsigned char)target_task_id;
	send_packet.my_control_task_id = (unsigned char)my_control_task_id;
	send_packet.my_control_priority = (unsigned char)control_channel[index].control_priority;
	send_packet.my_control_group_id = (unsigned char)control_channel[index].control_group_id;
	send_packet.command_type = (unsigned char)SET_PARAM_FLOAT;
	send_packet.param_id = (unsigned char)param_id;
	*(double*)(send_packet.param_value) = p_value;
	send_packet.my_go_time = time_base_id + time_offset;

	pthread_mutex_lock(&(control_job_queue.q_mutex));
	push_pqueue(send_packet);
	pthread_mutex_unlock(&(control_job_queue.q_mutex));

	return;

}

void run_task(char* t_name, int time_base_id, unsigned int time_offset)
{
	CONTROL_SEND_PACKET send_packet;

	unsigned int i;
	int index=-1;
	int target_task_id=-1;
	int my_control_task_id=-1;


	my_control_task_id = get_my_task_id();
	index = get_control_channel_index(my_control_task_id);
	/* find target task id */
	for(i=0;i<ARRAYLEN(tasks);i++){
		if(strcmp(t_name, tasks[i].name)==0) {
			target_task_id = i;
			break;
		}
	}
	/* make set_param packet */
	send_packet.valid = 1;
	send_packet.target_task_id = (unsigned char)target_task_id;
	send_packet.my_control_task_id = (unsigned char)my_control_task_id;
	send_packet.my_control_priority = (unsigned char)control_channel[index].control_priority;
	send_packet.my_control_group_id = (unsigned char)control_channel[index].control_group_id;
	send_packet.command_type = (unsigned char)RUN_TASK;
	send_packet.param_id = (unsigned char)0;
	send_packet.param_value = (void*) 0;
	send_packet.my_go_time = time_base_id + time_offset;
	//send_packet.my_go_time = control_channel[index].time_base[time_base_id] + time_offset;

	pthread_mutex_lock(&(control_job_queue.q_mutex));
	push_pqueue(send_packet);
	pthread_mutex_unlock(&(control_job_queue.q_mutex));

	return;

}

void control_begin(int time_base)
{
	CONTROL_SEND_PACKET send_packet;
	int time_base_id=0;
	int my_control_task_id = -1;
	int index=-1;

	my_control_task_id = get_my_task_id();

	index = get_control_channel_index(my_control_task_id);
	time_base_id = get_valid_time_base_id(index);

	send_packet.valid = 1;
	send_packet.my_control_task_id = (unsigned char)my_control_task_id;
	send_packet.my_control_priority = (unsigned char)control_channel[index].control_priority;
	send_packet.my_control_group_id = (unsigned char)control_channel[index].control_group_id;
	send_packet.command_type = (unsigned char)BEGIN;
	send_packet.my_go_time = time_base;

	control_channel[index].time_base[time_base_id] = time_base;

	pthread_mutex_lock(&(control_job_queue.q_mutex));
	push_pqueue(send_packet);
	pthread_mutex_unlock(&(control_job_queue.q_mutex));

	return;
}

void control_end(int time_base_id, unsigned int time_offset)
{
	CONTROL_SEND_PACKET send_packet;

	int my_control_task_id = -1;
	int index=-1;
	int t_index=-1;

	my_control_task_id = get_my_task_id();
	index = get_control_channel_index(my_control_task_id);
	t_index = get_valid_time_base_id(index);

	send_packet.my_control_task_id = (unsigned char)my_control_task_id;
	send_packet.my_control_priority = (unsigned char)control_channel[index].control_priority;
	send_packet.my_control_group_id = (unsigned char)control_channel[index].control_group_id;
	send_packet.command_type = (unsigned char)END;
	send_packet.my_go_time = time_base_id + time_offset;

	control_channel[index].time_base[t_index] = 0; // make valid slot 

	if((time_base_id + 1 == control_channel[index].empty_base_index) ||
			(time_base_id==MAX_TIME_BASE_COUNT && control_channel[index].empty_base_index==0))
	{
		control_channel[index].empty_base_index = time_base_id;
	}

	pthread_mutex_lock(&(control_job_queue.q_mutex));
	push_pqueue(send_packet);
	pthread_mutex_unlock(&(control_job_queue.q_mutex));

	return;
}

int get_timer_alarmed(unsigned int timer_id)
{
	unsigned int my_task_id = get_my_task_id();
	unsigned int cont_ch_index = get_control_channel_index(my_task_id);
	int p_group, p;

	p = control_channel[cont_ch_index].control_priority;
	p_group = control_channel[cont_ch_index].control_group_id;

	if(control_channel[cont_ch_index].timer_slot[timer_id]  <= get_current_time_base()) {
		control_channel[cont_ch_index].timer_slot[timer_id]=0; // reset timer
		control_channel[cont_ch_index].empty_slot_index = timer_id;
		return 0;
	}
	else
		return -1;
}

int set_timer(int time_base_id, unsigned int time_offset)
{
	int timer_id;
	unsigned int my_task_id = get_my_task_id();
	unsigned int cont_ch_index = get_control_channel_index(my_task_id);
	timer_id = get_valid_timer_slot_id(cont_ch_index);

	control_channel[cont_ch_index].timer_slot[timer_id] = get_current_time_base() + time_offset;
	return timer_id;
}

void reset_timer(unsigned int timer_id)
{
	unsigned int my_task_id = get_my_task_id();
	unsigned int cont_ch_index = get_control_channel_index(my_task_id);
	control_channel[cont_ch_index].timer_slot[timer_id]=0; // reset timer
}

void program_kill()
{
    printf("Application Terminated! (on PPE)\n");
    exit(-1);
}

void program_stop()
{
    // Not implemented yet.
}


// Phase 1 : Scan con_channel
static void *control_scheduler_push_routine(void* pdata)
{
	int i=0, j=0;
	int old_state = 0;
	CONTROL_SEND_PACKET temp_packet;

	pthread_mutex_init(&(control_scheduler[0].cs_mutex), NULL);
	pthread_cond_init(&(control_scheduler[0].cs_cond), NULL);

	pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, &old_state);

	while(!all_threads_created);

	while(true) 
	{
		for(i=0; i<ARRAYLEN(con_channels); i++)
		{
			if(con_channels[i].send_channel.used == 1)
			{
				int index = check_con_acport(i, 1);
				read_con_acport(i, 1, (unsigned char*)&temp_packet, sizeof(CONTROL_SEND_PACKET), index);

				if(temp_packet.command_type == GET_PARAM_INT)
				{
					for(j=0; j<ARRAYLEN(param_list); j++)
					{
						if(param_list[j].task_id == temp_packet.target_task_id && param_list[j].param_id == temp_packet.param_id)
						{
							temp_packet.param_value = (param_list[j].param_value);
							write_con_acport(i, 0, (unsigned char*)&temp_packet, sizeof(CONTROL_SEND_PACKET), index);
							break;
						}
					}
				}
				else if(temp_packet.command_type == SET_PARAM_INT)
				{
                    pthread_mutex_lock(&(control_job_queue.q_mutex));
                    push_pqueue(temp_packet);
                    write_con_acport(i, 0, (unsigned char*)&temp_packet, sizeof(CONTROL_SEND_PACKET), index);
                    pthread_mutex_unlock(&(control_job_queue.q_mutex));
                }
                else if(temp_packet.command_type == GET_PARAM_FLOAT)
                {
                    for(j=0; j<ARRAYLEN(param_list); j++)
                    {
                        if(param_list[j].task_id == temp_packet.target_task_id && param_list[j].param_id == temp_packet.param_id)
                        {	
                            temp_packet.param_value = (param_list[j].param_value);
                            write_con_acport(i, 0, (unsigned char*)&temp_packet, sizeof(CONTROL_SEND_PACKET), index);
                            break;
                        }
                    }
                }
                else if(temp_packet.command_type == SET_PARAM_FLOAT)
                {
                    pthread_mutex_lock(&(control_job_queue.q_mutex));
                    push_pqueue(temp_packet);
                    write_con_acport(i, 0, (unsigned char*)&temp_packet, sizeof(CONTROL_SEND_PACKET), index);
                    pthread_mutex_unlock(&(control_job_queue.q_mutex));
                }
                else if(temp_packet.command_type == RUN_TASK)
                {
                    pthread_mutex_lock(&(control_job_queue.q_mutex));
                    push_pqueue(temp_packet);
                    write_con_acport(i, 0, (unsigned char*)&temp_packet, sizeof(CONTROL_SEND_PACKET), index);
                    pthread_mutex_unlock(&(control_job_queue.q_mutex));
                }
                else if(temp_packet.command_type == BEGIN)
                {
                    pthread_mutex_lock(&(control_job_queue.q_mutex));
                    push_pqueue(temp_packet);
                    write_con_acport(i, 0, (unsigned char*)&temp_packet, sizeof(CONTROL_SEND_PACKET), index);
                    pthread_mutex_unlock(&(control_job_queue.q_mutex));
                }
                else if(temp_packet.command_type == END)
                {
                    pthread_mutex_lock(&(control_job_queue.q_mutex));
                    push_pqueue(temp_packet);
                    write_con_acport(i, 0, (unsigned char*)&temp_packet, sizeof(CONTROL_SEND_PACKET), index);
                    pthread_mutex_unlock(&(control_job_queue.q_mutex));
                }
                else
                {
                    printf("Not Implement yet!\n");
                    exit(-1);
                }
            }
		}
	}
}


// Phase 2 : Process work in queue	
static void *control_scheduler_pop_routine(void* pdata)
{
	//unsigned int work_time = 0;
	int p_group, p;
	int i=0;
	int old_state=0;
	int param_index=0;
	CONTROL_SEND_PACKET temp_packet;

	pthread_mutex_init(&(control_scheduler[1].cs_mutex), NULL);
	pthread_cond_init(&(control_scheduler[1].cs_cond), NULL);

	pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, &old_state);

	while(!all_threads_created);

spin_here:

	while(true)
	{
		pthread_mutex_lock(&(control_job_queue.q_mutex));
		if(control_job_queue.cur_size <= 0)
		{
		    pthread_mutex_unlock(&(control_job_queue.q_mutex));
			goto spin_here;
		}
		else
		{
			temp_packet = pop_pqueue();
			pthread_mutex_unlock(&(control_job_queue.q_mutex));
			//print_packet(temp_packet);
			//printf("cur_size = %d, cur_time = %d, work_time = %d\n", control_job_queue.cur_size , current_time, work_time);
		}

		unsigned int target_task_id = (unsigned int)temp_packet.target_task_id;
		unsigned int target_param_id = (unsigned int)temp_packet.param_id;
		void* param_value = temp_packet.param_value;

		p = (int)temp_packet.my_control_priority;
		p_group = (int)temp_packet.my_control_group_id;

		if(temp_packet.valid == 1) 
		{
			switch(temp_packet.command_type) 
			{
				case SET_PARAM_INT:
					for(i=0;i<ARRAYLEN(param_list);i++) 
					{
						if(target_task_id == param_list[i].task_id && target_param_id == param_list[i].param_id) 
						{
							param_index = i;
							break;
						}
					}
					(param_list[param_index].param_value) = param_value;
					break;

				case SET_PARAM_FLOAT:
					for(i=0;i<ARRAYLEN(param_list);i++) 
					{
						if(target_task_id == param_list[i].task_id && target_param_id == param_list[i].param_id) 
						{
							param_index = i;
							break;
						}
					}
					(param_list[param_index].param_value) = param_value;

					break;

				case RUN_TASK:
					for(i=0; i<ARRAYLEN(tasks); i++)
					{
						if(temp_packet.target_task_id == tasks[i].task_id)
						{
wait_run:                   
							pthread_mutex_lock(&(tasks[i].p_mutex));                                            
							if(tasks[i].state == Run)                                                           
							{   
								pthread_mutex_unlock(&(tasks[i].p_mutex));                                      
								goto wait_run;                                                                  
							}
							tasks[i].state = Run;
							pthread_cond_broadcast(&(tasks[i].p_cond));
							pthread_cond_wait(&(control_scheduler[1].cs_cond), &(tasks[i].p_mutex));
							pthread_mutex_unlock(&(tasks[i].p_mutex));
							break;
						}
					}   
					break;

				case BEGIN:
					if( p > ccp[p_group].priority )
					{
						ccp[p_group].priority = p; // change current control priority
						ccp[p_group].age = 1;
					}
					else if( p == ccp[p_group].priority) 
					{
						ccp[p_group].age++;
					}
					break;

				case END:
					if( p == ccp[p_group].priority ) 
					{
						ccp[p_group].age--;
						ccp[p_group].priority = -1; // reset Current control priority
					}
					break;

				default:
					break;
			}
		}
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////


void exit_application(void)
{
	exit(EXIT_FAILURE);
}

#include <mars/task.h>

#include "CIC_SPEs.h"

struct mars_context *mars_ctx;

static void spe_init(void)
{
	int ret; 

	ret = mars_context_create(&mars_ctx, 0, 0);
	if(ret)
	{    
		printf("MARS context create failed! (%d)\n", ret);
		exit(EXIT_FAILURE);
	}    
}

static void spe_term(void)
{
	int ret; 

	ret = mars_context_destroy(mars_ctx);
	if(ret)
	{    
		// this is not the bug....
		//        printf("MARS context destroy failed! (%d)\n", ret);
		//        exit(EXIT_FAILURE);
	}    
}


/*
#include <limits.h>
static int count;

static void generate_trace_file()
{
    int i=0, j=0, index=0, ret=0;
    int ret_array[ARRAYLEN(stubs)+ARRAYLEN(tasks)] = {0};
    char temp[255];
    char* lib_name;
    char* task_name;
    FILE* src[ARRAYLEN(stubs)+ARRAYLEN(tasks)];
    FILE* dst;
    save_unit save_array[ARRAYLEN(stubs)+ARRAYLEN(tasks)] = {{0}};
    save_unit save_one = {0, 0, 0, 0, 0, 0, 0};

    dst = fopen("trace_result.txt", "w");

    for(i=0; i < ARRAYLEN(stubs); i++)
    {
        lib_name = temp;
        strcpy(lib_name, "trace_");
        strcat(lib_name, stubs[i].name);
        strcat(lib_name, ".txt");

        src[i] = fopen(lib_name, "r");

        ret_array[i] = fread(&save_array[i], sizeof(save_unit), 1, src[i]);
    }

    for(; i < ARRAYLEN(stubs) + ARRAYLEN(tasks); i++)
    {
        task_name = temp;
        strcpy(task_name, "trace_");
        strcat(task_name, tasks[i-ARRAYLEN(stubs)].name);
        strcat(task_name, ".txt");

        src[i] = fopen(task_name, "r");

        ret_array[i] = fread(&save_array[i], sizeof(save_unit), 1, src[i]);
    }

    while(ret == 0)
    {
        count++;

        save_one.sec_time_record = LONG_MAX;
        save_one.usec_time_record = LONG_MAX;
        for(j=0; j<ARRAYLEN(stubs)+ARRAYLEN(tasks); j++)
        {
            if(ret_array[j] != 0)
            {
                if(save_one.sec_time_record > save_array[j].sec_time_record)
                {
                    save_one = save_array[j];
                    index = j;
                }

                else if(save_one.sec_time_record == save_array[j].sec_time_record)
                {
                    if(save_one.usec_time_record > save_array[j].usec_time_record)
                    {
                        save_one = save_array[j];
                        index = j;
                    }
                }
            }
        }

        fprintf(dst, "count : %.3d, start_end : %.3d, task_library : %.3d, task_id : %.3d, library_id : %.3d, func_id : %.3d, time : %ldsec %ldusec\n", count, save_one.start_end, save_one.task_library, save_one.task_id, save_one.library_id, save_one.func_id, save_one.sec_time_record, save_one.usec_time_record);

        ret_array[index] = fread(&save_array[index], sizeof(save_unit), 1, src[index]);

        ret = 1;
        for(j=0; j<ARRAYLEN(stubs)+ARRAYLEN(tasks); j++)
        {
            if(ret_array[j] == 1)
            {
                ret = 0;
                break;
            }
        }
    }

    for(i=0; i < ARRAYLEN(stubs)+ARRAYLEN(tasks); i++)
    {
        fclose(src[i]);
    }
    fclose(dst);
}
*/


int main(int argc, char *argv[])
{

	printf("spe_init()\n");
	spe_init();

	printf("init_channel()\n");
	init_channel();
	printf("init_lib_channel()\n");
	init_lib_channel();
	printf("init_con_channel()\n");
	init_con_channel();

    if(ARRAYLEN(control_channel) != 0)
    {
	    printf("init_control_job_queue()\n");
	    control_job_queue_init();
    }

	printf("init_task()\n");
	init_task();
	printf("init_lib()\n");
	init_lib();

	printf("go_stub()\n");
	go_stub();
	printf("go_task()\n");
	go_task();

	printf("wrapup_lib()\n");
	wrapup_lib();
	printf("wrapup_task()\n");
	wrapup_task();

	printf("wrapup_channel()\n");
	wrapup_channel();
	printf("wrapup_lib_channel()\n");
	wrapup_lib_channel();
	printf("wrapup_con_channel()\n");
	wrapup_con_channel();

	printf("spe_term()\n");
	spe_term();

//    printf("generate_trace_file()\n\n");
//    generate_trace_file();

	return EXIT_SUCCESS;
}

