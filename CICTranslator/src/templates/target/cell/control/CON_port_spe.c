
#ifndef __SPU__

    #error

#endif

#include <stdbool.h>
//#define NDEBUG
#include <assert.h>

#include "CON_port.h"
#include "CON_wrapper.h"

#include <math.h>
#include <unistd.h>
#include <sys/time.h>
////////////////////////////////////////
#include <string.h>
#include "param_list.h"
#include "con_taskmap.h"
#include "con_portmap.h"
////////////////////////////////////////

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
	/*
    int index[4];
    mars_task_queue_pop(cic_channel_index_queue, &index);

#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
    //printf("check_acport(%d) returns %d\n", channel_id, index[3]);
#endif

    return index[3];
	*/
	return -1;
}



static int init_con_port(char* my_task_name)
{
	int i=0;

	for(i=0; i<ARRAYLEN(c_addressmap); i++)
	{
		if(strcmp(my_task_name, c_addressmap[i].task_name) == 0)
		{
			return c_addressmap[i].channel_id;
		}

	}
    return -1;
}


static int read_con_acport(int channel_id, int send_recv, unsigned char *buf, int len, int index)
{
	int i;
    uint32_t cond=0;

    if(send_recv == 0)		cond = MARS_TASK_QUEUE_HOST_TO_MPU;
    else if(send_recv == 1)	cond = MARS_TASK_QUEUE_MPU_TO_HOST;

    for(i=0;i<ARRAYLEN(con_channel_info_spe);i++)
    {
        if(con_channel_info_spe[i].channel_id == channel_id && con_channel_info_spe[i].port_id == send_recv)
        {
            con_channel_info_entry *entry;
            entry = &con_channel_info_spe[i];
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

static int write_con_acport(int channel_id, int send_recv, unsigned char *buf, int len, int index)
{
    int i;
    uint32_t cond=0;

    if(send_recv == 0)		cond = MARS_TASK_QUEUE_HOST_TO_MPU;
    else if(send_recv == 1)	cond = MARS_TASK_QUEUE_MPU_TO_HOST;

    for(i=0;i<ARRAYLEN(con_channel_info_spe);i++)
    {
        if(con_channel_info_spe[i].channel_id == channel_id && con_channel_info_spe[i].port_id == send_recv)
        {
            con_channel_info_entry *entry;
            entry = &con_channel_info_spe[i];
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

static int check_con_acport(int channel_id, int send_recv)
{
    int index[4];
    mars_task_queue_pop(con_channel_index_queue, &index);

#if defined(PROC_DEBUG) && (PROC_DEBUG==1)
    //printf("check_acport(%d) returns %d\n", channel_id, index[3]);
#endif

    return index[3];
}


static int get_my_task_id(char* name)
{
	int i;
	for(i=0; i<ARRAYLEN(con_taskmap); i++)
	{
		if(strcmp(con_taskmap[i].t_name, name) == 0)
		{
			return con_taskmap[i].t_id;
		}
	}
	printf("PROC_DEBUG(ERR) get_my_task_id(): Can't find task's id\n");
	exit(EXIT_FAILURE);
}

static int get_my_contask_id(char* name)
{
	int i;
	for(i=0; i<ARRAYLEN(con_taskmap); i++)
	{
		if(strcmp(con_taskmap[i].t_name, name) == 0)
		{
			return con_taskmap[i].c_id;
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


static long get_param_int(char* t_name, char* p_name)
{
	CONTROL_SEND_PACKET send_packet;
	CONTROL_SEND_PACKET receive_packet;

	int i=0;
	char *task_name;
	char *param_name;
	
	int my_control_task_id = 0;
	int target_task_id = 0;
	int param_id = 0;

	char temp[255];
	char* my_task_name = temp;
	strcpy(my_task_name, CONTASKNAME);

	if(t_name == NULL)
	{
		my_control_task_id = get_my_contask_id(my_task_name);
		target_task_id = get_my_task_id(my_task_name);

		task_name = con_taskmap[target_task_id].t_name;
		param_name = p_name;
	}

	else
	{
		/* find target task id */
		for(i=0;i<ARRAYLEN(con_taskmap);i++){
			if(strcmp(t_name, con_taskmap[i].t_name)==0) {
				target_task_id = i;
				break;
			}
		}

		my_control_task_id = get_my_contask_id(my_task_name);
		task_name = t_name;
		param_name = p_name;
	}

	/* find param id */
	for(i=0;i<ARRAYLEN(param_list);i++) {
		if(strcmp(p_name, param_list[i].param_name)==0 &&
				target_task_id == param_list[i].task_id) {
			param_id = param_list[i].param_id;
			break;
		}
	}

	send_packet.valid = 1;
	send_packet.target_task_id = (unsigned char)target_task_id;
	send_packet.my_control_task_id = (unsigned char)my_control_task_id;
	send_packet.my_control_priority = (unsigned char)0;
	send_packet.my_control_group_id = (unsigned char)0;
	send_packet.command_type = (unsigned char)GET_PARAM_INT;
	send_packet.param_id = (unsigned char)param_id;
	send_packet.param_value = (void*)0;
	send_packet.my_go_time = get_current_time_base();

	int channel_id = init_con_port(my_task_name);	

	CON_SEND(channel_id, 1, (unsigned char*)&send_packet, sizeof(CONTROL_SEND_PACKET), 0);

	int c_index = CON_AC_CHECK(channel_id, 0);
	CON_RECEIVE(channel_id, 0, (unsigned char*)&receive_packet, sizeof(CONTROL_SEND_PACKET), c_index);

	return (long)receive_packet.param_value;
}

static void set_param_int(char* t_name, char* p_name, long p_value, int time_base_id, unsigned int time_offset)
{
	CONTROL_SEND_PACKET send_packet;
	CONTROL_SEND_PACKET receive_packet;

	unsigned int i;
	int index=-1;
	unsigned int target_task_id=0;
	int param_id=-1;
	int my_control_task_id = -1;
	char *task_name;
	char *param_name;

	index = get_control_channel_index(my_control_task_id);

	char temp[255];
	char* my_task_name = temp;
	strcpy(my_task_name, CONTASKNAME);

	if(t_name == NULL)
	{
		my_control_task_id = get_my_contask_id(my_task_name);
		target_task_id = get_my_task_id(my_task_name);

		task_name = con_taskmap[target_task_id].t_name;
		param_name = p_name;
	}

	else
	{
		/* find target task id */
		for(i=0;i<ARRAYLEN(con_taskmap);i++){
			if(strcmp(t_name, con_taskmap[i].t_name)==0) {
				target_task_id = i;
				break;
			}
		}

		my_control_task_id = get_my_contask_id(my_task_name);
		task_name = t_name;
		param_name = p_name;
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
	send_packet.my_go_time = get_current_time_base();

	int channel_id = init_con_port(my_task_name);	
	CON_SEND(channel_id, 1, (unsigned char*)&send_packet, sizeof(CONTROL_SEND_PACKET), 0);

	int c_index = CON_AC_CHECK(channel_id, 0);
	CON_RECEIVE(channel_id, 0, (unsigned char*)&receive_packet, sizeof(CONTROL_SEND_PACKET), c_index);

	return;
}


static double get_param_float(char* t_name, char* p_name)
{
	CONTROL_SEND_PACKET send_packet;
	CONTROL_SEND_PACKET receive_packet;

	int i=0;
	char *task_name;
	char *param_name;
	
	int my_control_task_id;
	int target_task_id;
	int param_id;

	char temp[255];
	char* my_task_name = temp;
	strcpy(my_task_name, CONTASKNAME);

	if(t_name == NULL)
	{
		my_control_task_id = get_my_contask_id(my_task_name);
		target_task_id = get_my_task_id(my_task_name);

		task_name = con_taskmap[target_task_id].t_name;
		param_name = p_name;
	}

	else
	{
		/* find target task id */
		for(i=0;i<ARRAYLEN(con_taskmap);i++){
			if(strcmp(t_name, con_taskmap[i].t_name)==0) {
				target_task_id = i;
				break;
			}
		}

		my_control_task_id = get_my_contask_id(my_task_name);
		task_name = t_name;
		param_name = p_name;
	}

	/* find param id */
	for(i=0;i<ARRAYLEN(param_list);i++) {
		if(strcmp(p_name, param_list[i].param_name)==0 &&
				target_task_id == param_list[i].task_id) {
			param_id = param_list[i].param_id;
			break;
		}
	}

	send_packet.valid = 1;
	send_packet.target_task_id = (unsigned char)target_task_id;
	send_packet.my_control_task_id = (unsigned char)my_control_task_id;
	send_packet.my_control_priority = (unsigned char)0;
	send_packet.my_control_group_id = (unsigned char)0;
	send_packet.command_type = (unsigned char)GET_PARAM_INT;
	send_packet.param_id = (unsigned char)param_id;
	send_packet.param_value = (void*)0;
	send_packet.my_go_time = get_current_time_base();

	int channel_id = init_con_port(my_task_name);
	
	CON_SEND(channel_id, 1, (unsigned char*)&send_packet, sizeof(CONTROL_SEND_PACKET), 0);

	int c_index = CON_AC_CHECK(channel_id, 0);
	CON_RECEIVE(channel_id, 0, (unsigned char*)&receive_packet, sizeof(CONTROL_SEND_PACKET), c_index);

	return *(double*)(receive_packet.param_value);
}


static void set_param_float(char* t_name, char* p_name, double p_value, int time_base_id, unsigned int time_offset)
{
	CONTROL_SEND_PACKET send_packet;
	CONTROL_SEND_PACKET receive_packet;

	unsigned int i;
	int index=-1;
	unsigned int target_task_id=0;
	int param_id=-1;
	int my_control_task_id = -1;
	char *task_name;
	char *param_name;

	char temp[255];
	char* my_task_name = temp;
	strcpy(my_task_name, CONTASKNAME);

	index = get_control_channel_index(my_control_task_id);

	if(t_name == NULL)
	{
		my_control_task_id = get_my_contask_id(my_task_name);
		target_task_id = get_my_task_id(my_task_name);

		task_name = con_taskmap[target_task_id].t_name;
		param_name = p_name;
	}

	else
	{
		/* find target task id */
		for(i=0;i<ARRAYLEN(con_taskmap);i++){
			if(strcmp(t_name, con_taskmap[i].t_name)==0) {
				target_task_id = i;
				break;
			}
		}

		my_control_task_id = get_my_contask_id(my_task_name);
		task_name = t_name;
		param_name = p_name;
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
	*(double*)(send_packet.param_value) = p_value;
	send_packet.my_go_time = get_current_time_base();

	int channel_id = init_con_port(my_task_name);
	CON_SEND(channel_id, 1, (unsigned char*)&send_packet, sizeof(CONTROL_SEND_PACKET), 0);

	int c_index = CON_AC_CHECK(channel_id, 0);
	CON_RECEIVE(channel_id, 0, (unsigned char*)&receive_packet, sizeof(CONTROL_SEND_PACKET), c_index);

	return;
}

static void run_task(char* t_name, int time_base_id, unsigned int time_offset)
{
	CONTROL_SEND_PACKET send_packet;
	CONTROL_SEND_PACKET receive_packet;

	unsigned int i;
	int index=-1;
	int target_task_id=-1;
	int my_control_task_id=-1;

	char temp[255];
	char* my_task_name = temp;
	strcpy(my_task_name, CONTASKNAME);

	my_control_task_id = get_my_contask_id(my_task_name);
	//printf("run_task start\\n");
	index = get_control_channel_index(my_control_task_id);
	/* find target task id */
	for(i=0;i<ARRAYLEN(con_taskmap);i++){
		if(strcmp(t_name, con_taskmap[i].t_name)==0) {
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
	send_packet.param_value = (void*) 0;
	send_packet.my_go_time = get_current_time_base();

	int channel_id = init_con_port(my_task_name);
	CON_SEND(channel_id, 1, (unsigned char*)&send_packet, sizeof(CONTROL_SEND_PACKET), 0);

	int c_index = CON_AC_CHECK(channel_id, 0);
	CON_RECEIVE(channel_id, 0, (unsigned char*)&receive_packet, sizeof(CONTROL_SEND_PACKET), c_index);

	return;
}

static void control_begin(int time_base)
{
	/*
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

	push_pqueue(send_packet);

	return;
	*/
	return;
}

static void control_end(int time_base_id, unsigned int time_offset)
{
	/*
	CONTROL_SEND_PACKET send_packet;

	int my_control_task_id = -1;
	int index=-1;

	my_control_task_id = get_my_task_id();
	index = get_control_channel_index(my_control_task_id);

	send_packet.my_control_task_id = (unsigned char)my_control_task_id;
	send_packet.my_control_priority = (unsigned char)control_channel[index].control_priority;
	send_packet.my_control_group_id = (unsigned char)control_channel[index].control_group_id;
	send_packet.command_type = (unsigned char)END;
	send_packet.my_go_time = control_channel[index].time_base[time_base_id] + time_offset;
	control_channel[index].time_base[time_base_id] = 0; // make valid slot 

	if((time_base_id + 1 == control_channel[index].empty_base_index) ||
			(time_base_id==MAX_TIME_BASE_COUNT && control_channel[index].empty_base_index==0))
	{
		control_channel[index].empty_base_index = time_base_id;
	}

	push_pqueue(send_packet);

	return;
	*/
	return;
}

static int get_timer_alarmed(unsigned int timer_id)
{
    /*
	unsigned int my_task_id;
	unsigned int cont_ch_index;
	int p_group, p;

	char temp[255];
	char* my_task_name = temp;
	strcat(my_task_name, "_");
	strcat(my_task_name, SUBCONTASKNAME);

	my_task_id = get_my_task_id(my_task_name);
	cont_ch_index = get_control_channel_index(my_task_id);

	p = control_channel[cont_ch_index].control_priority;
	p_group = control_channel[cont_ch_index].control_group_id;

	if(control_channel[cont_ch_index].timer_slot[timer_id]  <= get_current_time_base()) {
		control_channel[cont_ch_index].timer_slot[timer_id]=0; // reset timer
		control_channel[cont_ch_index].empty_slot_index = timer_id;
		return 0;
	}
	else
		return -1;
    */
	return 0;
}

static int set_timer(int time_base_id, unsigned int time_offset)
{
    /*
	int timer_id;
	unsigned int my_task_id;
	unsigned int cont_ch_index;

	char temp[255];
	char* my_task_name = temp;
	strcat(my_task_name, "_");
	strcat(my_task_name, SUBCONTASKNAME);

	my_task_id = get_my_task_id(my_task_name);
	cont_ch_index = get_control_channel_index(my_task_id);
	timer_id = get_valid_timer_slot_id(cont_ch_index);

	control_channel[cont_ch_index].timer_slot[timer_id] = get_current_time_base() + time_offset;
	return timer_id;
    */
	return 0;
}

static void reset_timer(unsigned int timer_id)
{
    /*
	unsigned int my_task_id;
	unsigned int cont_ch_index;

	char temp[255];
	char* my_task_name = temp;
	strcat(my_task_name, "_");
	strcat(my_task_name, SUBCONTASKNAME);

	my_task_id = get_my_task_id(my_task_name);
	cont_ch_index = get_control_channel_index(my_task_id);

	control_channel[cont_ch_index].timer_slot[timer_id]=0; // reset timer
    */
	return;
}

static void program_kill()
{
    printf("Application Terminated! (on SPE)\n");
	exit(-1);
}

static void program_stop()
{
}



#undef ROUNDUP
#undef ARRAYLEN
#undef PROC_DEBUG

