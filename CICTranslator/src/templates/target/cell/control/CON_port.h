// produced by generateOutput() of genPTHREAD.py

#ifndef __C_PORT_H__
#define __C_PORT_H__

#ifdef __PPU__

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

extern int read_con_acport(int channel_id, int send_recv, unsigned char *buf, int len, int index);
extern int write_con_acport(int channel_id, int send_recv, unsigned char *buf, int len, int index);
extern int check_con_acport(int channel_id, int send_recv);

#define CON_RECEIVE(a, b, c, d, e) read_con_acport(a, b, (unsigned char *)c, d, e)
#define CON_SEND(a, b, c, d, e) write_con_acport(a, b, (unsigned char *)c, d, e)
#define CON_AC_CHECK(a, b) check_con_acport(a, b)


extern long get_param_int(char* t_name, char* p_name);
extern void set_param_int(char* t_name, char* p_name, long p_value, int time_base_id, unsigned int time_offset);
extern double get_param_float(char* t_name, char* p_name);
extern void set_param_float(char* t_name, char* p_name, double p_value, int time_base_id, unsigned int time_offset);
extern void run_task(char* t_name, int time_base_id, unsigned int time_offset);
extern void control_begin(int time_base_id);
extern void control_end(int time_base_id, unsigned int time_offset);
extern unsigned int get_current_time_base();
extern int set_timer(int time_base_id, unsigned int time_offset);
extern int get_timer_alarmed(unsigned int timer_id);
extern void reset_timer(unsigned int timer_id);
extern void program_kill();
extern void program_stop();

#define SYS_REQ_GET_PARAM_INT(a, b) get_param_int(a, b)
#define SYS_REQ_SET_PARAM_INT(a, b, c, d, e) set_param_int(a, b, c, d, e)
#define SYS_REQ_GET_PARAM_FLOAT(a, b) get_param_float(a, b)
#define SYS_REQ_SET_PARAM_FLOAT(a, b, c, d, e) set_param_float(a, b, c, d, e)
#define SYS_REQ_RUN_TASK(a, b, c) run_task(a, b, c)
#define SYS_REQ_CONTROL_BEGIN(a) control_begin(a)
#define SYS_REQ_CONTROL_END(a, b) control_end(a, b)
#define SYS_REQ_GET_CURRENT_TIME_BASE() get_current_time_base()
#define SYS_REQ_SET_TIMER(a, b) set_timer(a, b)
#define SYS_REQ_GET_TIMER_ALARMED(a) get_timer_alarmed(a)
#define SYS_REQ_RESET_TIMER(a) reset_timer(a)
#define SYS_REQ_KILL() program_kill()
#define SYS_REQ_STOP() program_stop()

#elif defined(__SPU__)

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <mars/task.h>

static int init_con_port(char* my_task_name);
static int read_con_acport(int channel_id, int send_recv, unsigned char *buf, int len, int index);
static int write_con_acport(int channel_id, int send_recv, unsigned char *buf, int len, int index);
static int check_con_acport(int channel_id, int send_recv);

#define CON_RECEIVE(a, b, c, d, e) read_con_acport(a, b, (unsigned char *)c, d, e)
#define CON_SEND(a, b, c, d, e) write_con_acport(a, b, (unsigned char *)c, d, e)
#define CON_AC_CHECK(a, b) check_con_acport(a, b)

static long get_param_int(char* t_name, char* p_name);
static void set_param_int(char* t_name, char* p_name, long p_value, int time_base_id, unsigned int time_offset);
static double get_param_float(char* t_name, char* p_name);
static void set_param_float(char* t_name, char* p_name, double p_value, int time_base_id, unsigned int time_offset);
static void run_task(char* t_name, int time_base_id, unsigned int time_offset);
static void control_begin(int time_base);
static void control_end(int time_base_id, unsigned int time_offset);
static unsigned int get_current_time_base();
static int set_timer(int time_base_id, unsigned int time_offset);
static int get_timer_alarmed(unsigned int timer_id);
static void reset_timer(unsigned int timer_id);
static void program_kill();
static void program_stop();

#define SYS_REQ_GET_PARAM_INT(a, b) get_param_int(a, b)
#define SYS_REQ_SET_PARAM_INT(a, b, c, d, e) set_param_int(a, b, c, d, e)
#define SYS_REQ_GET_PARAM_FLOAT(a, b) get_param_float(a, b)
#define SYS_REQ_SET_PARAM_FLOAT(a, b, c, d, e) set_param_float(a, b, c, d, e)
#define SYS_REQ_RUN_TASK(a, b, c) run_task(a, b, c)
#define SYS_REQ_CONTROL_BEGIN(a) control_begin(a)
#define SYS_REQ_CONTROL_END(a, b) control_end(a, b)
#define SYS_REQ_GET_CURRENT_TIME_BASE() get_current_time_base()
#define SYS_REQ_SET_TIMER(a, b) set_timer(a, b)
#define SYS_REQ_GET_TIMER_ALARMED(a) get_timer_alarmed(a)
#define SYS_REQ_RESET_TIMER(a) reset_timer(a)
#define SYS_REQ_KILL() program_kill()
#define SYS_REQ_STOP() program_stop()


#else

    #error

#endif

#endif /* __PORT_H__ */

