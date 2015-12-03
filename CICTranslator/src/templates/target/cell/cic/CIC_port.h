
// produced by generateOutput() of genPTHREAD.py

#ifndef __PORT_H__
#define __PORT_H__

#ifdef __PPU__

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

extern int init_port(int task_id, int port_id);
extern int init_task_port(int task_id, const char* port_name);
extern int read_port(int channel_id, unsigned char *buf, int len);
extern int write_port(int channel_id, unsigned char *buf, int len);
extern int available(int channel_id);
extern int read_acport(int channel_id, unsigned char *buf, int len, int index);
extern int write_acport(int channel_id, unsigned char *buf, int len, int index);
extern int check_acport(int channel_id);

#define BUF_RECEIVE(a, b, c) read_port(a, (unsigned char *)b, c)
#define BUF_SEND(a, b, c) write_port(a, (unsigned char *)b, c)
#define MQ_RECEIVE(a, b, c) read_port(a, (unsigned char *)b, c)
#define MQ_SEND(a, b, c) write_port(a, (unsigned char *)b, c)
#define AC_RECEIVE(a, b, c, d) read_acport(a, (unsigned char *)b, c, d)
#define AC_SEND(a, b, c, d) write_acport(a, (unsigned char *)b, c, d)
#define AC_CHECK(a) check_acport(a)

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

static int init_port(int task_id, int port_id);
static int init_task_port(int task_id, const char* port_name);
//static int read_port(int channel_id, unsigned char *buf, int len);
//static int write_port(int channel_id, unsigned char *buf, int len);
//static int available(int channel_id);
static int read_acport(int channel_id, unsigned char *buf, int len, int index);
static int write_acport(int channel_id, unsigned char *buf, int len, int index);
static int check_acport(int channel_id);

//#define MQ_RECEIVE(a, b, c) read_port(a, (unsigned char *)b, c)
//#define MQ_SEND(a, b, c) write_port(a, (unsigned char *)b, c)
#define AC_RECEIVE(a, b, c, d) read_acport(a, (unsigned char *)b, c, d)
#define AC_SEND(a, b, c, d) write_acport(a, (unsigned char *)b, c, d)
#define AC_CHECK(a) check_acport(a)

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

