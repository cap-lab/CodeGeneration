

#ifndef __LIB_PORT_H__
#define __LIB_PORT_H__

#ifdef __PPU__

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

extern int get_mytask_id();
extern int lock_lib_channel(int channel_id);
extern int unlock_lib_channel(int channel_id);
extern int init_lib_port(int task_id);
extern int read_libport(int channel_id, int func_ret, unsigned char *buf, int len, int index);
extern int write_libport(int channel_id, int func_ret, unsigned char *buf, int len, int index);
extern int lib_available(int channel_id, int func_ret, int index);
extern int check_libport(int channel_id, int func_ret);

#define LIB_RECEIVE(a, b, c, d, e) read_libport(a, b, (unsigned char *)c, d, e)
#define LIB_SEND(a, b, c, d, e) write_libport(a, b, (unsigned char *)c, d, e)
#define LIB_AC_CHECK(a, b) check_libport(a, b)

#elif defined(__SPU__)

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <mars/task.h>

//static int lock_lib_channel(int channel_id);
//static int unlock_lib_channel(int channel_id);
static int init_lib_port(int task_id);
static int read_libport(int channel_id, int func_ret, unsigned char *buf, int len, int index);
static int write_libport(int channel_id, int func_ret, unsigned char *buf, int len, int index);
static int check_libport(int channel_id, int func_ret);

#define LIB_RECEIVE(a, b, c, d, e) read_libport(a, b, (unsigned char *)c, d, e)
#define LIB_SEND(a, b, c, d, e) write_libport(a, b, (unsigned char *)c, d, e)
#define LIB_AC_CHECK(a, b) check_libport(a, b)


#else

    #error

#endif

#endif /* __LIB_PORT_H__ */

