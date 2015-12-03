##LIBRARY_DEFINITION
##SYSPORT_DEFINITION

##INCLUDE_HEADERS

#ifdef WIN32
#include <windows.h>
#endif

#ifdef WIN32
#define SLEEP(a) Sleep(a*1000)
#else
#define SLEEP(a) sleep(a)
#endif

#define RUN 0
#define STOP 1
#define WAIT 2

#define TASK_CODE_BEGIN
#define TASK_CODE_END
#define TASK_NAME "##TASK_NAME"
#define TASK_INIT void ##TASK_NAME_init(int TASK_ID)
#define TASK_GO __declspec(target(mic)) void ##TASK_NAME_go(volatile long ** task_data_addr, volatile long ** task_check_addr, int *task_data_size, int parallel_id, int max_parallel, char* argument)
#define TASK_WRAPUP void ##TASK_NAME_wrapup()

##MTM_DEFINITION

#define STATIC static
##CIC_INCLUDE

