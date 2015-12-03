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
#define TASK_INIT CIC_T_VOID ##TASK_NAME_Init(CIC_T_INT TASK_ID)
#define TASK_GO CIC_T_VOID ##TASK_NAME_Go()
#define TASK_WRAPUP CIC_T_VOID ##TASK_NAME_Wrapup()

#define STATIC static
##CIC_INCLUDE

