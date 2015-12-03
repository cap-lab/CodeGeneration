##LIBRARY_DEFINITION
##SYSPORT_DEFINITION

##INCLUDE_HEADERS

#define Run 0
#define Stop 1
#define Wait 2

#define TASK_CODE_BEGIN
#define TASK_CODE_END
#define TASK_NAME "##TASK_NAME"	//tasks[TASK_ID].name
#define TASK_INIT void ##TASK_NAME_init(int TASK_ID)
#define TASK_GO void ##TASK_NAME_go(int TASK_ID)
#define TASK_WRAPUP void ##TASK_NAME_wrapup(int TASK_ID)

##MTM_DEFINITION

#define STATIC static
##CIC_INCLUDE

