
#include <UFPort.h> 
#include <UFPort_deprecated.h>


#define SYS_REQ(x, ...) SYS_REQ_##x(__VA_ARGS__)
 

#define TASK_STATUS_RUN 0
#define TASK_STATUS_STOP 1
#define TASK_STATUS_WAIT 2

#define TASK_CODE_BEGIN
#define TASK_CODE_END
#define TASK_NAME "/*[NESTED_TASK_NAME]*/"
#define TASK_INIT void /*[NESTED_TASK_NAME]*/_Init/*[TASK_FUNCTION_ID]*/(int TASK_ID)
#define TASK_GO void /*[NESTED_TASK_NAME]*/_Go/*[TASK_FUNCTION_ID]*/()
#define TASK_WRAPUP void /*[NESTED_TASK_NAME]*/_Wrapup/*[TASK_FUNCTION_ID]*/()

#define STATIC static
#include "/*[TASK_CODE_FILE]*/"



