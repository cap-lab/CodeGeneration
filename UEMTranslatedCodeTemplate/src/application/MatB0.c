
#include <UFPort.h> 
#include <UFPort_deprecated.h>


#define SYS_REQ(x, ...) SYS_REQ_##x(__VA_ARGS__)
 

#define TASK_STATUS_RUN 0
#define TASK_STATUS_STOP 1
#define TASK_STATUS_WAIT 2

#define TASK_CODE_BEGIN
#define TASK_CODE_END
#define TASK_NAME "MatB"
#define TASK_INIT void MatB_Init0(int TASK_ID)
#define TASK_GO void MatB_Go0()
#define TASK_WRAPUP void MatB_Wrapup0()

#define STATIC static
#include "MatB.cic"



