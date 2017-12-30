
#include <UFPort.h> 
#include <UFPort_deprecated.h>


#define SYS_REQ(x, ...) SYS_REQ_##x(__VA_ARGS__)
 

#define TASK_STATUS_RUN 0
#define TASK_STATUS_STOP 1
#define TASK_STATUS_WAIT 2

#define TASK_CODE_BEGIN
#define TASK_CODE_END
#define TASK_NAME "VecMul"
#define TASK_INIT void VecMul_Init0(int TASK_ID)
#define TASK_GO void VecMul_Go0()
#define TASK_WRAPUP void VecMul_Wrapup0()

#define STATIC static
#include "VecMul.cic"



