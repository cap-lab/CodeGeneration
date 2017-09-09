
#define SYS_REQ(x, ...) SYS_REQ_##x(__VA_ARGS__)
 

#define TASK_STATUS_RUN 0
#define TASK_STATUS_STOP 1
#define TASK_STATUS_WAIT 2

#define TASK_CODE_BEGIN
#define TASK_CODE_END
#define TASK_NAME "Display"
#define TASK_INIT void Display_Init(int TASK_ID)
#define TASK_GO void Display_Go()
#define TASK_WRAPUP void Display_Wrapup()

#define STATIC static
#include "Display.cic"


