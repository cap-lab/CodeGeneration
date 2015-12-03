
#include "helper_functions.h"
#include "helper_cuda.h"
#define CUDA_ERROR_CHECK checkCudaErrors

##LIBRARY_DEFINITION
##SYSPORT_DEFINITION
//////////////////////////////
##TASK_INDEX_DEFINITION
##SYNC_DEFINITION
//////////////////////////////
##DEVICE_BUFFER_DEFINITION
##DIMM_DEFINITION
##KERNEL_DEFINITION
//////////////////////////////

// for CIC port api
extern "C" {

#define TASK_CODE_BEGIN
#define TASK_CODE_END
#define TASK_INIT void ##TASK_NAME_init(int TASK_ID)
#define TASK_GO void ##TASK_NAME_go()
#define TASK_WRAPUP void ##TASK_NAME_wrapup()
##PRE_POST_DEFINITION

#define STATIC static

#include "CIC_CUDA_port.h"

##CIC_PORT_CUDA_INCLUDE

##CIC_INCLUDE
}
