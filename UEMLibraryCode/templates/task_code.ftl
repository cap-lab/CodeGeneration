
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif


#define TASK_STATUS_RUN 0
#define TASK_STATUS_STOP 1
#define TASK_STATUS_WAIT 2

#define TASK_CODE_BEGIN
#define TASK_CODE_END
#define TASK_NAME "${task_info.name}"

<#if task_gpu_mapping_info??>
#define GRID __grid

#define GRID_X ${task_gpu_mapping_info.inBlockSizeWidth}
#define GRID_Y ${task_gpu_mapping_info.inBlockSizeHeight}
#define GRID_Z ${task_gpu_mapping_info.inBlockSizeDepth}

static dim3 __grid(GRID_X, GRID_Y, GRID_Z);

#define THREADS __threads

#define THREAD_X ${task_gpu_mapping_info.inThreadSizeWidth}
#define THREAD_Y ${task_gpu_mapping_info.inThreadSizeHeight}
#define THREAD_Z ${task_gpu_mapping_info.inThreadSizeDepth}

static dim3 __threads(THREAD_X, THREAD_Y, THREAD_Z);

#define KERNEL_CALL(x, ...) x<<<GRID, THREADS>>>(__VA_ARGS__);
</#if>


#include <UFPort.h> 
#include <UFPort_deprecated.h>

#include <UFTask.h>
#include <UFTask_deprecated.h>

#include <UFControl.h>
#include <UFControl_deprecated.h>

#include <UFSystem.h>
#include <UFSystem_deprecated.h>

<#if task_info.masterPortToLibraryMap??>
#define LIBCALL(x, ...) LIBCALL_##x(__VA_ARGS__)
	<#list task_info.masterPortToLibraryMap as portName, library>
#define LIBCALL_${portName}(f, ...) l_${library.name}_##f(__VA_ARGS__)
#include "${library.header}"
	</#list>
</#if>

#define SYS_REQ(x, ...) SYS_REQ_##x(__VA_ARGS__)

#define TASK_INIT void ${task_info.name}_Init${task_func_id}(int TASK_ID)
#define TASK_GO void ${task_info.name}_Go${task_func_id}(int nTaskId)
#define TASK_WRAPUP void ${task_info.name}_Wrapup${task_func_id}()

#define STATIC static

<#list task_info.extraHeaderSet as headerFile>
#include "${headerFile}"
</#list>

#include "${task_info.taskCodeFile}"



