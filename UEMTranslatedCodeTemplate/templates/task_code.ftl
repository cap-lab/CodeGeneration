

#define TASK_STATUS_RUN 0
#define TASK_STATUS_STOP 1
#define TASK_STATUS_WAIT 2

#define TASK_CODE_BEGIN
#define TASK_CODE_END
#define TASK_NAME "${task_info.name}"

#include <UFPort.h> 
#include <UFPort_deprecated.h>

#include <UFTask.h>
#include <UFTask_deprecated.h>

#include <UFControl.h>
#include <UFControl_deprecated.h>

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



