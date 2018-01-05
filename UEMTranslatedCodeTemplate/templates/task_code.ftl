
#include <UFPort.h> 
#include <UFPort_deprecated.h>

#include <UFTask.h>
#include <UFTask_deprecated.h>

<#if task_info.linkedLibraryNameList??>
	<#list task_info.linkedLibraryNameList as libraryHeader>
#include "${libraryHeader}"
	</#list>
</#if>

#define SYS_REQ(x, ...) SYS_REQ_##x(__VA_ARGS__)
 

#define TASK_STATUS_RUN 0
#define TASK_STATUS_STOP 1
#define TASK_STATUS_WAIT 2

#define TASK_CODE_BEGIN
#define TASK_CODE_END
#define TASK_NAME "${task_info.name}"
#define TASK_INIT void ${task_info.name}_Init${task_func_id}(int TASK_ID)
#define TASK_GO void ${task_info.name}_Go${task_func_id}()
#define TASK_WRAPUP void ${task_info.name}_Wrapup${task_func_id}()

#define STATIC static
#include "${task_info.taskCodeFile}"



