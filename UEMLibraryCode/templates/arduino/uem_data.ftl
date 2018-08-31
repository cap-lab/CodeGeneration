/* uem_data.c made by UEM Translator */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_data.h>

<#assign timerSlotSize=3 />
#define MAX_TIMER_SLOT_SIZE (${timerSlotSize})

<#list flat_task as task_name, task>
	<#if task.type == "CONTROL">
STimer g_stTimer_${task.name}[MAX_TIMER_SLOT_SIZE] = {
		<#list 0..(timerSlotSize-1) as index>
	{
		${index}, // Slot ID 
		INVALID_TIME_VALUE, // Timer value
		FALSE, // Alarm checked
		0, // Alarm time
	},
		</#list>
};

	</#if>
</#list>


// ##TASK_CODE_TEMPLATE::START
<#list flat_task as task_name, task>
	<#if !task.childTaskGraphName??>
		<#if task.language=="C" && gpu_used == false>
#ifdef __cplusplus
extern "C"
{
#endif 
		</#if>
		<#assign task_func_id=0 />
void ${task.name}_Init${task_func_id}(int nTaskId);
void ${task.name}_Go${task_func_id}(int nTaskId);
void ${task.name}_Wrapup${task_func_id}();
		<#if task.language=="C" && gpu_used == false>
#ifdef __cplusplus
}
#endif 
		</#if>
	</#if>

</#list>
// ##TASK_CODE_TEMPLATE::END


// ##LIBRARY_INIT_WRAPUP_TEMPLATE::START
<#list library_info as libraryName, library>
		<#if library.language=="C">
#ifdef __cplusplus
extern "C"
{
#endif 
		</#if>
void l_${libraryName}_init();
void l_${libraryName}_wrapup();
		<#if library.language=="C">
#ifdef __cplusplus
}
#endif 
		</#if>

</#list>
// ##LIBRARY_INIT_WRAPUP_TEMPLATE::END


// ##TASK_LIST_DECLARATION_TEMPLATE::START
<#list task_graph as graph_name, task_graph>
extern STask g_astTasks_${task_graph.name}[];
</#list>
// ##TASK_LIST_DECLARATION_TEMPLATE::END


// ##TASK_GRAPH_DECLARATION_TEMPLATE::START
<#list task_graph as graph_name, task_graph>
extern STaskGraph g_stGraph_${task_graph.name};
</#list>
// ##TASK_GRAPH_DECLARATION_TEMPLATE::END


// ##TASK_PARAMETER_TEMPLATE::START
<#list flat_task as task_name, task>
	<#if (task.taskParamList?size > 0)>
STaskParameter g_astTaskParameter_${task.name}[] = {
		<#list task.taskParamList as task_param>
	{
		${task_param.id},
		PARAMETER_TYPE_${task_param.type},
		"${task_param.name}",
		{ <#if task_param.type == "INT">.nParam = ${task_param.value?c},<#else>.dbParam = ${task_param.value?c}</#if> },
	},
		</#list>
};
	</#if>
</#list>
// ##TASK_PARAMETER_TEMPLATE::END


// ##TASK_LIST_TEMPLATE::START
<#assign task_func_id=0 />
<#list task_graph as graph_name, task_graph>
STask g_astTasks_${task_graph.name}[] = {
	<#list task_graph.taskList as task>
	{ 	${task.id}, // Task ID
		"${task.name}", // Task name
		TASK_TYPE_${task.type}, // Task Type
		{
			<#if !task.childTaskGraphName??>${task.name}_Init${task_func_id}<#else>NULL</#if>, // Task init function
			<#if !task.childTaskGraphName??>${task.name}_Go${task_func_id}<#else>NULL</#if>, // Task go function
			<#if !task.childTaskGraphName??>${task.name}_Wrapup${task_func_id}<#else>NULL</#if>, // Task wrapup function
		} // Task function array
		RUN_CONDITION_${task.runCondition}, // Run condition
		1, // Run rate
		${task.period?c}, // Period
		TIME_METRIC_${task.periodMetric}, // Period metric
		<#if task.childTaskGraphName??>&g_stGraph_${task.childTaskGraphName}<#else>(STaskGraph *) NULL</#if>, // Subgraph
		&g_stGraph_${task.parentTaskGraphName}, // Parent task graph
		<#if (task.taskParamList?size > 0)>g_astTaskParameter_${task.name}<#else>(STaskParameter *) NULL</#if>, // Task parameter information
		${task.taskParamList?size}, // Task parameter number
		<#if task.staticScheduled == true>TRUE<#else>FALSE</#if>, // Statically scheduled or not
		${task.iterationCountList["0"]}, // Task iteration count (only used when the parent task graph is data flow)
		0, // current run count in iteration
		0, // current iteration
		0, // target iteration (this variable is used for calling delayed stop task)
		<#if task.type == "CONTROL">g_stTimer_${task.name}<#else>(STimer *) NULL</#if>, // Timer slot (used by control task)
	},
	</#list>
};

</#list>

// ##TASK_LIST_TEMPLATE::END

// ##TASK_GRAPH_TEMPLATE::START
<#list task_graph as graph_name, task_graph_element>
STaskGraph g_stGraph_${task_graph_element.name} = {
		GRAPH_TYPE_${task_graph_element.taskGraphType}, // Task graph type
		g_astTasks_${task_graph_element.name}, // current task graph's task list
		${task_graph_element.taskList?size}, // number of tasks
		<#if task_graph_element.parentTask??>&g_astTasks_${task_graph_element.parentTask.parentTaskGraphName}[${task_graph_element.parentTask.inGraphIndex}]<#else>(STask *) NULL</#if>, // parent task
};

</#list>
// ##TASK_GRAPH_TEMPLATE::END


// ##TASK_ID_TO_TASK_MAP_TEMPLATE::START
STaskIdToTaskMap g_astTaskIdToTask[] = {
<#list flat_task as task_name, task>
	{ 	${task.id}, // Task ID
		"${task.name}", // Task name
		&g_astTasks_${task.parentTaskGraphName}[${task.inGraphIndex}], // Task structure pointer
	},
</#list>
};
// ##TASK_ID_TO_TASK_MAP_TEMPLATE::END


<#macro printScheduledCode scheduleItem space compositeMappedProcessor parentTaskName>
	<#if scheduleItem.itemType == "LOOP">
		<#if (scheduleItem.repetition > 1) >
${space}for(${scheduleItem.variableName} = 0 ; ${scheduleItem.variableName} < ${scheduleItem.repetition} ; ${scheduleItem.variableName}++)
${space}{
		</#if>
		<#list scheduleItem.scheduleItemList as loop_schedule_item>
			<#if (scheduleItem.repetition <= 1) >
				<#assign newspace="${space}" />
			<#else>
				<#assign newspace="${space}	" />
			</#if>
			<@printScheduledCode loop_schedule_item newspace compositeMappedProcessor parentTaskName />
		</#list>
		<#if (scheduleItem.repetition > 1) >		
${space}}

		</#if>
	<#else>
		<#if (scheduleItem.repetition > 1) >
${space}for(${scheduleItem.variableName} = 0 ; ${scheduleItem.variableName} < ${scheduleItem.repetition} ; ${scheduleItem.variableName}++)
${space}{
			<#assign innerspace="${space}	" />
		<#else>
			<#assign innerspace="${space}" />
		</#if>
		<#if compositeMappedProcessor.srcTaskMap[scheduleItem.taskName]?? && (flat_task[parentTaskName].modeTransition.modeMap?size > 1)>
${innerspace}{
${innerspace}	EModeState enModeState = MODE_STATE_TRANSITING;
${innerspace}	uem_result result;
${innerspace}	STask *pstTask = NULL;
${innerspace}	
${innerspace}	enModeState = UKModeTransition_GetModeState(nTaskId);
${innerspace}
${innerspace}	if(enModeState == MODE_STATE_TRANSITING)
${innerspace}	{
${innerspace}		${scheduleItem.taskName}_Go${scheduleItem.taskFuncId}(${flat_task[scheduleItem.taskName].id});//UEM_DEBUG_PRINT("${scheduleItem.taskName}_Go${scheduleItem.taskFuncId} called-- (Line: %d)\n", __LINE__);
${innerspace}		result = UKTask_GetTaskFromTaskId(nTaskId, &pstTask);
${innerspace}		if(result == ERR_UEM_NOERROR)
${innerspace}		{
${innerspace}			result = UCThreadMutex_Lock(pstTask->hMutex);
${innerspace}			if(result == ERR_UEM_NOERROR){
${innerspace}				transitMode_${parentTaskName}(g_astTasks_${flat_task[parentTaskName].parentTaskGraphName}[${flat_task[parentTaskName].inGraphIndex}].pstMTMInfo);
${innerspace}				UCThreadMutex_Unlock(pstTask->hMutex);
${innerspace}			}	
${innerspace}			return; // exit when the mode is MODE_STATE_TRANSITING
${innerspace}		}
${innerspace}	}
${innerspace}}
		<#else>
${innerspace}${scheduleItem.taskName}_Go${scheduleItem.taskFuncId}(${flat_task[scheduleItem.taskName].id});//UEM_DEBUG_PRINT("${scheduleItem.taskName}_Go${scheduleItem.taskFuncId} called (Line: %d)\n", __LINE__);
		</#if>
		<#if compositeMappedProcessor.srcTaskMap[scheduleItem.taskName]??>
${innerspace}{
${innerspace}	EInternalTaskState enState = INTERNAL_STATE_STOP;
${innerspace}	UKTask_GetTaskState(nTaskId, "${parentTaskName}", &enState);
${innerspace}	if(enState == INTERNAL_STATE_STOP || enState == INTERNAL_STATE_END) return; 
${innerspace}}
		</#if>			
		<#if (scheduleItem.repetition > 1) >
${space}}

		</#if>
	</#if>
</#macro>

// ##SCHEDULED_COMPOSITE_TASK_FUNCTION_IMPLEMENTATION::START
<#list schedule_info as task_name, mapped_schedule>
	<#list mapped_schedule.mappedProcessorList as compositeMappedProcessor>
		<#list compositeMappedProcessor.compositeTaskScheduleList as task_schedule>
void ${mapped_schedule.parentTaskName}_${compositeMappedProcessor.modeId}_${compositeMappedProcessor.processorId}_${compositeMappedProcessor.processorLocalId}_${task_schedule.throughputConstraint?c}_Go(int nTaskId) 
{
			<#if (task_schedule.maxLoopVariableNum > 0) >
				<#list 0..(task_schedule.maxLoopVariableNum-1) as variable_id>
	int depth${variable_id};
				</#list>

			</#if>
<#list task_schedule.scheduleList as scheduleItem>
	<@printScheduledCode scheduleItem "	" compositeMappedProcessor mapped_schedule.parentTaskName />
</#list>
}

		</#list>
	</#list>
</#list>
// ##SCHEDULED_COMPOSITE_TASK_FUNCTION_IMPLEMENTATION::END

// ##SCHEDULED_COMPOSITE_TASKS_TEMPLATE::START
<#list schedule_info as task_name, scheduled_task>
	<#list scheduled_task.mappedProcessorList as compositeMappedProcessor>
SScheduleList g_astScheduleList_${scheduled_task.parentTaskName}_${compositeMappedProcessor.modeId}_${compositeMappedProcessor.processorId}_${compositeMappedProcessor.processorLocalId}[] = {
		<#list compositeMappedProcessor.compositeTaskScheduleList as task_schedule>
	{
		${scheduled_task.parentTaskName}_${compositeMappedProcessor.modeId}_${compositeMappedProcessor.processorId}_${compositeMappedProcessor.processorLocalId}_${task_schedule.throughputConstraint?c}_Go, // Composite GO function
		${task_schedule.throughputConstraint?c}, // Throughput constraint
		<#if task_schedule.hasSourceTask == true>TRUE<#else>FALSE</#if>,
	},
		</#list>
};
	</#list>
</#list>
// ##SCHEDULED_COMPOSITE_TASKS_TEMPLATE::END


SScheduledTasks g_astScheduledTaskList[] = {
<#list schedule_info as task_name, mapped_schedule>
	<#list mapped_schedule.mappedProcessorList as compositeMappedProcessor>
	{	<#if mapped_schedule.parentTaskId == -1>NULL<#else>&g_astTasks_${flat_task[task_name].parentTaskGraphName}[${flat_task[task_name].inGraphIndex}]</#if>, // Parent Task ID
		${mapped_schedule.parentTaskName}_${compositeMappedProcessor.modeId}_${compositeMappedProcessor.processorId}_${compositeMappedProcessor.processorLocalId}_${compositeMappedProcessor.compositeTaskScheduleList[0].throughputConstraint?c}_Go, // Composite GO function 
	},
	</#list>
</#list>
};

// ##LIBRARY_INFO_TEMPLATE::START
SLibrary g_stLibraryInfo[] = {
<#list library_info as libraryName, library>
	{
		"${libraryName}",
		l_${libraryName}_init,
		l_${libraryName}_wrapup,
	},
</#list>
};

// ##LIBRARY_INFO_TEMPLATE::END


int g_nNumOfTasks_top = ARRAYLEN(g_astTasks_top);
int g_nTaskIdToTaskNum = ARRAYLEN(g_astTaskIdToTask);
int g_nLibraryInfoNum = <#if (library_info?size > 0)>ARRAYLEN(g_stLibraryInfo)<#else>0</#if>;
int g_nTimerSlotNum = MAX_TIMER_SLOT_SIZE;
int g_nScheduledTaskListNum = ARRAYLEN(g_astScheduledTaskList);


