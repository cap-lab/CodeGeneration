/* uem_data.c made by UEM Translator */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_data.h>

#include <UKTaskScheduler.h>

<#assign timerSlotSize=3 />
#define MAX_TIMER_SLOT_SIZE (${timerSlotSize})

<#include "../uem_common_data.ftl">

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
		}, // Task function array
		RUN_CONDITION_${task.runCondition}, // Run condition
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


SScheduledTasks g_astScheduledTaskList[] = {
<#list schedule_info as task_name, mapped_schedule>
	<#list mapped_schedule.mappedProcessorList as compositeMappedProcessor>
	{	<#if mapped_schedule.parentTaskId == -1>NULL<#else>&g_astTasks_${flat_task[task_name].parentTaskGraphName}[${flat_task[task_name].inGraphIndex}]</#if>, // Parent Task ID
		${mapped_schedule.parentTaskName}_${compositeMappedProcessor.modeId}_${compositeMappedProcessor.processorId}_${compositeMappedProcessor.processorLocalId}_${compositeMappedProcessor.compositeTaskScheduleList[0].throughputConstraint?c}_Go, // Composite GO function 
	},
	</#list>
</#list>
};

// Target general task: control task
SGeneralTaskRuntimeInfo g_astControlTaskRuntimeInfo[] = {
<#list flat_task as task_name, task>
	<#if task.staticScheduled == false && !task.childTaskGraphName?? && task.type == "CONTROL">
	{	&g_astTasks_${task.parentTaskGraphName}[${task.inGraphIndex}], // task structure pointer
		0, // next run time
		1, // remained run count inside millisec
		<#if task.runCondition != "CONTROL_DRIVEN">TRUE<#else>FALSE</#if>,	// running
	},
	</#if>
</#list>
};


// Target general task: not static scheduled, no child task, not control task
SGeneralTaskRuntimeInfo g_astGeneralTaskRuntimeInfo[] = {
<#list flat_task as task_name, task>
	<#if task.staticScheduled == false && !task.childTaskGraphName?? && task.type != "CONTROL">
	{	&g_astTasks_${task.parentTaskGraphName}[${task.inGraphIndex}], // task structure pointer
		0, // next run time
		1, // remained run count inside millisec
		<#if task.runCondition != "CONTROL_DRIVEN">TRUE<#else>FALSE</#if>,	// running
	},
	</#if>
</#list>
};

SCompositeTaskRuntimeInfo g_astCompositeTaskRuntimeInfo[] = {
<#list schedule_info as task_name, mapped_schedule>
	<#list mapped_schedule.mappedProcessorList as compositeMappedProcessor>
	{ &g_astScheduledTaskList[${compositeMappedProcessor?index}], // composite task schedule pointer
	  0, // next run time
	  1, // run count inside millisec
	  TRUE, // running
	},
	</#list>
</#list>
};

int g_nControlTaskNum = ARRAYLEN(g_astControlTaskRuntimeInfo);
int g_nGeneralTaskNum = ARRAYLEN(g_astGeneralTaskRuntimeInfo);
int g_nCompositeTaskNum = ARRAYLEN(g_astCompositeTaskRuntimeInfo);


int g_nNumOfTasks_top = ARRAYLEN(g_astTasks_top);
int g_nTaskIdToTaskNum = ARRAYLEN(g_astTaskIdToTask);
int g_nLibraryInfoNum = <#if (library_info?size > 0)>ARRAYLEN(g_stLibraryInfo)<#else>0</#if>;
int g_nTimerSlotNum = MAX_TIMER_SLOT_SIZE;
int g_nScheduledTaskListNum = ARRAYLEN(g_astScheduledTaskList);


