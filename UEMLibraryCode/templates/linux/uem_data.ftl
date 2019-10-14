/* uem_data.c made by UEM Translator */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_data.h>
#include <UKTask.h>
#include <UKModeTransition.h>
#include <UKHostSystem.h>
#include <UKLoopModelController.h>
#include <UKModeTransitionModelController.h>

<#if gpu_used == true>
#include <UKGPUSystem.h>
</#if>

SExecutionTime g_stExecutionTime = { ${execution_time.value?c}, TIME_METRIC_${execution_time.metric} } ;


<#assign timerSlotSize=10 />
#define MAX_TIMER_SLOT_SIZE (${timerSlotSize})

<#list flat_task as task_name, task>
	<#if task.modeTransition??>
		<#if (task.modeTransition.modeMap?size > 1)>
static uem_bool transitMode_${task.name}(SModeTransitionMachine *pstModeTransition);
		</#if>
	</#if>
</#list>

<#include "../uem_common_data.ftl">

// ##LOOP_STRUCTURE_TEMPLATE::START
<#list flat_task as task_name, task>
	<#if task.loopStruct??>
SLoopInfo g_stLoopStruct_${task.name} = {
	LOOP_TYPE_${task.loopStruct.loopType}, // loop Type
	${task.loopStruct.loopCount?c}, // loop count
	${task.loopStruct.designatedTaskId}, // designated task id
};

	</#if>
</#list>
// ##LOOP_STRUCTURE_TEMPLATE::END

// ##MODE_TRANSITION_TEMPLATE::START
<#list flat_task as task_name, task>
	<#if task.modeTransition??>
		<#list task.modeTransition.modeMap as task_name, task_mode>
STask *g_pastRelatedChildTasks_${task.name}_${task_mode.name}[] = {
			<#list task_mode.relatedChildTaskSet as child_task>
	&g_astTasks_${child_task.parentTaskGraphName}[${child_task.inGraphIndex}],
			</#list>
};
		</#list>
		
SModeMap g_astModeMap_${task.name}[] = {
		<#list task.modeTransition.modeMap as mode_name, task_mode>
	{
		${task_mode.id},
		"${task_mode.name}",
		g_pastRelatedChildTasks_${task.name}_${task_mode.name},
		${task_mode.relatedChildTaskSet?size},
	},
		</#list>
};

SVariableIntMap g_astVariableIntMap_${task.name}[] = {
		<#list task.modeTransition.variableMap as var_name, var_type>
	{
		${var_name?index},
		"${var_name}",
		0, 
	},
		</#list>
};

		<#if (task.modeTransition.modeMap?size > 1)>
static uem_bool transitMode_${task.name}(SModeTransitionMachine *pstModeTransition) 
{
	uem_bool bModeChanged = FALSE;
			<#list task.modeTransition.variableMap as var_name, var_type>
	int ${var_name};
			</#list>
	int nCurrentModeId = pstModeTransition->astModeMap[pstModeTransition->nCurModeIndex].nModeId;
	int nNextModeId = nCurrentModeId;
	int nVarIndex = 0;
	
			<#list task.modeTransition.variableMap as var_name, var_type>
	nVarIndex = UKModeTransition_GetVariableIndexByName(pstModeTransition, "${var_name}");
	${var_name} = pstModeTransition->astVarIntMap[nVarIndex].nValue;
			</#list>
		
			<#list task.modeTransition.transitionList as transition>
	if(nCurrentModeId == ${task.modeTransition.modeMap[transition.srcMode].id}
	  <#list transition.conditionList as condition>&& ${condition.leftOperand} ${condition.operator} ${condition.rightOperand}</#list> )
	{
		nNextModeId = ${task.modeTransition.modeMap[transition.dstMode].id};
		bModeChanged = TRUE;
	}
			</#list>
	if(bModeChanged == TRUE)
	{	// update only the mode is changed
		pstModeTransition->nNextModeIndex = UKModeTransition_GetModeIndexByModeId(pstModeTransition, nNextModeId);
		pstModeTransition->enModeState = MODE_STATE_TRANSITING;
	}
	
	return bModeChanged;
}
		</#if>

SModeTransitionMachine g_stModeTransition_${task.name} = {
	${task.id},
	g_astModeMap_${task.name}, // mode list
	${task.modeTransition.modeMap?size}, // number of modes
	g_astVariableIntMap_${task.name}, // Integer variable list
	${task.modeTransition.variableMap?size}, // number of integer variables
	<#if (task.modeTransition.modeMap?size > 1)>transitMode_${task.name}<#else>NULL</#if>, // mode transition function
	0, // Current mode index
	0, // Next mode index
	MODE_STATE_TRANSITING, // mode state (to decide source task execution)
};
	</#if>
</#list>
// ##MODE_TRANSITION_TEMPLATE::END


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

// ##TASK_FUNCTION_LIST::START
<#list flat_task as task_name, task>
STaskFunctions g_ast_${task.name}_functions[] = {
	<#if !task.childTaskGraphName??>
		<#list 0..(task.taskFuncNum-1) as task_func_id>
	{
		${task.name}_Init${task_func_id}, // Task init function
		${task.name}_Go${task_func_id}, // Task go function
		${task.name}_Wrapup${task_func_id}, // Task wrapup function
	},
		</#list>
	</#if>
};

</#list>
// ##TASK_FUNCTION_LIST::END

// ##TASK_THREAD_CONTEXT_LIST::START
<#list flat_task as task_name, task>
STaskThreadContext g_ast_${task.name}_thread_context[] = {
	<#if !task.childTaskGraphName??>
		<#list 0..(task.taskFuncNum-1) as task_func_id>
	{
		0, // current run index used for getting loop task iteration
		0, // current run count of thread
		0, // target run count of thread
	},
		</#list>
	</#if>
};

</#list>
// ##TASK_THREAD_CONTEXT_LIST::END

// ##TASK_ITERATION_TEMPLATE::START
<#list flat_task as task_name, task>
STaskIteration g_astTaskIteration_${task_name}[] = {
	<#list task.iterationCountList as mode_id, count_value>
	{
		${mode_id}, // Mode ID
		<#if count_value == 0>1<#else>${count_value?c}</#if>, // iteration count
	},
	</#list>	
};

</#list>
// ##TASK_ITERATION_TEMPLATE::END


// ##TASK_LIST_TEMPLATE::START
<#list task_graph as graph_name, task_graph>
STask g_astTasks_${task_graph.name}[] = {
	<#list task_graph.taskList as task>
	{ 	${task.id}, // Task ID
		"${task.name}", // Task name
		TASK_TYPE_${task.type}, // Task Type
		g_ast_${task.name}_functions, // Task function array
		g_ast_${task.name}_thread_context, // Task thread context
		${task.taskFuncNum}, // Task function array number
		RUN_CONDITION_${task.runCondition}, // Run condition
		${task.period?c}, // Period
		TIME_METRIC_${task.periodMetric}, // Period metric
		<#if task.childTaskGraphName??>&g_stGraph_${task.childTaskGraphName}<#else>(STaskGraph *) NULL</#if>, // Subgraph
		&g_stGraph_${task.parentTaskGraphName}, // Parent task graph
		<#if (task.taskParamList?size > 0)>g_astTaskParameter_${task.name}<#else>(STaskParameter *) NULL</#if>, // Task parameter information
		${task.taskParamList?size}, // Task parameter number
		<#if task.staticScheduled == true>TRUE<#else>FALSE</#if>, // Statically scheduled or not
		0,	  // Throughput constraint
		(HThreadMutex) NULL, // Mutex
		(HThreadEvent) NULL, // Conditional variable
		g_astTaskIteration_${task.name}, // Task iteration count (only used when the parent task graph is data flow)
		0, // current run count in iteration
		0, // current iteration
		0, // target iteration (this variable is used for calling delayed stop task)
		<#if task.type == "CONTROL">g_stTimer_${task.name}<#else>(STimer *) NULL</#if>, // Timer slot (used by control task)
	},
	</#list>
};

</#list>

// ##TASK_LIST_TEMPLATE::END


SModelControllerFunctionSet g_stDynamicModeTransitionFunctions = {
	UKModeTransitionMachineController_HandleModelGeneral,
	UKModeTransitionMachineController_GetTaskIterationIndex,
	UKModeTransitionMachineController_Clear,
	UKModeTransitionMachineController_ChangeSubGraphTaskState,
	UKModeTransitionMachineController_HandleModelGeneralDuringStopping,
};

SModelControllerFunctionSet g_stStaticModeTransitionFunctions = {
	UKModeTransitionMachineController_HandleModelComposite,
	UKModeTransitionMachineController_GetTaskIterationIndex,
	UKModeTransitionMachineController_Clear,
	UKModeTransitionMachineController_ChangeTaskThreadState,
	(FnHandleModel) NULL,
};

SModelControllerFunctionSet g_stDynamicConvergentLoopFunctions = {
	(FnHandleModel ) UKLoopModelController_HandleConvergentLoop,
	(FnGetTaskIterationIndex) NULL,
	(FnControllerClear) NULL,
	(FnChangeTaskThreadState) NULL,
	(FnHandleModel) NULL,
};

SModelControllerFunctionSet g_stDynamicDataLoopFunctions = {
	(FnHandleModel ) NULL,
	(FnGetTaskIterationIndex) NULL,
	(FnControllerClear) NULL,
	(FnChangeTaskThreadState) NULL,
	(FnHandleModel) NULL,
};

SModelControllerFunctionSet g_stStaticConvergentLoopFunctions = {
	(FnHandleModel ) NULL,
	(FnGetTaskIterationIndex) NULL,
	(FnControllerClear) NULL,
	(FnChangeTaskThreadState) NULL,
	(FnHandleModel) NULL,
};

SModelControllerFunctionSet g_stStaticDataLoopFunctions = {
	(FnHandleModel ) NULL,
	(FnGetTaskIterationIndex) NULL,
	(FnControllerClear) NULL,
	(FnChangeTaskThreadState) NULL,
	(FnHandleModel) NULL,
};


// ##TASK_GRAPH_TEMPLATE::START
<#list task_graph as graph_name, task_graph_element>
	<#switch task_graph_element.controllerType>
		<#case "VOID">
			<#break>
		<#case "CONTROL_TASK_INCLUDED">
SModelControllerCommon g_stController_${task_graph_element.name} = {
	(HThreadMutex) NULL,
	0,
	(SModelControllerFunctionSet *) NULL,
};

			<#break>
		<#case "DYNAMIC_MODE_TRANSITION">
SModeTransitionController g_stController_${task_graph_element.name} = {
	{
		(HThreadMutex) NULL,
		0,
		&g_stDynamicModeTransitionFunctions,
	},
	<#if flat_task[task_graph_element.name].modeTransition??>&g_stModeTransition_${task_graph_element.name}<#else>(SModeTransitionMachine *) NULL</#if>, // MTM information
};

			<#break>
		<#case "STATIC_MODE_TRANSITION">
SModeTransitionController g_stController_${task_graph_element.name} = {
	{
		(HThreadMutex) NULL,
		0,
		&g_stStaticModeTransitionFunctions,
	},
	<#if flat_task[task_graph_element.name].modeTransition??>&g_stModeTransition_${task_graph_element.name}<#else>(SModeTransitionMachine *) NULL</#if>, // MTM information
};

			<#break>
		<#case "STATIC_CONVERGENT_LOOP">
// Static convergent loop not implemented
			<#break>
		<#case "DYNAMIC_CONVERGENT_LOOP">
SLoopController g_stController_${task_graph_element.name} = {
	{
		(HThreadMutex) NULL,
		0,
		&g_stDynamicConvergentLoopFunctions,
	},
	<#if flat_task[task_graph_element.name].loopStruct??>&g_stLoopStruct_${task_graph_element.name}<#else>(SLoopInfo *) NULL</#if>, // Loop information
};

			<#break>
		<#case "STATIC_DATA_LOOP">
// Static data loop not implemented 
			<#break>
		<#case "DYNAMIC_DATA_LOOP">
SLoopController g_stController_${task_graph_element.name} = {
	{
		(HThreadMutex) NULL,
		0,
		&g_stDynamicDataLoopFunctions,
	},
	<#if flat_task[task_graph_element.name].loopStruct??>&g_stLoopStruct_${task_graph_element.name}<#else>(SLoopInfo *) NULL</#if>, // Loop information
};

			<#break>
	</#switch>
STaskGraph g_stGraph_${task_graph_element.name} = {
		GRAPH_TYPE_${task_graph_element.taskGraphType}, // Task graph type
		CONTROLLER_TYPE_${task_graph_element.controllerType}, // graph controller type
		<#if task_graph_element.controllerType == "VOID">(void *) NULL<#else>&g_stController_${task_graph_element.name}</#if>, // task graph controller (SLoopController, SModeTransitionController, or SModelControllerCommon)
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


// ##PROCESSOR_INFO_TEMPLATE::START
SProcessor g_astProcessorInfo[] = {

<#list device_info as device_name, device>
	<#list device.processorList as processor>
	{ 	${processor.id}, // Processor ID
		<#if processor.isCPU == true>TRUE<#else>FALSE</#if>, // Processor is CPU?			
		"${processor.name}", // Processor name
		${processor.poolSize}, // Processor pool size
	},
	</#list>
</#list>
};
// ##PROCESSOR_INFO_TEMPLATE::END


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
	{	<#if mapped_schedule.parentTaskId == -1>(STaskGraph *) &g_stGraph_top<#else>&g_stGraph_${task_name}</#if>, // Parent Task Structure
		${compositeMappedProcessor.modeId}, // Mode transition mode ID
		g_astScheduleList_${mapped_schedule.parentTaskName}_${compositeMappedProcessor.modeId}_${compositeMappedProcessor.processorId}_${compositeMappedProcessor.processorLocalId}, // schedule list per throughput constraint
		${compositeMappedProcessor.compositeTaskScheduleList?size}, // The number of schedules in the schedule list
		0, // Schedule Index (Default to set 0)
		${compositeMappedProcessor.sequenceIdInMode}, // Mode Sequence ID 
	},
	</#list>
</#list>
};

SGenericMapProcessor g_stCPUProcessor = {
	UKHostSystem_MapCPU,
};

<#if gpu_used == true>
SGenericMapProcessor g_stGPUProcessor = {
	UKGPUSystem_MapGPU,
};
</#if>

// ##MAPPING_SCHEDULING_INFO_TEMPLATE::START

SMappedGeneralTaskInfo g_astGeneralTaskMappingInfo[] = {
<#list mapping_info as task_name, mapped_task>
	<#list mapped_task.mappedProcessorList as mappedProcessor>
	{	TASK_TYPE_${mapped_task.mappedTaskType}, // Task type
		&g_astTasks_${mapped_task.parentTaskGraphName}[${mapped_task.inGraphIndex}], // Task ID or composite task information
		${mappedProcessor.processorId}, // Processor ID
		${mappedProcessor.processorLocalId}, // Processor local ID
		<#list device_info as device_name, device>
			<#if device_name == mapped_task.mappedDeviceName>
				<#list device.processorList as processor>
					<#if processor.id == mappedProcessor.processorId>
						<#if processor.isCPU == true>
		&g_stCPUProcessor, // CPU Processor API
						<#else>
		&g_stGPUProcessor, // GPU Processor API
						</#if>
					</#if>
				</#list>
			</#if>
		</#list>
	},
	</#list>
</#list>
};


SMappedCompositeTaskInfo g_astCompositeTaskMappingInfo[] = {
<#list schedule_info as task_name, scheduled_task>
	<#list scheduled_task.mappedProcessorList as scheduledProcessor>
	{
		&g_astScheduledTaskList[${scheduledProcessor.inArrayIndex}],
		${scheduledProcessor.processorId}, // Processor ID
		${scheduledProcessor.processorLocalId}, // Processor local ID		
	},
	</#list>
</#list>
};


SMappedTaskInfo g_stMappingInfo = {
	<#if (mapping_info?size > 0)>g_astGeneralTaskMappingInfo<#else>(SMappedGeneralTaskInfo *)NULL</#if>, // general task array
	<#if (mapping_info?size > 0)>ARRAYLEN(g_astGeneralTaskMappingInfo)<#else>0</#if>, // size of general task array
	<#if (schedule_info?size > 0)>g_astCompositeTaskMappingInfo<#else>(SMappedCompositeTaskInfo *) NULL</#if>, // composite task array
	<#if (schedule_info?size > 0)>ARRAYLEN(g_astCompositeTaskMappingInfo)<#else>0</#if>, // size of composite task array
};

// ##MAPPING_SCHEDULING_INFO_TEMPLATE::END


int g_nNumOfTasks_top = ARRAYLEN(g_astTasks_top);
int g_nTaskIdToTaskNum = ARRAYLEN(g_astTaskIdToTask);
int g_nProcessorInfoNum = ARRAYLEN(g_astProcessorInfo);
int g_nLibraryInfoNum = <#if (library_info?size > 0)>ARRAYLEN(g_stLibraryInfo)<#else>0</#if>;
int g_nTimerSlotNum = MAX_TIMER_SLOT_SIZE;
int g_nDeviceId = ${device_id};

