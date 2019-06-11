// ##TIMER_CODE_TEMPLATE::START
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
// ##TIMER_CODE_TEMPLATE::END


// ##TASK_CODE_TEMPLATE::START
<#list flat_task as task_name, task>
	<#if !task.childTaskGraphName??>
		<#if task.language=="C" && gpu_used == false>
#ifdef __cplusplus
extern "C"
{
#endif 
		</#if>
		<#list 0..(task.taskFuncNum-1) as task_func_id>
void ${task.name}_Init${task_func_id}(int nTaskId);
void ${task.name}_Go${task_func_id}(int nTaskId);
void ${task.name}_Wrapup${task_func_id}();
		</#list>
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
		<#if library.language=="C" || library.isMasterLanguageC == true>
#ifdef __cplusplus
extern "C"
{
#endif 
		</#if>
void l_${libraryName}_init();
void l_${libraryName}_wrapup();
		<#if library.language=="C" || library.isMasterLanguageC == true>
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
		
		<#if device_constrained_info == "unconstrained">
${innerspace}{
${innerspace}	STask *pstTask = (STask *)NULL;
${innerspace}	uem_result result;
${innerspace}
${innerspace}	result = UKTask_GetTaskFromTaskId(nTaskId, &pstTask);
${innerspace}	if(result == ERR_UEM_NOERROR)
${innerspace}	{
${innerspace}		pstTask->astThreadContext[${scheduleItem.taskFuncId}].nCurRunIndex = pstTask->nCurIteration;
${innerspace}	}
${innerspace}}
		</#if>
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





