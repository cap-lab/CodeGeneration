/* This file is generated by UEM Translator */

#ifndef SRC_KERNEL_GENERATED_MANUAL_H_
#define SRC_KERNEL_GENERATED_MANUAL_H_

<#macro printFillColor task comma>
<@compress single_line=true>
  <#assign comma_val="">
  <#if comma==true>
    <#assign comma_val=",">
  </#if>
  <#switch task.type>
  	<#case "COMPUTATIONAL">
  	<#case "COMPOSITE">
  	  <#if task.childTaskGraphName??>
  	    <#if task_graph[task.childTaskGraphName].taskGraphType == "PROCESS_NETWORK">
${comma_val}fillcolor="#f0f0f0"
  	    <#else>
  	      <#if task.modeTransition??>
${comma_val}fillcolor="#ffcccc"
          <#else>
${comma_val}fillcolor="#ffffff"
          </#if>
  	    </#if>
  	  </#if>
  	  <#break>
  	<#case "CONTROL">
${comma_val}fillcolor="#eaffcc"
  	  <#break>
  	<#case "LOOP">
  	  <#if task.loopStruct.loopType == "CONVERGENT">
${comma_val}fillcolor="#ffe6cc"
  	  <#else>
${comma_val}fillcolor="#cce5ff"
  	  </#if>
  	  <#break>
  </#switch>
</@compress>
</#macro>

<#macro printSingleTaskNode task space>
${space}${task.name} [label="${task.shortName}<#if device_info.taskMap[task.name]??>*</#if> (${task.id})"<@printFillColor task true/><#if device_info.taskMap[task.name]??>,peripheries=2,penwidth=2</#if>];
</#macro>

<#macro printRecursiveNode task space>
<#assign innerspace="${space}  " />
${space}subgraph cluster_${task.name} {
${space}  label = "${task.shortName}";
${space}  style=filled;
${space}  color=black;
${space}  <@printFillColor task false/>;
  <#list task_graph[task.name].taskList as child_task>
    <#if child_task.childTaskGraphName??>
      <@printRecursiveNode child_task innerspace />
    <#else>
${space}  <@printSingleTaskNode child_task innerspace />
    </#if>
  </#list>
${space}}

</#macro>
/*!
 \mainpage UEM-generated Software Manual (Mapped Device: ${device_info.name})
  
  ## Application Task Graph

  @dot
digraph application_task_graph  {
  node [shape=box, style=filled, fillcolor="#ffffcc"];
<#list task_graph["top"].taskList as task>
  <#if task.childTaskGraphName??>
    <@printRecursiveNode task "  " />
  <#else>
    <@printSingleTaskNode task "  " />
  </#if>
</#list>
<#list library_map as library_name, library>
  ${library_name}[labal="${library_name}",fillcolor="#ccccff"];
</#list>
  
<#list channel_list as channel>
  ${channel.outputPort.mostInnerPort.taskName} -> ${channel.inputPort.mostInnerPort.taskName} [label=" ${channel.index}"<@compress single_line=true>
  <#if channel.communicationType != "SHARED_MEMORY">
    ,color="#78136f",penwidth=2
  </#if>];</@compress>
</#list>

<#list flat_task as task_name, task>
  <#list task.masterPortToLibraryMap as port_name, library>
  ${task.name} -> ${library.name} [dir=back,arrowtail=rbox,penwidth=2];
  </#list>
</#list>
}
  @enddot

### Task List  

| ID | Task  | Type | Run condition | Language | Source file | Mapped |
| -- | ----- | ---- | ------------- | -------- | ----------- | ------ |
<#list flat_task as task_name, task>
<@compress single_line=true>| ${task.id} |
 <#if device_info.taskMap[task.name]??>\ref doxygenpageTask_${task.name}<#else>${task.name}</#if> |
 ${task.type} |
 ${task.runCondition}<#if task.runCondition == "TIME_DRIVEN"> (${task.period} ${task.periodMetric?lower_case})</#if> |
 <#if task.childTaskGraphName??>-<#else>${task.language}</#if> |
 <#if task.childTaskGraphName??>-<#else>\ref ${task.taskCodeFile}</#if> |
 <#if device_info.taskMap[task.name]??>O<#else>X</#if> |
</@compress>

</#list>

<#if (library_map?size > 0) >
### Library Task List

| Library task | Language | Header file | Source file | Mapped |
| ------------ | -------- | ----------- | ----------- | ------ |
<#list library_map as library_name, library>
<@compress single_line=true>
| <#if device_info.libraryMap[library.name]??>\ref doxygenpageLibrary_${library.name}<#else>${library.name}</#if> |
 ${library.language} | ${library.header} | ${library.file} | <#if device_info.libraryMap[library.name]??>O<#else>X</#if> |
</@compress>

</#list>

</#if>

<#if (channel_list?size > 0) >
### Channel List

| ID | Communication | Channel Size | Initial data size |
| -- | ------------- | ------------ | ----------------- |
<#list channel_list as channel>
| ${channel.index} | <#if channel.communicationType == "SHARED_MEMORY">Local<#else>Remote</#if> | ${channel.size} | ${channel.initialDataLen} |
</#list>

</#if>

### Device List

| Name | Platform | Architecture | Mapped |
| ---- | -------- | ------------ | ------ |
<#list device_map as device_name, device>
| ${device.name} | ${device.platform} | ${device.architecture} | <#if device == device_info>O<#else>X</#if> |
</#list>


<#if (device_connection_map?size > 0)>
### Connection List

| Type | Role | Connection information | Device |
| ---- | ---- | ---------------------- | ------ |
<#list device_connection_map as master_device_name, device_connection>
    <#list device_connection.connectionToSlaveMap as master_name, master_to_slave_connection>
      <@compress single_line=true>| ${master_to_slave_connection.master.network} / ${master_to_slave_connection.master.protocol} | ${master_to_slave_connection.master.role} |
        <#if master_to_slave_connection.master.protocol == "TCP" || master_to_slave_connection.master.protocol == "SECURE_TCP">
        :${master_to_slave_connection.master.port?c}
        <#else>
        	<#if master_to_slave_connection.master.portAddress??>
        	${master_to_slave_connection.master.portAddress}
        	<#else>
        	Board TX: ${master_to_slave_connection.master.boardTXPinNumber}, Board RX: ${master_to_slave_connection.master.boardRXPinNumber}
        	</#if>
        </#if>
        | ${master_device_name} |
      </@compress>
      
      <#list master_to_slave_connection.slaveDeviceToConnectionMap as slave_device_name, slave_connection_list>
        <#list slave_connection_list as slave_connection>
          <@compress single_line=true>| ${slave_connection.network} / ${slave_connection.protocol} | ${slave_connection.role} |
            <#if slave_connection.protocol == "TCP" || slave_connection.protocol == "SECURE_TCP">
            ${slave_connection.IP}:${slave_connection.port?c}
            <#else>
            	<#if slave_connection.portAddress??>
            	${slave_connection.portAddress}
            	<#else>
            	Board TX: ${slave_connection.boardTXPinNumber}, Board RX: ${slave_connection.boardRXPinNumber}
            	</#if>
            </#if>
            | ${slave_device_name} |
          </@compress>
        </#list>
      </#list>
    </#list>
</#list>

</#if>

*/

<#macro printPortSampleRate port>
<@compress single_line=true>
  <#switch port.portSampleRateType>
    <#case "FIXED">
      Fixed: ${port.portSampleRateList[0].sampleRate}
      <#break>
    <#case "VARIABLE">
      Variable
      <#break>
    <#case "MULTIPLE">
      Multiple: <#list port.portSampleRateList as sampleRateType>
        <#if (sampleRateType?index > 0)>/</#if>${sampleRateType.sampleRate}
      </#list>
      <#break>
  </#switch>
</@compress>
</#macro>

<#list device_info.taskMap as task_name, task>
  <#assign input_port_num=0 />
  <#assign output_port_num=0 />
  <#list device_info.portList as port>
    <#if port.taskId == task.id>
      <#if port.direction == "input">
        <#assign input_port_num=input_port_num+1 />
      <#elseif port.direction == "output">
        <#assign output_port_num=output_port_num+1 />
      <#else>
        merong
      </#if>    
    </#if>
  </#list>
/*!
 \page doxygenpageTask_${task.name} ${task.name}
 
  @dot
digraph ${task.name}_task_graph {
  ${task.name} [
   shape=plaintext
   label=<
     <table border="0" cellborder="1" cellspacing="0">
       <tr><td bgcolor="#ffffcc" <#if (output_port_num + input_port_num > 0)>colspan="${output_port_num + input_port_num}"</#if>>${task.name}</td></tr>
       <#if (output_port_num + input_port_num > 0)>
       <tr>
         <#if (input_port_num > 0)><td bgcolor="#cbff2f" colspan="${input_port_num}">input port</td></#if><#if (output_port_num > 0)><td bgcolor="#ff8282" colspan="${output_port_num}">output port</td></#if></tr>
       <tr>
  <#list device_info.portList as port>
    <#if port.taskId == task.id>
      <#if port.direction == "input">
         <td bgcolor="#cbff2f" port='${port.portName}_${port?index}'>${port.portName}</td>
      </#if>    
    </#if>
  </#list>
  <#list device_info.portList as port>
    <#if port.taskId == task.id>
      <#if port.direction == "output">
         <td bgcolor="#ff8282" port='${port.portName}_${port?index}'>${port.portName}</td>
      </#if>    
    </#if>
  </#list>
       </tr>
       </#if>
     </table>
  >];
  <#list device_info.portList as port>
    <#if port.taskId == task.id>
      <#if port.direction == "input" || port.direction == "output">
  ${port.portName}_sample [shape=point,width=0.01,height=0.01];
      </#if>    
    </#if>
  </#list>
  
  <#list device_info.portList as port>
    <#if port.taskId == task.id>
      <#if port.direction == "input">
  ${task.name}:${port.portName}_${port?index} -> ${port.portName}_sample [dir=back,label=" <@printPortSampleRate port />"];
      <#elseif port.direction == "output">
  ${task.name}:${port.portName}_${port?index} -> ${port.portName}_sample [label=" <@printPortSampleRate port />"];
      </#if>    
    </#if>    
  </#list>
}
  @enddot

  ${task.description}
  
### Task Basic Information
  - ID: ${task.id}
  - Type: ${task.type}
  - Run Condition: ${task.runCondition} <#if task.runCondition == "TIME_DRIVEN"> (${task.period} ${task.periodMetric?lower_case})</#if>
  <#if !task.childTaskGraphName??>- Language: ${task.language}</#if>
  <#if !task.childTaskGraphName??>- Source file: ${task.taskCodeFile}</#if>

  <#if (input_port_num > 0)>
### Input Port List

| Name | Type | Sample size | Sample rate | Description |
| ---- | ---- | ----------- | ----------- | ----------- |
    <#list device_info.portList as port>
      <#if port.taskId == task.id && port.direction == "input">
| ${port.portName} | ${port.portType} | ${port.sampleSize} | <@printPortSampleRate port /> | ${port.description} |
      </#if>
    </#list>

  </#if>
  <#if (output_port_num > 0)>
### Output Port List

| Name | Type | Sample size | Sample rate | Description |
| ---- | ---- | ----------- | ----------- | ----------- |
    <#list device_info.portList as port>
      <#if port.taskId == task.id && port.direction == "output">
| ${port.portName} | ${port.portType} | ${port.sampleSize} | <@printPortSampleRate port /> | ${port.description} |
      </#if>
    </#list>

  </#if>
  <#if (task.extraHeaderSet?size + task.extraSourceSet?size > 0)>
### Extra File List

| Type | File name |
| ---- | --------- |
    <#list task.extraHeaderSet as header_file>
| Header | ${header_file} |
    </#list>
    <#list task.extraSourceSet as source_file>
| Source | ${source_file} |
    </#list>

  </#if>
  <#if (task.taskParamList?size > 0)>
### Parameter List

| Name | Type | Default value | Description |
| ---- | ---- | ------------- | ----------- |
    <#list task.taskParamList as param>
| ${param.name} | ${param.type} | ${param.value} | ${param.description} |
    </#list>

  </#if>
*/

</#list>

<#list device_info.libraryMap as library_name, library>
/*!
  \page doxygenpageLibrary_${library.name} ${library.name}
  
  ${library.description}
  
  <#list library.functionList as function>
<table style="width:100%">
<tr><td class=memproto style="font-size:20px;font-weight:bold"> <span style="color:blue;"> ${function.returnType}</span> ${function.name} (
	<#list function.argumentList as argument><#if (argument?index > 0)>, </#if><span style="color:blue">${argument.type}</span> ${argument.name}
	</#list>) </td></tr>
<tr><td class=memdoc>
${function.description}
    <#if (function.argumentList?size > 0)>
### Parameters
      <#list function.argumentList as argument>
  - <span class=paramname style="font-weight:bold">${argument.name}</span> ${argument.description} <br>
      </#list>
    </#if>
</td></tr></table>
  </#list>
 
*/

</#list>

#endif /* SRC_KERNEL_GENERATED_UEM_MANUAL_H_ */

