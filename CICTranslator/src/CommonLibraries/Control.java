package CommonLibraries;

import java.util.*;

import hopes.cic.xml.*;
import InnerDataStructures.*;

public class Control {

	public static String generateControlParamListEntries(CICControlType mControl, Map<String, Task> mTask){
		String controlParamListCode="";
		
		if(mControl.getControlTasks() != null){
			controlParamListCode += "CIC_STATIC CIC_UT_PARAM param_list[] = {";
			for(Task task: mTask.values()){
				int index = 0;
				for(TaskParameterType parameter: task.getParameter()){
					//if(parameter.getValue().startsWith("\""))
					controlParamListCode += "\t{" + task.getIndex() + ", " + index + ", \"" + task.getName()
					+ "\", \"" + parameter.getName() + "\", (CIC_T_VOID*)" + parameter.getValue() + "},\n";
					index++;
				}
			}
			controlParamListCode += "};";
		}
		
		return controlParamListCode;
	}
	
	public static String generateControlChannelListEntries(CICControlType mControl, Map<String, Task> mTask){
		String controlChannelListCode="";
		
		int groupIndex = 0;
		if(mControl.getControlTasks() != null){
			controlChannelListCode += "CIC_STATIC CIC_UT_CONTROL_CHANNEL control_channel[] = {";
			for(ControlTaskType controlTask: mControl.getControlTasks().getControlTask()){
				if(mTask.containsKey(controlTask.getTask())){
					
					String slaveProcId = null;
					String masterProcId = null;
					String taskID = null;
					int groupFlag = 0;
					
					for(Task task: mTask.values()){
						if(task.getName().equals(controlTask.getTask())){
							taskID = task.getIndex();
							masterProcId = task.getProc().get(0).toString();
							break;
						}
					}
					for(Task task: mTask.values()){
						if(task.getHasSubgraph().equalsIgnoreCase("No")){
							if(task.getName().equals(controlTask.getSlaveTask().get(0))){
								slaveProcId = task.getProc().get(0).toString();
								break;
							} 
						}
					}
					
					int index = 0;
					int groupId = 0;
					
					for(ExclusiveControlTasksType exclusive: mControl.getExclusiveControlTasksList().getExclusiveControlTasks()){
						for(String conTask: exclusive.getControlTask()){
							if(controlTask.getTask().equals(conTask)){
								groupId = index;
								groupFlag = 1;
								break;
							}
						}
						index++;
						if(groupFlag == 0){
							groupId = groupIndex;
							groupIndex++;
						}
					}
					
					controlChannelListCode += "\t{" + taskID + ", " + Integer.toString(controlTask.getPriority().intValue()) 
												+ ", " + groupId + ", 0, 0, {0, }, {0, } },\n"; 
				}
			}
			controlChannelListCode += "};";
		}
			
		return controlChannelListCode;
	}

}
