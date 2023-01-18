package org.snu.cse.cap.translator.structure.mapping;

import org.snu.cse.cap.translator.structure.task.TaskShapeType;

public class CompositeTaskMappingInfo extends MappingInfo {
	private String parentTaskName;
	private int parentTaskId;
	
	public CompositeTaskMappingInfo(String taskName, int taskId, int priority) {
		super(TaskShapeType.COMPOSITE, priority);
		this.parentTaskName = taskName;
		this.parentTaskId = taskId;
	}
	
	public CompositeTaskMappedProcessor getMappedProcessorInfo(int modeId, int procId, int localId)
	{
		CompositeTaskMappedProcessor compositeMappedProcessor = null;
		boolean found = false;
		
		for(MappedProcessor mappedProcessor: this.getMappedProcessorList())
		{
			compositeMappedProcessor = (CompositeTaskMappedProcessor) mappedProcessor;
			
			if(compositeMappedProcessor.getModeId() == modeId && 
				compositeMappedProcessor.getProcessorId() == procId && 
				compositeMappedProcessor.getProcessorLocalId() == localId)
			{
				found = true;
				break;
			}
		}
		
		if(found == false)
		{
			compositeMappedProcessor = null;
		}
		
		return compositeMappedProcessor;
	}
	
	public String getParentTaskName() {
		return parentTaskName;
	}
		
	public void setParentTaskName(String parentTaskName) {
		this.parentTaskName = parentTaskName;
	}


	public int getParentTaskId() {
		return parentTaskId;
	}


	public void setParentTaskId(int parentTaskId) {
		this.parentTaskId = parentTaskId;
	}
}
