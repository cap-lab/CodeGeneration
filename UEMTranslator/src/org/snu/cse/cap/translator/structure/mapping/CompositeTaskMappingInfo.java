package org.snu.cse.cap.translator.structure.mapping;

import org.snu.cse.cap.translator.structure.task.TaskShapeType;

public class CompositeTaskMappingInfo extends MappingInfo {
	private String parentTaskName;
	
	public CompositeTaskMappingInfo(String taskName) {
		super(TaskShapeType.COMPOSITE);
		this.parentTaskName = taskName;
	}

	
	public String getParentTaskName() {
		return parentTaskName;
	}
		
	public void setParentTaskName(String parentTaskName) {
		this.parentTaskName = parentTaskName;
	}
}
