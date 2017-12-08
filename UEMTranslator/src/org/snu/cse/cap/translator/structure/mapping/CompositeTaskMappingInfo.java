package org.snu.cse.cap.translator.structure.mapping;

import org.snu.cse.cap.translator.structure.task.TaskShapeType;

public class CompositeTaskMappingInfo extends MappingInfo {
	private String parentTaskName;
	private int parentTaskId;
	
	public CompositeTaskMappingInfo(String taskName, int taskId) {
		super(TaskShapeType.COMPOSITE);
		this.parentTaskName = taskName;
		this.parentTaskId = taskId;
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
