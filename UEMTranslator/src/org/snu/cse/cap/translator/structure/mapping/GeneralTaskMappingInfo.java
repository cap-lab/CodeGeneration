package org.snu.cse.cap.translator.structure.mapping;

import org.snu.cse.cap.translator.structure.task.TaskShapeType;

public class GeneralTaskMappingInfo extends MappingInfo {
	private String taskName;
	
	public GeneralTaskMappingInfo(String taskName, TaskShapeType mappedTaskType) {
		super(mappedTaskType);
		this.taskName = taskName;
	}

	public String getTaskName() {
		return taskName;
	}

	public void setTaskName(String taskName) {
		this.taskName = taskName;
	}
}
