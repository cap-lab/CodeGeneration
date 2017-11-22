package org.snu.cse.cap.translator.structure.mapping;

import org.snu.cse.cap.translator.structure.task.TaskShapeType;

public abstract class MappingInfo {
	protected TaskShapeType mappedTaskType;
	protected int processorId;
	protected int processorLocalId;
	
	public TaskShapeType getMappedTaskType() {
		return mappedTaskType;
	}
	
	public void setMappedTaskType(TaskShapeType mappedTaskType) {
		this.mappedTaskType = mappedTaskType;
	}
	
	public int getProcessorId() {
		return processorId;
	}
	
	public void setProcessorId(int processorId) {
		this.processorId = processorId;
	}
	
	public int getProcessorLocalId() {
		return processorLocalId;
	}
	
	public void setProcessorLocalId(int processorLocalId) {
		this.processorLocalId = processorLocalId;
	}
}
