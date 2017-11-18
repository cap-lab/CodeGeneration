package org.snu.cse.cap.translator.structure;

public abstract class MappingInfo {
	protected TaskType mappedTaskType;
	protected int processorId;
	protected int processorLocalId;
	
	public TaskType getMappedTaskType() {
		return mappedTaskType;
	}
	
	public void setMappedTaskType(TaskType mappedTaskType) {
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
