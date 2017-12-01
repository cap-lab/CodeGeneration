package org.snu.cse.cap.translator.structure.mapping;

import java.util.ArrayList;

import org.snu.cse.cap.translator.structure.task.TaskShapeType;

public class MappingInfo {
	protected String mappedDeviceName;
	protected TaskShapeType mappedTaskType; 
	protected ArrayList<MappedProcessor> mappedProcessorList;
	
	public MappingInfo(TaskShapeType mappedTaskType) {
		this.mappedTaskType = mappedTaskType;
		this.mappedProcessorList = new ArrayList<MappedProcessor>();
	}
	
	public void putProcessor(MappedProcessor proc) {
		this.mappedProcessorList.add(proc);
	}	

	public TaskShapeType getMappedTaskType() {
		return mappedTaskType;
	}
	
	public void setMappedTaskType(TaskShapeType mappedTaskType) {
		this.mappedTaskType = mappedTaskType;
	}

	public String getMappedDeviceName() {
		return mappedDeviceName;
	}

	public void setMappedDeviceName(String mappedDeviceName) {
		this.mappedDeviceName = mappedDeviceName;
	}

	public ArrayList<MappedProcessor> getMappedProcessorList() {
		return mappedProcessorList;
	}
}
