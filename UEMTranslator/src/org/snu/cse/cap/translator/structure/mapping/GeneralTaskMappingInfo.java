package org.snu.cse.cap.translator.structure.mapping;

import java.util.ArrayList;

import org.snu.cse.cap.translator.structure.task.TaskShapeType;

public class GeneralTaskMappingInfo extends MappingInfo {
	private String taskName;
	private int inGraphIndex;
	private String parentTaskGraphName;
	private ArrayList<GeneralMappedProcessor> mappedDefaultProcessorList;
	
	public GeneralTaskMappingInfo(String taskName, TaskShapeType mappedTaskType, String parentTaskGraphName,
			int inGraphIndex, int priority) {
		super(mappedTaskType, priority);
		this.taskName = taskName;
		this.inGraphIndex = inGraphIndex;
		this.parentTaskGraphName = parentTaskGraphName;
		this.mappedDefaultProcessorList = new ArrayList<GeneralMappedProcessor>();
	}

	public void putDefaultGeneralMappedProcessor(GeneralMappedProcessor proc) {
		this.mappedDefaultProcessorList.add(proc);
	}

	public String getTaskName() {
		return taskName;
	}

	public void setTaskName(String taskName) {
		this.taskName = taskName;
	}

	public int getInGraphIndex() {
		return inGraphIndex;
	}

	public String getParentTaskGraphName() {
		return parentTaskGraphName;
	}

	public void setInGraphIndex(int inGraphIndex) {
		this.inGraphIndex = inGraphIndex;
	}

	public void setParentTaskGraphName(String parentTaskName) {
		this.parentTaskGraphName = parentTaskName;
	}

	public ArrayList<GeneralMappedProcessor> getDefaultGeneralMappedProcessorList() {
		return mappedDefaultProcessorList;
	}
}
