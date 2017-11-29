package org.snu.cse.cap.translator.structure.mapping;

import java.util.ArrayList;

public class CompositeTaskMappingInfo extends MappingInfo {
	private String parentTaskName;
	private ArrayList<CompositeTaskSchedule> scheduleLists;
	private int modeId;
	
	public CompositeTaskMappingInfo(String taskName, int modeId) {
		this.parentTaskName = taskName;
		this.modeId = modeId;
		this.scheduleLists = new ArrayList<CompositeTaskSchedule>();
	}
	
	public void putCompositeTaskSchedule(CompositeTaskSchedule schedule) 
	{
		this.scheduleLists.add(schedule);
	}
	
	public String getParentTaskName() {
		return parentTaskName;
	}
	
	public int getModeId() {
		return modeId;
	}
	
	public void setParentTaskName(String parentTaskName) {
		this.parentTaskName = parentTaskName;
	}
	
	public void setModeId(int modeId) {
		this.modeId = modeId;
	}
}
