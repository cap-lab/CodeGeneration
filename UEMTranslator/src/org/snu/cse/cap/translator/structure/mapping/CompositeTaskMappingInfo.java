package org.snu.cse.cap.translator.structure.mapping;

public class CompositeTaskMappingInfo extends MappingInfo {
	private String parentTaskName;
	private int modeId;
	
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
