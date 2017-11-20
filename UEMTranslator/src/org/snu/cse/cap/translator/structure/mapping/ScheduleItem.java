package org.snu.cse.cap.translator.structure.mapping;

public class ScheduleItem {
	private String taskName;
	private int goFuncId;
	private int repetition;
	
	public String getTaskName() {
		return taskName;
	}
	
	public void setTaskName(String taskName) {
		this.taskName = taskName;
	}
	
	public int getGoFuncId() {
		return goFuncId;
	}
	
	public void setGoFuncId(int goFuncId) {
		this.goFuncId = goFuncId;
	}
	
	public int getRepetition() {
		return repetition;
	}
	
	public void setRepetition(int repetition) {
		this.repetition = repetition;
	}
}
