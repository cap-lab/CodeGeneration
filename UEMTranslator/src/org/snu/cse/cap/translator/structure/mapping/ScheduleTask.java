package org.snu.cse.cap.translator.structure.mapping;

public class ScheduleTask extends ScheduleItem {
	private String taskName;
	
	public ScheduleTask(String taskName, int repetition) {
		super(repetition);
		this.taskName = taskName;
	}

	public String getTaskName() {
		return taskName;
	}

	public void setTaskName(String taskName) {
		this.taskName = taskName;
	}

}
