package org.snu.cse.cap.translator.structure.mapping;

public class ScheduleTask extends ScheduleItem {
	private String taskName;
	private int taskFuncId;
	
	public ScheduleTask(String taskName, int repetition, int depth) {
		super(ScheduleItemType.TASK, repetition, depth);
		this.taskName = taskName;
	}

	public String getTaskName() {
		return taskName;
	}

	public void setTaskName(String taskName) {
		this.taskName = taskName;
	}

	public int getTaskFuncId() {
		return taskFuncId;
	}

	public void setTaskFuncId(int taskFuncId) {
		this.taskFuncId = taskFuncId;
	}
}
