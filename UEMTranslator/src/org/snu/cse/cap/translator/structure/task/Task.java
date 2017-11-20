package org.snu.cse.cap.translator.structure.task;

enum TaskType {
	COMPUTATIONAL,
	CONTROL,
	LOOP,
	COMPOSITE,
}

enum TimeMetric {
	CYCLE,
	COUNT, 
	MICROSEC,
	MILLISEC,
	SEC,
	MINUTE,
	HOUR,
}

enum TaskRunCondition {
	DATA_DRIVEN,
	TIME_DRIVEN,
	CONTROL_DRIVEN,
}

public class Task {
	private int taskId;
	private String taskName;
	private TaskType taskType;
	private int taskFuncNum;
	private int runRate;
	private int period;
	private TimeMetric periodMetric;
	private String parentTaskGraphName;
	private int inGraphIndex;
	private String childTaskGraphName;
	private TaskModeTransition modeTransition;
	private TaskLoop loop;
	private TaskParameter taskParam;
	private boolean staticScheduled;
	private int mappedTaskNum;
	private TaskRunCondition runCondition;
	
	public int getTaskId() {
		return taskId;
	}
	
	public void setTaskId(int taskId) {
		this.taskId = taskId;
	}
	
	public String getTaskName() {
		return taskName;
	}
	
	public void setTaskName(String taskName) {
		this.taskName = taskName;
	}
	
	public TaskType getTaskType() {
		return taskType;
	}
	
	public void setTaskType(TaskType taskType) {
		this.taskType = taskType;
	}
	
	public int getTaskFuncNum() {
		return taskFuncNum;
	}
	
	public void setTaskFuncNum(int taskFuncNum) {
		this.taskFuncNum = taskFuncNum;
	}
	
	public int getRunRate() {
		return runRate;
	}
	
	public void setRunRate(int runRate) {
		this.runRate = runRate;
	}
	
	public int getPeriod() {
		return period;
	}
	
	public void setPeriod(int period) {
		this.period = period;
	}
	
	public TimeMetric getPeriodMetric() {
		return periodMetric;
	}
	
	public void setPeriodMetric(TimeMetric periodMetric) {
		this.periodMetric = periodMetric;
	}
	
	public String getParentTaskGraphName() {
		return parentTaskGraphName;
	}
	
	public void setParentTaskGraphName(String parentTaskGraphName) {
		this.parentTaskGraphName = parentTaskGraphName;
	}
	
	public int getInGraphIndex() {
		return inGraphIndex;
	}
	
	public void setInGraphIndex(int inGraphIndex) {
		this.inGraphIndex = inGraphIndex;
	}
	
	public String getChildTaskGraphName() {
		return childTaskGraphName;
	}
	
	public void setChildTaskGraphName(String childTaskGraphName) {
		this.childTaskGraphName = childTaskGraphName;
	}
	
	public TaskModeTransition getModeTransition() {
		return modeTransition;
	}
	
	public void setModeTransition(TaskModeTransition modeTransition) {
		this.modeTransition = modeTransition;
	}
	
	public TaskLoop getLoop() {
		return loop;
	}
	
	public void setLoop(TaskLoop loop) {
		this.loop = loop;
	}
	
	public TaskParameter getTaskParam() {
		return taskParam;
	}
	
	public void setTaskParam(TaskParameter taskParam) {
		this.taskParam = taskParam;
	}
	
	public boolean isStaticScheduled() {
		return staticScheduled;
	}
	
	public void setStaticScheduled(boolean staticScheduled) {
		this.staticScheduled = staticScheduled;
	}
	
	public int getMappedTaskNum() {
		return mappedTaskNum;
	}
	
	public void setMappedTaskNum(int mappedTaskNum) {
		this.mappedTaskNum = mappedTaskNum;
	}

	public TaskRunCondition getRunCondition() {
		return runCondition;
	}

	public void setRunCondition(TaskRunCondition runCondition) {
		this.runCondition = runCondition;
	}
}
