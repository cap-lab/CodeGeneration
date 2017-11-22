package org.snu.cse.cap.translator.structure.task;

import hopes.cic.xml.LoopStructureType;
import hopes.cic.xml.TaskType;

enum TimeMetric {
	CYCLE("cycle"),
	COUNT("count"), 
	MICROSEC("us"),
	MILLISEC("ms"),
	SEC("s"),
	MINUTE("m"),
	HOUR("h"),
	;
	
	private final String value;
	
	private TimeMetric(String value) {
		this.value = value;
	}
	
	@Override
	public String toString() {
		return value;
	}
}

enum TaskRunCondition {
	DATA_DRIVEN("data-driven"),
	TIME_DRIVEN("time-driven"),
	CONTROL_DRIVEN("control-driven"),
	;
	
	private final String value;
	
	private TaskRunCondition(String value) {
		this.value = value;
	}
	
	@Override
	public String toString() {
		return value;
	}
}

public class Task {
	private int id;
	private String name;
	private TaskShapeType type;
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
	private String taskCodeFile;
	
	public Task(int id, TaskType xmlTaskData)
	{
//		private int taskId;
//		private String taskName;
//		private TaskType taskType;
//		private int taskFuncNum; => later
//		private int runRate; => later
//		private int period; => later
//		private TimeMetric periodMetric; => later
//		private String parentTaskGraphName;
//		private int inGraphIndex; => later
//		private String childTaskGraphName; => later
//		private TaskModeTransition modeTransition; => later ????
//		private TaskLoop loop; => later ????
//		private TaskParameter taskParam; => later ?????
//		private boolean staticScheduled; => later
//		private int mappedTaskNum; => later
//		private TaskRunCondition runCondition; 
//		private String taskCodeFile;		
		
		setId(id);
		setName(xmlTaskData.getName());
		setType(xmlTaskData.getTaskType(), xmlTaskData.getLoopStructure());
		setTaskCodeFile(xmlTaskData.getFile());
		setParentTaskGraphName(xmlTaskData.getParentTask());
		setRunCondition(xmlTaskData.getRunCondition().toString());
		// setPeriod(xmlTaskData.get);
		
		//setType();
	}
	
	public int getId() {
		return id;
	}
	
	public void setId(int taskId) {
		this.id = taskId;
	}
	
	public String getName() {
		return name;
	}
	
	public void setName(String taskName) {
		this.name = taskName;
	}
	
	public TaskShapeType getType() {
		return type;
	}
	
	public void setType(String taskType, LoopStructureType loopStructure) {
		
		if(taskType.equalsIgnoreCase("Computational"))
		{
			if(loopStructure != null)
			{
				this.type = TaskShapeType.LOOP;
			}
			else 
			{
				this.type = TaskShapeType.COMPUTATIONAL;
			}
		}
		else if(taskType.equalsIgnoreCase("Control")) 
		{
			this.type = TaskShapeType.CONTROL;
		}
		
		//this.type = taskType;
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
		if(parentTaskGraphName.equals(this.name))
		{
			this.parentTaskGraphName = null;
		}
		else
		{
			this.parentTaskGraphName = parentTaskGraphName;	
		}
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

	public void setRunCondition(String runCondition) {
		this.runCondition = TaskRunCondition.valueOf(runCondition);
	}

	public String getTaskCodeFile() {
		return taskCodeFile;
	}

	public void setTaskCodeFile(String cicFile) {
		this.taskCodeFile = cicFile;
	}
}
