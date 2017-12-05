package org.snu.cse.cap.translator.structure.task;

public class TaskLoop {
	private TaskLoopType loopType;
	private int loopCount;
	private String designatedTaskName;
	
	public TaskLoop (String loopType, int loopCount) 
	{
		this.loopType = TaskLoopType.valueOf(loopType);
		this.loopCount = loopCount;
		this.designatedTaskName = null;
	}
	
	public TaskLoop (String loopType, int loopCount, String designatedTaskName) 
	{
		this.loopType = TaskLoopType.valueOf(loopType);
		this.loopCount = loopCount;
		this.designatedTaskName = designatedTaskName;
	}
	
	public TaskLoopType getLoopType() {
		return loopType;
	}
	
	public void setLoopType(TaskLoopType loopType) {
		this.loopType = loopType;
	}
	
	public int getLoopCount() {
		return loopCount;
	}
	
	public void setLoopCount(int loopCount) {
		this.loopCount = loopCount;
	}
	
	public String getDesignatedTaskName() {
		return designatedTaskName;
	}
	
	public void setDesignatedTaskName(String designatedTaskName) {
		this.designatedTaskName = designatedTaskName;
	}
	

}
