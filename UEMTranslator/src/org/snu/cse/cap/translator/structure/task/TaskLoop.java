package org.snu.cse.cap.translator.structure.task;

import org.snu.cse.cap.translator.Constants;

public class TaskLoop {
	private TaskLoopType loopType;
	private int loopCount;
	private String designatedTaskName;
	private int designatedTaskId;
	
	public TaskLoop (String loopType, int loopCount) 
	{
		this.loopType = TaskLoopType.fromValue(loopType);
		this.loopCount = loopCount;
		this.designatedTaskName = null;
		this.designatedTaskId = Constants.INVALID_ID_VALUE;
	}
	
	public TaskLoop (String loopType, int loopCount, String designatedTaskName) 
	{
		this.loopType = TaskLoopType.fromValue(loopType);
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

	public int getDesignatedTaskId() {
		return designatedTaskId;
	}

	public void setDesignatedTaskId(int designatedTaskId) {
		this.designatedTaskId = designatedTaskId;
	}
	

}
