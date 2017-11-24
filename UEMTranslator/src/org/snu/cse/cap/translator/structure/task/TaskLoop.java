package org.snu.cse.cap.translator.structure.task;

enum LoopType {
	CONVERGENT("convergent"),
	DATA("data"),
	;
	
	private final String value;
	
	private LoopType(final String value) {
		this.value = value;
	}
	
	@Override
	public String toString() {
		return value;
	}
}

public class TaskLoop {
	private LoopType loopType;
	private int loopCount;
	private String designatedTaskName;
	
	public TaskLoop (String loopType, int loopCount) 
	{
		this.loopType = LoopType.valueOf(loopType);
		this.loopCount = loopCount;
		this.designatedTaskName = null;
	}
	
	public TaskLoop (String loopType, int loopCount, String designatedTaskName) 
	{
		this.loopType = LoopType.valueOf(loopType);
		this.loopCount = loopCount;
		this.designatedTaskName = designatedTaskName;
	}
	
	public LoopType getLoopType() {
		return loopType;
	}
	
	public void setLoopType(LoopType loopType) {
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
