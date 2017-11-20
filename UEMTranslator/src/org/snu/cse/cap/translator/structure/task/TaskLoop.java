package org.snu.cse.cap.translator.structure.task;

enum LoopType {
	CONVERGENT,
	DATA,
}

public class TaskLoop {
	private LoopType loopType;
	private int loopCount;
	private String designatedTaskName;
	
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
