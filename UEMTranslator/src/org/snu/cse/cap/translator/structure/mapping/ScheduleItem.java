package org.snu.cse.cap.translator.structure.mapping;

public abstract class ScheduleItem {
	private int repetition;
	
	public ScheduleItem(int repetition) {
		this.repetition = repetition;
	}
	
	public int getRepetition() {
		return repetition;
	}
	
	public void setRepetition(int repetition) {
		this.repetition = repetition;
	}
}
