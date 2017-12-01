package org.snu.cse.cap.translator.structure.mapping;

public abstract class ScheduleItem {
	private int repetition;
	ScheduleItemType itemType;
	
	public ScheduleItem(ScheduleItemType itemType, int repetition) {
		this.itemType = itemType;
		this.repetition = repetition;
	}
	
	public int getRepetition() {
		return repetition;
	}
	
	public void setRepetition(int repetition) {
		this.repetition = repetition;
	}

	public ScheduleItemType getItemType() {
		return itemType;
	}
}
