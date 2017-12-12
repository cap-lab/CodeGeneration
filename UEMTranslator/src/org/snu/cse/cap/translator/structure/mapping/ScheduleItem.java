package org.snu.cse.cap.translator.structure.mapping;

public abstract class ScheduleItem {
	private static final String variablePrefix = "depth";
	private int repetition;
	private ScheduleItemType itemType;
	private String variableName;
	private int loopDepth;
	
	public ScheduleItem(ScheduleItemType itemType, int repetition, int depth) {
		this.itemType = itemType;
		this.repetition = repetition;
		this.variableName = variablePrefix + depth;
		this.loopDepth = depth;
		
	}
	
	public String getVariableName() {
		return variableName;
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
	
	public int getLoopDepth() {
		return loopDepth;
	}
}
