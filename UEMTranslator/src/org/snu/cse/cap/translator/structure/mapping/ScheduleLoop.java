package org.snu.cse.cap.translator.structure.mapping;

import java.util.ArrayList;

public class ScheduleLoop extends ScheduleItem {
	private static final String variablePrefix = "depth";
	private ArrayList<ScheduleItem> scheduleItemList;
	private String variableName;

	public ScheduleLoop(int repetition, int depth) {
		super(ScheduleItemType.LOOP, repetition);
		this.scheduleItemList = new ArrayList<ScheduleItem>();
		this.variableName = variablePrefix + depth;
	}
	
	public String getVariableName() {
		return variableName;
	}

	public ArrayList<ScheduleItem> getScheduleItemList() {
		return scheduleItemList;
	}
	
	public void putScheduleTask(ScheduleTask scheduledTask) {
		this.scheduleItemList.add(scheduledTask);
	}
	
	public void putScheduleLoop(ScheduleLoop scheduledLoop) {
		this.scheduleItemList.add(scheduledLoop);
	}

}
