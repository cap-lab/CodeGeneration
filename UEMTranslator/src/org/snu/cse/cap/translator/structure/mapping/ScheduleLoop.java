package org.snu.cse.cap.translator.structure.mapping;

import java.util.ArrayList;

public class ScheduleLoop extends ScheduleItem {
	private ArrayList<ScheduleItem> scheduleItemList;

	public ScheduleLoop(int repetition) {
		super(ScheduleItemType.LOOP, repetition);
		this.scheduleItemList = new ArrayList<ScheduleItem>(); 
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
