package org.snu.cse.cap.translator.structure.mapping;

import java.util.ArrayList;

public class ScheduleLoop extends ScheduleItem {
	private ArrayList<ScheduleItem> scheduleItems;

	public ScheduleLoop(int repetition) {
		super(ScheduleItemType.LOOP, repetition);
	}
	
	public ArrayList<ScheduleItem> getScheduleItemList() {
		return scheduleItems;
	}
	
	public void putScheduleTask(ScheduleTask scheduledTask) {
		this.scheduleItems.add(scheduledTask);
	}
	
	public void putScheduleLoop(ScheduleLoop scheduledLoop) {
		this.scheduleItems.add(scheduledLoop);
	}

}
