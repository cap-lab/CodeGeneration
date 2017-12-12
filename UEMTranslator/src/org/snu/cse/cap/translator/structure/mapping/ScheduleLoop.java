package org.snu.cse.cap.translator.structure.mapping;

import java.util.ArrayList;

public class ScheduleLoop extends ScheduleItem {
	private ArrayList<ScheduleItem> scheduleItemList;

	public ScheduleLoop(int repetition, int depth) {
		super(ScheduleItemType.LOOP, repetition, depth);
		this.scheduleItemList = new ArrayList<ScheduleItem>();
	}
	
	public ArrayList<ScheduleItem> getScheduleItemList() {
		return scheduleItemList;
	}
}
