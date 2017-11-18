package org.snu.cse.cap.translator.structure;

import java.util.ArrayList;

public class CompositeTaskSchedule {
	private int scheduleId;
	private ArrayList<CompositeTaskSchedule> scheduleLists;
	private int throughputConstraint;
	
	public int getScheduleId() {
		return scheduleId;
	}
	
	public void setScheduleId(int scheduleId) {
		this.scheduleId = scheduleId;
	}
	
	public int getThroughputConstraint() {
		return throughputConstraint;
	}
	
	public void setThroughputConstraint(int throughputConstraint) {
		this.throughputConstraint = throughputConstraint;
	}
}
